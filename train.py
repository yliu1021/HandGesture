import os
import time
import sys
import glob

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule, pruning_callbacks
import matplotlib.pyplot as plt

import model
import data
from constants import *


def temporal_crossentropy(y_true, y_pred):
    y_pred = y_pred[:, -5:]
    avg_pred = tf.reduce_mean(y_pred[:, -5:], axis=1)
    return tf.nn.softmax_cross_entropy_with_logits(y_true, avg_pred)


def temporal_accuracy(y_true, y_pred):
    y_pred = y_pred[:, -5:]
    avg_pred = tf.reduce_mean(y_pred, axis=1)

    class_id_true = tf.argmax(y_true, axis=-1)
    class_id_pred = tf.argmax(avg_pred, axis=-1)
    acc = tf.cast(tf.equal(class_id_true, class_id_pred), 'int32')
    acc = tf.reduce_sum(acc) / BATCH_SIZE
    return acc


def temporal_top_k_accuracy(y_true, y_pred):
    y_pred = y_pred[:, -5:]
    avg_pred = tf.reduce_mean(y_pred, axis=1)
    return tf.keras.metrics.sparse_top_k_categorical_accuracy(tf.argmax(y_true, axis=-1), avg_pred, k=2)


def main(should_prune=False):

    def yield_from_generator(g):
        def callable_generator():
            yield from g
        return callable_generator

    training_dir = os.path.join('./training', TRAINING_RUN)
    tensorboard_dir = os.path.join(training_dir, 'logs')
    pruning_dir = os.path.join(training_dir, 'pruning')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(pruning_dir, exist_ok=True)
    single_frame_encoder_model_save_dir = os.path.join(training_dir, 'single_frame_encoder.hdf5')
    multi_frame_encoder_model_save_dir = os.path.join(training_dir, 'multi_frame_model.hdf5')
    multi_frame_encoder_weight_model_save_dir = os.path.join(training_dir, 'multi_frame_model_weights.hdf5')
    starting_epoch = 0

    single_frame_encoder = model.single_frame_model()
    multi_frame_model = model.multi_frame_model(single_frame_encoder, num_frames=None)
    single_frame_encoder.summary()
    multi_frame_model.summary()

    previous_saves = sorted(glob.glob(os.path.join(training_dir, 'multi_frame_model.??.hdf5')))
    if len(previous_saves) != 0:
        last_save = previous_saves[-1]
        starting_epoch = int(last_save.split('.')[-2])
        print(f'Restoring weights from last run: {last_save}')
        multi_frame_model.load_weights(last_save)

    train_data_generator = data.train_dataset.data_generator(
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    validation_data_generator = data.validation_dataset.data_generator(
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    sample_batch = next(train_data_generator)
    print(f'Testing speed on batch size {BATCH_SIZE}')
    start = time.time()
    multi_frame_model.predict_on_batch(sample_batch)
    end = time.time()
    print(f'Predicted in {end - start}, {(end - start) / BATCH_SIZE} per sample')

    train_data_generator = tf.data.Dataset.from_generator(
        yield_from_generator(train_data_generator),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([BATCH_SIZE, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
                       tf.TensorShape([BATCH_SIZE, NUM_CLASSES]))
    )
    validation_data_generator = tf.data.Dataset.from_generator(
        yield_from_generator(validation_data_generator),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([BATCH_SIZE, NUM_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
                       tf.TensorShape([BATCH_SIZE, NUM_CLASSES]))
    )

    if should_prune:
        pruning_params = {
            'pruning_schedule': pruning_schedule.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.90,
                                                                 begin_step=PRUNING_START_EPOCH,
                                                                 end_step=PRUNING_END_EPOCH,
                                                                 frequency=PRUNE_FREQ)
        }
        multi_frame_model = prune.prune_low_magnitude(multi_frame_model, **pruning_params)

    optimizer = Adam(LEARNING_RATE)
    multi_frame_model.compile(
        optimizer=optimizer,
        loss=temporal_crossentropy,
        metrics=[temporal_accuracy, temporal_top_k_accuracy]
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_temporal_accuracy', factor=0.1, patience=10, mode='max'),
        EarlyStopping(monitor='val_temporal_accuracy', patience=11, mode='max'),
        ModelCheckpoint(filepath=os.path.join(training_dir, 'multi_frame_model.{epoch:02d}.hdf5')),
        TensorBoard(log_dir=tensorboard_dir, histogram_freq=2, write_images=True)
    ]

    if should_prune:
        callbacks.extend([
            pruning_callbacks.UpdatePruningStep(),
            pruning_callbacks.PruningSummaries(log_dir=pruning_dir)
        ])

    hist = multi_frame_model.fit(
        train_data_generator,
        steps_per_epoch=data.train_dataset.num_samples() // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=validation_data_generator,
        validation_steps=VALIDATION_STEPS,
        max_queue_size=10,
        workers=16,
        use_multiprocessing=True,
        shuffle=True,
        initial_epoch=starting_epoch,
    )

    if should_prune:
        multi_frame_model = prune.strip_pruning(multi_frame_model)

    single_frame_encoder.save(single_frame_encoder_model_save_dir)
    multi_frame_model.save(multi_frame_encoder_model_save_dir)
    multi_frame_model.save(multi_frame_encoder_weight_model_save_dir, include_optimizer=False)

    # Plot training & validation accuracy values
    plt.plot(hist.history['temporal_accuracy'])
    plt.plot(hist.history['val_temporal_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main(should_prune='prune' in sys.argv)

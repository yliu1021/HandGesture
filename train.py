import os
import time
import sys
import glob

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_schedule, pruning_callbacks
import matplotlib.pyplot as plt

from cosine_warmup_lr import WarmUpCosineDecayScheduler
import model
import data
from constants import *


def temporal_avg(y_pred):
    # y_pred = y_pred[:, -5:]
    avg_pred = tf.reduce_mean(y_pred, axis=1)
    return avg_pred


def temporal_crossentropy(y_true, y_pred):
    avg_pred = temporal_avg(y_pred)
    return tf.nn.softmax_cross_entropy_with_logits(y_true, avg_pred)


def temporal_accuracy(y_true, y_pred):
    avg_pred = temporal_avg(y_pred)
    class_id_true = tf.argmax(y_true, axis=-1)
    class_id_pred = tf.argmax(avg_pred, axis=-1)
    acc = tf.cast(tf.equal(class_id_true, class_id_pred), 'int32')
    acc = tf.reduce_sum(acc) / BATCH_SIZE
    return acc


def temporal_top_k_accuracy(y_true, y_pred):
    avg_pred = temporal_avg(y_pred)
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
    single_frame_encoder_model_save_dir = os.path.join(training_dir, 'single_frame_encoder.h5')
    multi_frame_encoder_model_save_dir = os.path.join(training_dir, 'multi_frame_model.h5')
    full_model_save_dir = os.path.join(training_dir, 'full_model.h5')
    starting_epoch = 0

    single_frame_encoder = model.single_frame_model()
    multi_frame_model = model.multi_frame_model(num_frames=None)

    if should_prune:
        pruning_params = {
            'pruning_schedule': pruning_schedule.PolynomialDecay(initial_sparsity=0.2,
                                                                 final_sparsity=0.8,
                                                                 begin_step=PRUNING_START_EPOCH,
                                                                 end_step=PRUNING_END_EPOCH,
                                                                 frequency=PRUNE_FREQ)
        }
        single_frame_encoder = prune.prune_low_magnitude(single_frame_encoder, **pruning_params)
        multi_frame_model = prune.prune_low_magnitude(multi_frame_model, **pruning_params)

    full_model = model.full_model(single_frame_encoder, multi_frame_model, num_frames=None)
    single_frame_encoder.summary()
    multi_frame_model.summary()

    previous_saves = sorted(glob.glob(os.path.join(training_dir, 'full_model.??.h5')))
    if len(previous_saves) != 0:
        last_save = previous_saves[-1]
        starting_epoch = int(last_save.split('.')[-2])
        print(f'Restoring weights from last run: {last_save}')
        full_model.load_weights(last_save)

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
    full_model.predict_on_batch(sample_batch)
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

    optimizer = SGD(LEARNING_RATE, momentum=0.9, nesterov=True)
    full_model.compile(
        optimizer=optimizer,
        loss=temporal_crossentropy,
        metrics=[temporal_accuracy, temporal_top_k_accuracy]
    )

    steps_per_epoch = data.train_dataset.num_samples() // BATCH_SIZE
    callbacks = [
        ReduceLROnPlateau(monitor='val_temporal_accuracy', factor=0.1, patience=10, mode='max'),
        EarlyStopping(monitor='val_temporal_accuracy', patience=15, mode='max'),
        ModelCheckpoint(filepath=os.path.join(training_dir, 'full_model.{epoch:02d}.h5')),
        TensorBoard(log_dir=tensorboard_dir, histogram_freq=2, write_images=True),
        WarmUpCosineDecayScheduler(LEARNING_RATE, total_steps=steps_per_epoch, warmup_learning_rate=0.0001,
                                   warmup_steps=100, hold_base_rate_steps=10)
    ]

    if should_prune:
        callbacks.extend([
            pruning_callbacks.UpdatePruningStep(),
            pruning_callbacks.PruningSummaries(log_dir=pruning_dir)
        ])

    hist = full_model.fit(
        train_data_generator,
        steps_per_epoch=steps_per_epoch,
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
        single_frame_encoder = prune.strip_pruning(single_frame_encoder)
        multi_frame_model = prune.strip_pruning(multi_frame_model)
        full_model = prune.strip_pruning(full_model)

    single_frame_encoder.save(single_frame_encoder_model_save_dir)
    multi_frame_model.save(multi_frame_encoder_model_save_dir)
    full_model.save(full_model_save_dir)

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

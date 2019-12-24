import os
import time

import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import matplotlib.pyplot as plt

import model
import data
from constants import *

single_frame_encoder = model.single_frame_model()
multi_frame_model = model.multi_frame_model(single_frame_encoder)
multi_frame_model.summary()
num_output_frames = multi_frame_model.output_shape[1]
print(f'Output frames: {num_output_frames}')


def temporal_crossentropy(y_true, y_pred):
    avg_pred = tf.reduce_mean(y_pred, axis=1)
    return tf.nn.softmax_cross_entropy_with_logits(y_true, avg_pred)


def temporal_accuracy(y_true, y_pred):
    avg_pred = tf.reduce_mean(y_pred, axis=1)

    class_id_true = tf.argmax(y_true, axis=-1)
    class_id_pred = tf.argmax(avg_pred, axis=-1)
    acc = tf.cast(tf.equal(class_id_true, class_id_pred), 'int32')
    acc = tf.reduce_sum(acc) / BATCH_SIZE
    return acc


def main():

    def yield_from_generator(g):
        def callable_generator():
            yield from g
        return callable_generator

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

    optimizer = Adam(LEARNING_RATE)
    multi_frame_model.compile(
        optimizer=optimizer,
        loss=temporal_crossentropy,
        metrics=[temporal_accuracy]
    )

    hist = multi_frame_model.fit(
        train_data_generator,
        steps_per_epoch=data.train_dataset.num_samples() // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[
            ReduceLROnPlateau(monitor='val_temporal_accuracy', factor=0.1, patience=10, mode='max'),
            EarlyStopping(monitor='val_temporal_accuracy', patience=11, mode='max')
        ],
        validation_data=validation_data_generator,
        validation_steps=VALIDATION_STEPS,
        max_queue_size=10,
        workers=15,
        use_multiprocessing=True,
        shuffle=True,
        initial_epoch=0,
    )

    training_dir = './training'
    single_frame_encoder_model_save_dir = os.path.join(training_dir, 'single_frame_encoder.hdf5')
    single_frame_encoder.save(single_frame_encoder_model_save_dir)
    multi_frame_encoder_model_save_dir = os.path.join(training_dir, 'multi_frame_encoder.hdf5')
    multi_frame_model.save(multi_frame_encoder_model_save_dir)

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
    main()

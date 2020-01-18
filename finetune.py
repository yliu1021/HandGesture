import sys
import time
import glob
import os

import tensorflow as tf
from tensorflow.keras.optimizers import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import model
from constants import *


BATCH_SIZE = 64
GESTURES = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things"
]
FRAME_TIME = 1 / MIN_FPS


def temporal_avg(y_pred):
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


def gather():
    print('Collecting fine tuning data...')
    cap = cv2.VideoCapture(0)
    x = list()
    y = list()
    while True:
        menu = map(lambda x: f'[{x[0]}] {x[1]}', enumerate(GESTURES))
        print('\n'.join(menu))
        try:
            i = int(input('Pick a gesture to record: '))
        except ValueError:
            break
        if 0 <= i < len(GESTURES):
            g = GESTURES[i]
        else:
            if i == -2:
                print('Removing last sample')
                x.pop()
                y.pop()
                continue
            else:
                print('Exiting')
                break
        print('Training gesture {}'.format(g))
        time.sleep(1)
        print('Start')
        image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

        label = np.zeros(27, dtype=np.float32)
        label[i] = 1.0
        frames = list()
        num_frames = 10
        for f in range(num_frames):
            print('\r{}/{}'.format(f+1, num_frames), flush=True, end='')
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Couldn't fetch camera frame")
                time.sleep(1)
                continue
            frame = cv2.resize(frame, image_size)
            frame = frame.astype(np.float32) / 255.0
            cv2.imshow('frame', frame)
            frames.append(frame)
            end = time.time()
            wait = max(.001, FRAME_TIME - (end - start))
            cv2.waitKey(int(wait * 1000))
        print()
        frames = np.array(frames)
        x.append(frames)
        y.append(label)

    cv2.destroyAllWindows()
    x = np.array(x)
    y = np.array(y)
    data_saves = glob.glob('fine_tune_data/data*.npz')
    save_num = len(data_saves) + 1
    np.savez(f'fine_tune_data/data{save_num}.npz', x=x, y=y)


def train(epochs, prev_data):
    if prev_data is None:
        data_saves = sorted(glob.glob('fine_tune_data/data*.npz'))
    else:
        data_saves = sorted(glob.glob('fine_tune_data/data*.npz'))[-prev_data:]
    x = list()
    y = list()
    for data_save in data_saves:
        data = np.load(data_save)
        x.append(data['x'])
        y.append(data['y'])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    single_frame_encoder = model.single_frame_model()
    multi_frame_encoder = model.multi_frame_model(num_frames=None)
    full_model = model.full_model(single_frame_encoder, multi_frame_encoder, num_frames=None)
    training_run = 'run9'
    single_frame_encoder_loc = os.path.join('./inference', training_run, 'single_frame.h5')
    multi_frame_encoder_loc = os.path.join('./inference', training_run, 'multi_frame.h5')
    single_frame_encoder.load_weights(single_frame_encoder_loc, by_name=True)
    multi_frame_encoder.load_weights(multi_frame_encoder_loc, by_name=True)
    
    for layer in single_frame_encoder.layers:
        if layer.name in ['batch_normalization_12', 'batch_normalization_13', 'batch_normalization_14']:
            layer.trainable = True
        else:
            layer.trainable = False
    for layer in multi_frame_encoder.layers:
        if layer.name == 'dense_1':
            layer.trainable = True
        else:
            layer.trainable = False

    full_model.compile(optimizer=SGD(learning_rate=1e-3, momentum=0.9, nesterov=True),
                       loss=temporal_crossentropy,
                       metrics=[temporal_accuracy, temporal_top_k_accuracy])
    full_model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, shuffle=True)
    
    single_frame_encoder.save(single_frame_encoder_loc)
    multi_frame_encoder.save(multi_frame_encoder_loc)
    del single_frame_encoder
    del multi_frame_encoder
    del full_model
    single_frame_encoder = model.single_frame_model()
    multi_frame_encoder = model.multi_frame_model(num_frames=7)
    full_model = model.full_model(single_frame_encoder, multi_frame_encoder, num_frames=8)
    training_run = 'run9'
    single_frame_encoder_loc = os.path.join('./inference', training_run, 'single_frame.h5')
    multi_frame_encoder_loc = os.path.join('./inference', training_run, 'multi_frame.h5')
    single_frame_encoder.load_weights(single_frame_encoder_loc, by_name=True)
    multi_frame_encoder.load_weights(multi_frame_encoder_loc, by_name=True)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(single_frame_encoder)
    converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print('Converting to tf lite...')
    tflite_model = converter.convert()
    print('Converted to tf lite')

    inference_dir = './inference'
    tflite_model_loc = os.path.join(inference_dir, training_run)
    os.makedirs(tflite_model_loc, exist_ok=True)
    tflite_model_loc = os.path.join(tflite_model_loc, 'single_frame_model.tflite')
    print('Saving tf lite')
    try:
        with open(tflite_model_loc, 'wb+') as f:
            f.write(tflite_model)
    except Exception as e:
        print(f'Unable to save: {e}')
    else:
        print('Saved tf lite')
    
    converter = tf.lite.TFLiteConverter.from_keras_model(multi_frame_encoder)
    converter.post_training_quantize = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    print('Converting to tf lite...')
    tflite_model = converter.convert()
    print('Converted to tf lite')

    inference_dir = './inference'
    tflite_model_loc = os.path.join(inference_dir, training_run)
    os.makedirs(tflite_model_loc, exist_ok=True)
    tflite_model_loc = os.path.join(tflite_model_loc, 'multi_frame_model.tflite')
    print('Saving tf lite')
    try:
        with open(tflite_model_loc, 'wb+') as f:
            f.write(tflite_model)
    except Exception as e:
        print(f'Unable to save: {e}')
    else:
        print('Saved tf lite')


def evaluate():
    data_saves = sorted(glob.glob('fine_tune_data/data*.npz'))
    x = list()
    y = list()
    for data_save in data_saves:
        data = np.load(data_save)
        x.append(data['x'])
        y.append(data['y'])
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    total_samples = y.sum(axis=0).astype(np.int)
    for gesture, num_samples in zip(GESTURES, total_samples):
        print('{} : {}'.format(gesture, num_samples))

    single_frame_encoder = model.single_frame_model()
    multi_frame_encoder = model.multi_frame_model(num_frames=None)
    full_model = model.full_model(single_frame_encoder, multi_frame_encoder, num_frames=None)
    training_run = 'run9'
    single_frame_encoder_loc = os.path.join('./inference', training_run, 'single_frame.h5')
    multi_frame_encoder_loc = os.path.join('./inference', training_run, 'multi_frame.h5')
    single_frame_encoder.load_weights(single_frame_encoder_loc, by_name=True)
    multi_frame_encoder.load_weights(multi_frame_encoder_loc, by_name=True)

    y_pred = full_model.predict(x)
    y_pred = np.average(y_pred, axis=1)
    conf_matrix = tf.math.confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1), num_classes=27).numpy()
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix, annot=True)
    plt.title('Confusion Matrix')
    plt.yticks(np.arange(len(GESTURES)), GESTURES, rotation='horizontal')
    plt.xticks(np.arange(len(GESTURES)), GESTURES, rotation='vertical')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python finetune.py [train|gather|evaluate]')
        exit(1)
    if sys.argv[1].lower() == 'train':
        try:
            epochs = int(sys.argv[2])
        except:
            epochs = 10
        try:
            prev_data = int(sys.argv[3])
        except:
            prev_data = None
        train(epochs=epochs, prev_data=prev_data)
    elif sys.argv[1].lower() == 'gather':
        gather()
    elif sys.argv[1].lower() == 'evaluate':
        evaluate()
    else:
        print('Unrecognized argument')

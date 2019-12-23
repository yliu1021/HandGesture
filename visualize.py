import os
import argparse
import random

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model as keras_load_model
import numpy as np
import cv2

import data
import model
import train
from constants import *


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load_model(path=None):
    if path is None:
        return model.single_frame_model()
    else:
        if os.path.exists(path):
            return keras_load_model(path, custom_objects={'temporal_crossentropy': train.temporal_crossentropy,
                                                          'temporal_accuracy': train.temporal_accuracy})
        else:
            raise FileNotFoundError(f"Model path {path} doesn't exist")


def main(model_path=None):
    np.set_printoptions(precision=3, linewidth=200, floatmode='fixed', suppress=True)

    print(f'Loaded model from {model_path}: ')
    single_frame_encoder = load_model(os.path.join(model_path, 'single_frame_encoder.hdf5'))
    single_frame_encoder.summary()
    multi_frame_encoder = load_model(os.path.join(model_path, 'multi_frame_encoder.hdf5'))
    multi_frame_encoder.summary()

    arr_ind_x = 0
    arr_ind_y = 1

    def get_xys(sample):
        prediction = single_frame_encoder.predict(sample)
        xs = list()
        ys = list()
        for p in prediction:
            xs.append(p[arr_ind_x])
            ys.append(p[arr_ind_y])
        xs = np.array(xs)
        ys = np.array(ys)
        xs = np.diff(xs, prepend=xs[0])
        ys = np.diff(ys, prepend=ys[0])
        return xs, ys

    while True:
        class1 = random.randint(0, NUM_CLASSES - 1)
        class2 = random.randint(0, NUM_CLASSES - 1)
        print(f'Visualizing classes: {data.labels[class1]} ({class1+1}), {data.labels[class2]} ({class2+1})')

        sample1 = data.train_dataset.sample_from_class(class1, frame_samples=NUM_FRAMES)
        sample2 = data.train_dataset.sample_from_class(class1, frame_samples=NUM_FRAMES)
        sample3 = data.train_dataset.sample_from_class(class2, frame_samples=NUM_FRAMES)

        print('Visualizing dims: ', arr_ind_x, arr_ind_y)
        sample1_pred = multi_frame_encoder.predict(np.expand_dims(sample1, axis=0))
        sample2_pred = multi_frame_encoder.predict(np.expand_dims(sample2, axis=0))
        sample3_pred = multi_frame_encoder.predict(np.expand_dims(sample3, axis=0))
        sample1_pred = softmax(sample1_pred)
        sample2_pred = softmax(sample2_pred)
        sample3_pred = softmax(sample3_pred)
        print(np.argmax(sample1_pred, axis=-1)+1, np.max(sample1_pred, axis=-1))
        print(np.argmax(sample2_pred, axis=-1)+1, np.max(sample2_pred, axis=-1))
        print(np.argmax(sample3_pred, axis=-1)+1, np.max(sample3_pred, axis=-1))
        xs1, ys1 = get_xys(sample1)
        xs2, ys2 = get_xys(sample2)
        xs3, ys3 = get_xys(sample3)
        arr_ind_x = random.randint(0, SINGLE_FRAME_ENCODER_DIMS - 1)
        arr_ind_y = random.randint(0, SINGLE_FRAME_ENCODER_DIMS - 1)

        plt.plot(xs1, ys1, label=f'prediction class {class1}')
        plt.scatter(xs1, ys1)
        plt.plot(xs2, ys2, label=f'prediction class {class1}')
        plt.scatter(xs2, ys2)
        plt.plot(xs3, ys3, label=f'prediction class {class2}')
        plt.scatter(xs3, ys3)

        plt.axhline(color='black', linestyle='dotted')
        plt.axvline(color='black', linestyle='dotted')

        plt.legend()
        plt.show()

        for img1, img2, img3 in zip(sample1, sample2, sample3):
            cv2.imshow('class 1', img1)
            cv2.imshow('class 1_2', img2)
            cv2.imshow('class 2', img3)
            cv2.waitKey(int(3000 / NUM_FRAMES))

        input('Enter to continue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the training process')
    parser.add_argument('--model', nargs='?', type=str, help='the location of the model to load')
    args = parser.parse_args()
    main(model_path=args.model)

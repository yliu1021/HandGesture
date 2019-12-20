import os
import argparse

import matplotlib.pyplot as plt
import tensorflow.keras
import cv2

import data
import model
from constants import *


def load_model(path=None):
    if path is None:
        return model.single_frame_model()
    else:
        if os.path.exists(path):
            return load_model(path)
        else:
            raise FileNotFoundError(f"Model path {path} doesn't exist")


def main(model_path=None):
    single_frame_encoder = load_model(model_path)
    print('Loaded model: ')
    single_frame_encoder.summary()

    def get_xys(sample):
        prediction = single_frame_encoder.predict(sample)
        xs = list()
        ys = list()
        for p in prediction:
            xs.append(p[0])
            ys.append(p[1])
        xs = [x - xs[0] for x in xs]
        ys = [y - ys[0] for y in ys]
        return xs, ys

    while True:
        class1 = 0
        class2 = 1

        sample1 = data.train_dataset.sample_from_class(class1, frame_samples=EXPERIENCE_TRAJECTORY_SAMPLES)
        sample2 = data.train_dataset.sample_from_class(class1, frame_samples=EXPERIENCE_TRAJECTORY_SAMPLES)
        sample3 = data.train_dataset.sample_from_class(class2, frame_samples=EXPERIENCE_TRAJECTORY_SAMPLES)
        xs1, ys1 = get_xys(sample1)
        xs2, ys2 = get_xys(sample2)
        xs3, ys3 = get_xys(sample3)

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
            cv2.waitKey(int(3000 / EXPERIENCE_TRAJECTORY_SAMPLES))

        input('Enter to continue')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the training process')
    parser.add_argument('--model', nargs='?', type=str, help='the location of the model to load')
    args = parser.parse_args()
    main(model_path=args.model)

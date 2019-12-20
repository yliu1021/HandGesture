import os
import glob
import random
from collections import defaultdict

import cv2
import numpy as np

from constants import *


dataset_dir = '/home/yuhan/Datasets/jester/'
video_dir = os.path.join(dataset_dir, '20bn-jester-v1')
labels_filename = os.path.join(dataset_dir, 'jester-v1-labels.csv')
train_filename = os.path.join(dataset_dir, 'jester-v1-train.csv')
validation_filename = os.path.join(dataset_dir, 'jester-v1-validation.csv')
with open(labels_filename, 'r') as labels_file:
    labels = [label.strip() for label in labels_file.readlines()]


class DataSet:
    def __init__(self, data_filename):
        self.data = defaultdict(set)
        with open(data_filename, 'r') as data_file:
            for line in data_file:
                dir_num, label = line.strip().split(';')
                self.data[label].add(dir_num)

    def labels(self):
        return set(self.data.keys())

    def sample_from_class(self, label_class, frame_samples=None):
        if type(label_class) is int:
            label_class = labels[label_class]
        if type(frame_samples) is int:
            frame_samples = self.generate_frame_samples(frame_samples)
        dir_nums = self.data[label_class]
        dir_num = random.choice(tuple(dir_nums))
        images = self.load_images(dir_num, frame_samples)
        return images

    @staticmethod
    def generate_frame_samples(num_frames=10):
        return [random.random() for _ in range(num_frames)]

    @staticmethod
    def load_images(dir_num, frame_samples):
        if type(dir_num) is not str:
            dir_num = str(dir_num)
        image_files = sorted(glob.glob(os.path.join(video_dir, dir_num, '*.jpg')))
        if frame_samples is not None:
            frame_indices = sorted(list(map(lambda x: int(x * len(image_files)), frame_samples)))
            image_files = [image_files[i] for i in frame_indices]
        images = [cv2.imread(image_file) for image_file in image_files]
        images = [cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)) for image in images]
        return np.array(images)


train_dataset = DataSet(train_filename)
validation_dataset = DataSet(validation_filename)


def main():
    frame_samples = train_dataset.generate_frame_samples(10)
    for _ in range(30):
        images_sample1 = train_dataset.sample_from_class(0, frame_samples=frame_samples)
        images_sample2 = train_dataset.sample_from_class(0, frame_samples=frame_samples)
        for image1, image2 in zip(images_sample1, images_sample2):
            cv2.imshow('image1', image1)
            cv2.imshow('image2', image2)
            cv2.waitKey(150)
        while True:
            key_press = cv2.waitKey(10)
            if key_press == ord(' '):
                break
            elif key_press == ord('q'):
                exit(0)
            continue


if __name__ == '__main__':
    main()

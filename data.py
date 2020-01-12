import os
import glob
import random
from collections import defaultdict
import time

import cv2
import numpy as np

from constants import *

dataset_dir = '/home/yuhan/Datasets/jester/'
if not os.path.isdir(dataset_dir):
    dataset_dir = '/Users/yuhanliu/Offline/Datasets/20bn_Jester/'
video_dir = os.path.join(dataset_dir, '20bn-jester-v1')
video_cache_dir = os.path.join(dataset_dir, 'cached')
labels_filename = os.path.join(dataset_dir, 'jester-v1-labels.csv')
train_filename = os.path.join(dataset_dir, 'jester-v1-train.csv')
validation_filename = os.path.join(dataset_dir, 'jester-v1-validation.csv')
with open(labels_filename, 'r') as labels_file:
    labels = [label.strip() for label in labels_file.readlines()]


class DataSet:
    def __init__(self, data_filename, augment_images):
        self.augment_images = augment_images
        self.categorized_data = defaultdict(set)
        self.raw_data = list()
        with open(data_filename, 'r') as data_file:
            for line in data_file:
                dir_num, label = line.strip().split(';')
                one_hot_label = self.one_hot(label)
                self.raw_data.append((dir_num, one_hot_label))
                self.categorized_data[label].add(dir_num)

    def labels(self):
        return set(self.categorized_data.keys())

    def sample_from_class(self, label_class, frame_samples=None):
        if type(label_class) is int:
            label_class = labels[label_class]
        if type(frame_samples) is int:
            frame_samples = self.generate_frame_samples(frame_samples)
        dir_nums = self.categorized_data[label_class]
        dir_num = random.choice(tuple(dir_nums))
        images = self.load_images(dir_num, frame_samples)
        return images

    def data_generator(self, num_frames, batch_size, shuffle=False):
        if self.augment_images:
            load_func = self.load_images_augment
        else:
            load_func = self.load_images
        while True:
            if shuffle:
                random.shuffle(self.raw_data)
            batched_data = self.batch(self.raw_data, batch_size)
            for batch in batched_data:
                frame_samples = self.generate_frame_samples(num_frames)
                video_batch = list()
                one_hot_label_batch = list()
                for dir_num, one_hot_label in batch:
                    images = load_func(dir_num, frame_samples)
                    video_batch.append(images)
                    one_hot_label_batch.append(one_hot_label)
                video_batch = np.array(video_batch)
                one_hot_label_batch = np.array(one_hot_label_batch)
                yield video_batch, one_hot_label_batch

    def num_samples(self):
        return len(self.raw_data)

    @staticmethod
    def batch(data, batch_size):
        new_data = list()
        for i in range(0, len(data) - batch_size, batch_size):
            new_data.append(data[i:i + batch_size])
        return new_data

    @staticmethod
    def one_hot(label):
        ind = labels.index(label)
        new_label = np.zeros(len(labels), dtype='float32')
        new_label[ind] = 1.0
        return new_label

    _cached_random_number_source = list()

    @staticmethod
    def generate_frame_samples(num_frames=10):
        # returns `num_frames` number of random numbers in [0, 1)
        return np.sort(np.random.rand(num_frames))

    @staticmethod
    def load_images_augment(dir_num, frame_samples):
        if type(dir_num) is not str:
            dir_num = str(dir_num)
        image_files = sorted(glob.glob(os.path.join(video_dir, dir_num, '*.jpg')))
        num_files = len(image_files)
        if frame_samples is not None:
            # frames are 12 fps or 1/12 = 0.0833 seconds apart
            # that means on average, we must pick frames that are
            # 12 / AVG_FPS frames apart
            avg_ind_diff = np.diff(frame_samples).mean() * num_files
            scale = max(avg_ind_diff / (12 / MIN_FPS), 1)
            frame_samples /= scale
            max_increase = 1 - frame_samples.max()
            des_inc = 0.5 - frame_samples.mean()
            inc = min(max(random.gauss(des_inc, 0.01), 0), max_increase)
            frame_samples += inc
            frame_indices = (frame_samples * (num_files - 1)).astype(np.int)
            image_files = [image_files[i] for i in frame_indices]
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        images = np.array([cv2.resize(cv2.imread(x), img_size) for x in image_files]) / 255.0
        images = images * random.gauss(1, 0.2) + random.gauss(0, 0.2)
        images = np.clip(images, 0, 1)
        return images

    @staticmethod
    def load_images(dir_num, frame_samples):
        if type(dir_num) is not str:
            dir_num = str(dir_num)
        image_files = sorted(glob.glob(os.path.join(video_dir, dir_num, '*.jpg')))
        num_files = len(image_files)
        if frame_samples is not None:
            # frames are 12 fps or 1/12 = 0.0833 seconds apart
            # that means on average, we must pick frames that are
            # 12 / AVG_FPS frames apart
            avg_ind_diff = np.diff(frame_samples).mean() * num_files
            scale = max(avg_ind_diff / (12 / MIN_FPS), 1)
            frame_samples /= scale
            max_increase = 1 - frame_samples.max()
            des_inc = 0.5 - frame_samples.mean()
            inc = min(max(random.gauss(des_inc, 0.01), 0), max_increase)
            frame_samples += inc
            frame_indices = (frame_samples * (num_files - 1)).astype(np.int)
            image_files = [image_files[i] for i in frame_indices]
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        images = np.array([cv2.resize(cv2.imread(x), img_size) for x in image_files]) / 255.0
        return images


train_dataset = DataSet(train_filename, augment_images=True)
validation_dataset = DataSet(validation_filename, augment_images=False)


def cv2_keyboard_block():
    while True:
        key_press = cv2.waitKey(10)
        if key_press == ord(' '):
            break
        elif key_press == ord('q'):
            exit(0)
        continue


def main():
    np.set_printoptions(linewidth=100)
    for label in labels:
        print(DataSet.one_hot(label), label)
    dataset = train_dataset
    for batch in dataset.data_generator(num_frames=NUM_FRAMES, batch_size=64, shuffle=True):
        print('Got batch')
        videos, one_hot_labels = batch
        for video, one_hot_label in zip(videos, one_hot_labels):
            print(one_hot_label, labels[one_hot_label.argmax()])
            for image in video:
                cv2.imshow('video', image)
                cv2.waitKey(int(1 / MIN_FPS * 1000))
            cv2_keyboard_block()

    frame_samples = train_dataset.generate_frame_samples(10)
    for _ in range(30):
        images_sample1 = train_dataset.sample_from_class(0, frame_samples=frame_samples)
        images_sample2 = train_dataset.sample_from_class(0, frame_samples=frame_samples)
        for image1, image2 in zip(images_sample1, images_sample2):
            cv2.imshow('image1', image1)
            cv2.imshow('image2', image2)
            cv2.waitKey(150)
        cv2_keyboard_block()


if __name__ == '__main__':
    main()

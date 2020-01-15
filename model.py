import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *

from constants import *


def tf_diff(x):
    return x[:, 1:] - x[:, :-1]


def stem(x):
    x = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)

    b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    b2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = Concatenate(axis=-1)([b1, b2])
    x = BatchNormalization(renorm=True)(x)

    b1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b1 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b1)
    b2 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(b2)
    b2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(b2)
    b2 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b2)
    x = Concatenate(axis=-1)([b1, b2])
    x = BatchNormalization(renorm=True)(x)

    b1 = Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    b2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Concatenate(axis=-1)([b1, b2])
    return x


def blockA(x):
    res1 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)

    res2 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)
    res2 = SeparableConv2D(filters=32, kernel_size=3, activation='relu', padding='same')(res2)

    res3 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)
    res3 = SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same')(res3)
    res3 = SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same')(res3)

    res = Concatenate(axis=-1)([res1, res2, res3])
    res = Conv2D(filters=384, kernel_size=1, activation=None, padding='same')(res)
    res = BatchNormalization(renorm=True)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.15)([x, res])
    return x


def blockB(x):
    res1 = Conv2D(filters=192, kernel_size=1, activation='relu', padding='same')(x)

    res2 = Conv2D(filters=128, kernel_size=1, activation='relu', padding='same')(x)
    res2 = SeparableConv2D(filters=160, kernel_size=(7, 1), activation='relu', padding='same')(res2)
    res2 = SeparableConv2D(filters=192, kernel_size=(1, 7), activation='relu', padding='same')(res2)

    res = Concatenate(axis=-1)([res1, res2])
    res = Conv2D(filters=1152, kernel_size=1, activation=None, padding='same')(res)
    res = BatchNormalization(renorm=True)(res)
    
    x = Lambda(lambda a: a[0] + a[1]*0.1)([x, res])
    return x


def reductionA(x):
    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(b3)
    b3 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(b3)

    x = Concatenate(axis=-1)([b1, b2, b3])
    x = BatchNormalization(renorm=True)(x)
    return x


def reductionB(x):
    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b2 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(b2)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b3)

    b4 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(b4)
    b4 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b4)

    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    x = BatchNormalization(renorm=True)(x)
    return x


def single_frame_model():
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frame_input = Input(shape=input_shape)

    x = stem(frame_input)

    for i in range(4):
        x = blockA(x)
    
    x = reductionA(x)

    for i in range(5):
        x = blockB(x)

    x = reductionB(x)

    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(num_frames=None):
    encoded_frame_input = Input(shape=(num_frames, 4 * 6 * 2048))

    x = Dense(512, activation='relu')(encoded_frame_input)
    filter_sizes = [512, 512, 512]
    for filter_size in filter_sizes:
        x = Conv1D(filter_size, kernel_size=3, activation='relu', padding='valid')(x)

    x = Dense(NUM_CLASSES)(x)
    return Model(encoded_frame_input, x, name='multi_frame_model')


def full_model(single_frame_encoder, multi_frame_encoder, num_frames=None):
    video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    frame_encoded = TimeDistributed(single_frame_encoder)(video_input)
    frame_diffs = Lambda(tf_diff)(frame_encoded)
    frame_diffs = TimeDistributed(Flatten())(frame_diffs)
    prediction = multi_frame_encoder(frame_diffs)
    return Model(video_input, prediction, name='full_model')


if __name__ == '__main__':
    single_frame_encoder = single_frame_model()
    multi_frame_encoder = multi_frame_model(num_frames=7)
    model = full_model(single_frame_encoder, multi_frame_encoder, num_frames=8)
    single_frame_encoder.summary()
    multi_frame_encoder.summary()

    FRAMES = 10
    start = time.time()
    for i in range(FRAMES):
        print(model.predict(np.zeros(shape=(1, 8, 108, 192, 3))).shape)
    end = time.time()
    print((end - start)/FRAMES)
    
    start = time.time()
    for i in range(FRAMES):
        print(single_frame_encoder.predict(np.zeros(shape=(1, 108, 192, 3))).shape)
    end = time.time()
    print((end - start)/FRAMES)
    
    start = time.time()
    for i in range(FRAMES):
        print(multi_frame_encoder.predict(np.zeros(shape=(1, 7, 4*6*2048))).shape)
    end = time.time()
    print((end - start)/FRAMES)

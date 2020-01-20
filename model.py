import time

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.regularizers import L1L2, l1_l2

from constants import *


def tf_diff(x):
    return x[:, 1:] - x[:, :-1]


def stem(x):
    x = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization(renorm=False)(x)

    b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    b2 = SeparableConv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    b2 = BatchNormalization(renorm=False)(b2)
    x = Concatenate(axis=-1)([b1, b2])

    b1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b1 = SeparableConv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b1)
    b1 = BatchNormalization(renorm=False)(b1)
    b2 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b2 = SeparableConv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(b2)
    b2 = SeparableConv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(b2)
    b2 = SeparableConv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b2)
    b2 = BatchNormalization(renorm=False)(b2)
    x = Concatenate(axis=-1)([b1, b2])

    b1 = SeparableConv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    b2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Concatenate(axis=-1)([b1, b2])
    return x


def blockA(x):
    reg = L1L2(l1=0., l2=0.001)

    res1 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)

    res2 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    res2 = SeparableConv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(res2)

    res3 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    res3 = SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(res3)
    res3 = SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(res3)

    res = Concatenate(axis=-1)([res1, res2, res3])
    res = Conv2D(filters=384, kernel_size=1, activation=None, padding='same')(res)
    res = BatchNormalization(renorm=False)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.15)([x, res])
    return x


def blockB(x):
    reg = L1L2(l1=0., l2=0.001)

    res1 = Conv2D(filters=192, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)

    res2 = Conv2D(filters=128, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    res2 = SeparableConv2D(filters=160, kernel_size=(7, 1), activation='relu', padding='same',
                           kernel_regularizer=reg)(res2)
    res2 = SeparableConv2D(filters=192, kernel_size=(1, 7), activation='relu', padding='same',
                           kernel_regularizer=reg)(res2)

    res = Concatenate(axis=-1)([res1, res2])
    res = Conv2D(filters=1152, kernel_size=1, activation=None, padding='same')(res)
    res = BatchNormalization(renorm=False)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.1)([x, res])
    return x


def reductionA(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    b2 = BatchNormalization(renorm=False)(b2)
    
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(b3)
    b3 = SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(b3)
    b3 = BatchNormalization(renorm=False)(b3)

    x = Concatenate(axis=-1)([b1, b2, b3])
    return x


def reductionB(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b2 = SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(b2)
    b2 = BatchNormalization(renorm=False)(b2)
    
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = SeparableConv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b3)
    b3 = BatchNormalization(renorm=False)(b3)
    
    b4 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b4 = SeparableConv2D(filters=256, kernel_size=3, activation='relu', padding='same')(b4)
    b4 = SeparableConv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b4)
    b4 = BatchNormalization(renorm=False)(b4)

    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    return x


def single_frame_model():
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frame_input = Input(shape=input_shape)

    x = stem(frame_input)

    for i in range(1):
        x = blockA(x)
    
    x = reductionA(x)

    for i in range(2):
        x = blockB(x)
    
    x = reductionB(x)

    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(num_frames=None):
    encoded_frame_input = Input(shape=(num_frames, 4 * 6 * 2048))

    x = Dense(256, activation='relu')(encoded_frame_input)
    x = BatchNormalization(renorm=False)(x)
    x = SpatialDropout1D(0.1)(x)
    x = Conv1D(256, kernel_size=2, activation='relu', padding='valid', strides=2)(x)
    x = BatchNormalization(renorm=False)(x)
    x = SpatialDropout1D(0.1)(x)
    filter_sizes = [256, 256]
    for filter_size in filter_sizes:
        x = SeparableConv1D(filter_size, kernel_size=3, activation='relu', padding='valid')(x)
        x = BatchNormalization(renorm=False)(x)

    x = Dense(NUM_CLASSES)(x)
    return Model(encoded_frame_input, x, name='multi_frame_model')


def full_model(single_frame_encoder, multi_frame_encoder, num_frames=None):
    video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    frame_encoded = TimeDistributed(single_frame_encoder)(video_input)

    use_diffs = False
    if use_diffs:
        frame_diffs = Lambda(tf_diff)(frame_encoded)
        frame_diffs = TimeDistributed(Flatten())(frame_diffs)
        prediction = multi_frame_encoder(frame_diffs)
    else:
        frame_encoded = TimeDistributed(Flatten())(frame_encoded)
        prediction = multi_frame_encoder(frame_encoded)

    return Model(video_input, prediction, name='full_model')


if __name__ == '__main__':
    single_frame_encoder = single_frame_model()
    multi_frame_encoder = multi_frame_model(num_frames=10)
    model = full_model(single_frame_encoder, multi_frame_encoder, num_frames=10)
    single_frame_encoder.summary()
    multi_frame_encoder.summary()

    frames = np.zeros(shape=(1, 10, 108, 192, 3))
    single_frame = np.zeros(shape=(1, 108, 192, 3))
    encoded_frames = np.zeros(shape=(1, 10, 4*6*2048))

    FRAMES = 10
    start = time.time()
    for i in range(FRAMES):
        print(model.predict(frames).shape)
    end = time.time()
    print((end - start)/FRAMES)
    
    start = time.time()
    for i in range(FRAMES):
        print(single_frame_encoder.predict(single_frame).shape)
    end = time.time()
    print((end - start)/FRAMES)
    
    start = time.time()
    for i in range(FRAMES):
        print(multi_frame_encoder.predict(encoded_frames).shape)
    end = time.time()
    print((end - start)/FRAMES)

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
    return Model(frame_input, x, name='single_frame_encoder')


def temporal_shuffle(x):
    num_frames = 8
    layer_depth = x.shape[-1]
    frames = tf.split(x, num_or_size_splits=num_frames, axis=1)
    frames = [tf.squeeze(frame, axis=[1]) for frame in frames]
    split_frames = [tf.split(frame, num_frames, axis=-1) for frame in frames]
    shuffled_frames = list(map(list, zip(*split_frames)))
    shuffled_frames = [Concatenate(axis=-1)(frame) for frame in shuffled_frames]
    shuffled_frames = tf.stack(shuffled_frames, axis=1)
    return shuffled_frames


def nonlocal_block(x):
    _, num_frames, height, width, channels = x.shape
    
    theta = Conv3D(filters=512, kernel_size=1)(x)
    phi = Conv3D(filters=512, kernel_size=1)(x)

    theta = Reshape((num_frames * height * width, 512))(theta)
    phi = Reshape((num_frames * height * width, 512))(phi)
    phi = tf.transpose(phi, perm=[0, 2, 1])
    
    att = theta @ phi
    att = Activation('softmax')(att)

    g = Conv3D(filters=512, kernel_size=1)(x)
    g = Reshape((num_frames * height * width, 512))(g)
    
    res = att @ g
    res = Reshape((num_frames, height, width, 512))(res)
    res = Conv3D(filters=channels, kernel_size=1, activation='relu')(res)
    res = BatchNormalization()(res)
    return res + x


def blockA_temporal(x):
    reg = L1L2(l1=0., l2=0.001)

    res1 = TimeDistributed(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg))(x)

    res2 = TimeDistributed(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg))(x)
    res2 = TimeDistributed(SeparableConv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg))(res2)

    res3 = TimeDistributed(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg))(x)
    res3 = TimeDistributed(SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg))(res3)
    res3 = TimeDistributed(SeparableConv2D(filters=48, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg))(res3)

    res = Concatenate(axis=-1)([res1, res2, res3])
    res = TimeDistributed(Conv2D(filters=384, kernel_size=1, activation=None, padding='same'))(res)
    res = BatchNormalization(renorm=False)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.15)([x, res])
    return x


def blockB_temporal(x):
    reg = L1L2(l1=0., l2=0.001)

    res1 = TimeDistributed(Conv2D(filters=192, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg))(x)

    res2 = TimeDistributed(Conv2D(filters=128, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg))(x)
    res2 = TimeDistributed(SeparableConv2D(filters=160, kernel_size=(7, 1), activation='relu', padding='same',
                           kernel_regularizer=reg))(res2)
    res2 = TimeDistributed(SeparableConv2D(filters=192, kernel_size=(1, 7), activation='relu', padding='same',
                           kernel_regularizer=reg))(res2)

    res = Concatenate(axis=-1)([res1, res2])
    res = TimeDistributed(Conv2D(filters=1152, kernel_size=1, activation=None, padding='same'))(res)
    res = BatchNormalization(renorm=False)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.1)([x, res])
    return x


def reductionA_temporal(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = TimeDistributed(MaxPooling2D(pool_size=3, strides=2, padding='same'))(x)

    b2 = TimeDistributed(SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same'))(x)
    b2 = BatchNormalization(renorm=False)(b2)

    b3 = TimeDistributed(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))(x)
    b3 = TimeDistributed(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))(b3)
    b3 = TimeDistributed(SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same'))(b3)
    b3 = BatchNormalization(renorm=False)(b3)

    x = Concatenate(axis=-1)([b1, b2, b3])
    return x


def reductionB_temporal(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = TimeDistributed(MaxPooling2D(pool_size=3, strides=2, padding='same'))(x)

    b2 = TimeDistributed(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))(x)
    b2 = TimeDistributed(SeparableConv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same'))(b2)
    b2 = BatchNormalization(renorm=False)(b2)

    b3 = TimeDistributed(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))(x)
    b3 = TimeDistributed(SeparableConv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'))(b3)
    b3 = BatchNormalization(renorm=False)(b3)

    b4 = TimeDistributed(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))(x)
    b4 = TimeDistributed(SeparableConv2D(filters=256, kernel_size=3, activation='relu', padding='same'))(b4)
    b4 = TimeDistributed(SeparableConv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same'))(b4)
    b4 = BatchNormalization(renorm=False)(b4)

    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    return x


def multi_frame_model(num_frames=None):
    if num_frames != 8:
        raise ValueError('This model requires 8 frames exactly')

    encoded_frame_input = Input(shape=(num_frames, 14, 24, 384))
    x = encoded_frame_input

    for _ in range(3):
        x = blockA_temporal(x)
        x = temporal_shuffle(x)

    x = reductionA_temporal(x)
    x = temporal_shuffle(x)
    
    for _ in range(4):
        x = blockB_temporal(x)
        x = temporal_shuffle(x)

    x = reductionB_temporal(x)
    x = temporal_shuffle(x)
    x = reductionB_temporal(x)
    x = temporal_shuffle(x)

    x = nonlocal_block(x)

    x = TimeDistributed(Flatten())(x)
    x = Dense(NUM_CLASSES)(x)
    return Model(encoded_frame_input, x, name='multi_frame_model')


def full_model(single_frame_encoder, multi_frame_encoder, num_frames=None):
    video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    frame_encoded = TimeDistributed(single_frame_encoder)(video_input)

    use_diffs = False
    if use_diffs:
        frame_diffs = Lambda(tf_diff)(frame_encoded)
        prediction = multi_frame_encoder(frame_diffs)
    else:
        prediction = multi_frame_encoder(frame_encoded)

    return Model(video_input, prediction, name='full_model')


if __name__ == '__main__':
    single_frame_encoder = single_frame_model()
    multi_frame_encoder = multi_frame_model(num_frames=8)
    model = full_model(single_frame_encoder, multi_frame_encoder, num_frames=8)
    single_frame_encoder.summary()
    multi_frame_encoder.summary()

    frames = np.zeros(shape=(1, 8, 108, 192, 3))
    single_frame = np.zeros(shape=(1, 108, 192, 3))
    encoded_frames = np.zeros(shape=(1, 8, 14, 24, 384))

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

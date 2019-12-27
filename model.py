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

    b1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b1 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b1)
    b2 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same')(x)
    b2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(b2)
    b2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(b2)
    b2 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same')(b2)
    x = Concatenate(axis=-1)([b1, b2])

    b1 = Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    b2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Concatenate(axis=-1)([b1, b2])
    return x


def blockA(x):
    res1 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)

    res2 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)
    res2 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(res2)

    res3 = Conv2D(filters=32, kernel_size=1, activation='relu', padding='same')(x)
    res3 = Conv2D(filters=48, kernel_size=3, activation='relu', padding='same')(res3)
    res3 = Conv2D(filters=48, kernel_size=3, activation='relu', padding='same')(res3)

    res = Concatenate(axis=-1)([res1, res2, res3])
    res = Conv2D(filters=384, kernel_size=1, activation=None, padding='same')(res)
    x = Lambda(lambda a: a[0] + a[1]*0.15)([x, res])
    return x


def blockB(x):
    res1 = Conv2D(filters=192, kernel_size=1, activation='relu', padding='same')(x)

    res2 = Conv2D(filters=128, kernel_size=1, activation='relu', padding='same')(x)
    res2 = Conv2D(filters=160, kernel_size=(7, 1), activation='relu', padding='same')(res2)
    res2 = Conv2D(filters=192, kernel_size=(1, 7), activation='relu', padding='same')(res2)

    res = Concatenate(axis=-1)([res1, res2])
    res = Conv2D(filters=1152, kernel_size=1, activation=None, padding='same')(res)
    x = Lambda(lambda a: a[0] + a[1]*0.1)([x, res])
    return x


def reductionA(x):
    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(b3)
    b3 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same')(b3)

    x = Concatenate(axis=-1)([b1, b2, b3])
    return x


def reductionB(x):
    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b2 = Conv2D(filters=385, kernel_size=3, strides=2, activation='relu', padding='same')(b2)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b3 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b3)

    b4 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same')(x)
    b4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(b4)
    b4 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(b4)

    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    return x


def single_frame_model():
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frame_input = Input(shape=input_shape)

    x = stem(frame_input)

    for i in range(4):
        x = blockA(x)

    x = reductionA(x)

    for i in range(7):
        x = blockB(x)

    x = reductionB(x)

    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(single_frame_encoder, num_frames=None):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    encoded_output = time_distributed_frame_encoder(video_input)

    diff_outputs = True
    if diff_outputs:
        diffed_outputs = Lambda(tf_diff)(encoded_output)
        x = diffed_outputs
    else:
        x = encoded_output

    x = ConvLSTM2D(filters=256, kernel_size=(3, 1), padding='valid', return_sequences=True, stateful=False)(x)
    x = ConvLSTM2D(filters=256, kernel_size=(1, 3), padding='valid', return_sequences=True, stateful=False)(x)

    s = x.shape[-1] * x.shape[-2] * x.shape[-3]
    x = Reshape(target_shape=(-1, s))(x)

    x = LSTM(256, return_sequences=True, stateful=False)(x)

    x = Dense(NUM_CLASSES)(x)
    return Model(video_input, x, name='multi_frame_model')

import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.regularizers import L1L2, l1_l2

from constants import *


def tf_diff(x):
    return x[:, 1:] - x[:, :-1]


def stem(x):
    reg = L1L2(l1=0.001, l2=0.001)

    x = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)

    b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    b2 = Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = Concatenate(axis=-1)([b1, b2])

    b1 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b1 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(b1)
    b2 = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same', kernel_regularizer=reg)(b2)
    b2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same', kernel_regularizer=reg)(b2)
    b2 = Conv2D(filters=96, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(b2)
    x = Concatenate(axis=-1)([b1, b2])

    b1 = Conv2D(filters=192, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(x)
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
    res = Conv2D(filters=384, kernel_size=1, activation=None, padding='same', kernel_regularizer=reg)(res)
    res = BatchNormalization(axis=-1)(res)

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
    res = Conv2D(filters=1152, kernel_size=1, activation=None, padding='same', kernel_regularizer=reg)(res)
    res = BatchNormalization(axis=-1)(res)

    x = Lambda(lambda a: a[0] + a[1]*0.1)([x, res])
    return x


def reductionA(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(x)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(b3)
    b3 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(b3)

    x = Concatenate(axis=-1)([b1, b2, b3])
    return x


def reductionB(x):
    reg = L1L2(l1=0., l2=0.001)

    b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    b2 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b2 = Conv2D(filters=384, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(b2)

    b3 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b3 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(b3)

    b4 = Conv2D(filters=256, kernel_size=1, activation='relu', padding='same', kernel_regularizer=reg)(x)
    b4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(b4)
    b4 = Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=reg)(b4)

    x = Concatenate(axis=-1)([b1, b2, b3, b4])
    return x


def single_frame_model():
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frame_input = Input(shape=input_shape)

    x = stem(frame_input)

    for i in range(3):
        x = blockA(x)
        x = Dropout(0.2)(x)

    x = reductionA(x)

    for i in range(4):
        x = blockB(x)
        x = Dropout(0.2)(x)

    x = reductionB(x)

    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(single_frame_encoder, num_frames=None, stateful=False):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    if stateful:
        video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3), batch_size=1)
    else:
        video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    encoded_output = time_distributed_frame_encoder(video_input)

    diff_outputs = False
    if diff_outputs:
        diffed_outputs = Lambda(tf_diff)(encoded_output)
        x = diffed_outputs
    else:
        x = encoded_output

    s = x.shape[-1] * x.shape[-2] * x.shape[-3]
    x = Reshape(target_shape=(-1, s))(x)
    x = Dense(1024, kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)
    x = BatchNormalization(axis=-1)(x)

    filter_sizes = [1024, 512, 256, 128]
    for filter_size in filter_sizes:
        x = Conv1D(filter_size, kernel_size=3, activation='relu', padding='valid',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)

    x = Dense(NUM_CLASSES, kernel_regularizer=l1_l2(l1=0., l2=0.001))(x)
    return Model(video_input, x, name='multi_frame_model')

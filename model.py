import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *

from constants import *


def tf_diff(x):
    return x[:, 1:] - x[:, :-1]


def single_frame_model():
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    frame_input = Input(shape=input_shape)

    x = Conv2D(filters=32, kernel_size=3, activation='relu')(frame_input)

    InceptionResNetV2

    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(single_frame_encoder, num_frames=None):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    video_input = Input(shape=(num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    # encoded_output = time_distributed_frame_encoder(video_input)
    # diffed_outputs = Lambda(tf_diff)(encoded_output)
    # x = diffed_outputs
    x = video_input

    filter_sizes = [64, 128, 256, 256, 512, 512]
    strides = [1, (1, 2, 2), (1, 2, 2), (1, 2, 2), 2, 2]
    for filter_size, stride in zip(filter_sizes, strides):
        x = Conv3D(filters=filter_size, kernel_size=3, strides=stride, activation='relu')(x)

    if x.shape[1] is not None:
        t = x.shape[1]
    else:
        t = -1

    s = x.shape[-1] * x.shape[-2] * x.shape[-3]
    x = Reshape((t, s))(x)

    x = Dense(NUM_CLASSES)(x)
    return Model(video_input, x, name='multi_frame_model')

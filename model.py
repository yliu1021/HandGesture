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
    mobile_net_tensor = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_tensor=frame_input,
        include_top=False,
        weights=None,
        pooling=None
    )(frame_input)
    x = Flatten()(mobile_net_tensor)
    x = Dropout(0.25)(x)
    x = Dense(EXPERIENCE_TRAJECTORY_DIMS)(x)
    return Model(frame_input, x, name='single_frame_encoder')


def multi_frame_model(single_frame_encoder):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    video_input = Input(shape=(EXPERIENCE_TRAJECTORY_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    encoded_output = time_distributed_frame_encoder(video_input)
    diffed_outputs = Lambda(tf_diff)(encoded_output)

    x = Conv1D(filters=256, kernel_size=3)(diffed_outputs)

    filter_sizes = [256, 256, 256]
    for filter_size in filter_sizes:
        res = Dense(filter_size, activation='relu')(x)
        res = Conv1D(filters=int(filter_size * 1.5), kernel_size=3, padding='SAME')(res)
        res = Dense(filter_size)(res)
        x = Add()([x, res])

    x = Dense(NUM_CLASSES)(x)
    return Model(video_input, x, name='multi_frame_model')


def training_model(single_frame_encoder):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    inputs = [Input(shape=(EXPERIENCE_TRAJECTORY_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
              for _ in range(TRAIN_BRANCHES+1)]
    encoded_outputs = map(lambda x: time_distributed_frame_encoder(x), inputs)
    diffed_outputs = map(lambda x: Lambda(tf_diff)(x), encoded_outputs)

    return Model(inputs, list(diffed_outputs), name='training_model')

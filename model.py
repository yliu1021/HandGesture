import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *

from constants import *


def tf_diff(x):
    return x[1:] - x[:-1]


def single_frame_model():
    model = Sequential(name='single_frame_encoder')

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', strides=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(EXPERIENCE_TRAJECTORY_DIMS))

    return model


def training_model(single_frame_encoder):
    time_distributed_frame_encoder = TimeDistributed(single_frame_encoder)

    inputs = [Input(shape=(EXPERIENCE_TRAJECTORY_SAMPLES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) for _ in range(NUM_CLASSES+1)]
    encoded_outputs = map(lambda x: time_distributed_frame_encoder(x), inputs)
    diffed_outputs = map(lambda x: Lambda(tf_diff)(x), encoded_outputs)

    return Model(inputs, list(diffed_outputs), name='training_model')

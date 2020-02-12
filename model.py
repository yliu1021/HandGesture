import time

import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.regularizers import L1L2, l1_l2

from constants import *


def identity_block(input_tensor, kernel_size, filters, non_degenerate_temporal_conv=True):
    filters1, filters2, filters3 = filters
    if non_degenerate_temporal_conv:
        x = Conv3D(filters1, (3, 1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = Conv3D(filters1, (1, 1, 1))(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv3D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(filters3, (1, 1, 1))(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=(1, 2, 2), non_degenerate_temporal_conv=True):
    filters1, filters2, filters3 = filters
    if non_degenerate_temporal_conv:
        x = Conv3D(filters1, (3, 1, 1), strides=strides, padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    else:
        x = Conv3D(filters1, (1, 1, 1), strides=strides)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    x = Conv3D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(filters3, (1, 1 ,1))(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv3D(filters3, (1, 1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def lateral_connection(fast_res_block, slow_res_block, alpha=8, beta=1/8):
    num_filters = int(2*beta*int(fast_res_block.shape[4]))
    lateral = Conv3D(num_filters, padding='same', kernel_size=(5, 1, 1), strides=(alpha, 1, 1))(fast_res_block)
    connection = Concatenate(axis=-1)([slow_res_block,lateral])
    return connection


def fast_pathway_model():
    fast_frame_input = Input(shape=(FAST_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = fast_frame_input
    
    x = Conv3D(filters=8, kernel_size=(5, 7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2))(x)
    res_0 = x
    
    x = conv_block(res_0, kernel_size=(1, 3, 3), filters=(64//8, 64//8, 256//8), strides=(1, 1, 1))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(64//8, 64//8, 256//8))
    res_1 = identity_block(x, kernel_size=(1, 3, 3), filters=(64//8, 64//8, 256//8))
    
    x = conv_block(res_1, kernel_size=(1, 3, 3), filters=(128//8, 128//8, 512//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(128//8, 128//8, 512//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(128//8, 128//8, 512//8))
    res_2 = identity_block(x, kernel_size=(1, 3, 3), filters=(128//8, 128//8, 512//8))

    x = conv_block(res_2, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))
    res_3 = identity_block(x, kernel_size=(1, 3, 3), filters=(256//8, 256//8, 1024//8))

    x = conv_block(res_3, kernel_size=(1, 3, 3), filters=(512//8, 512//8, 2048//8))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(512//8, 512//8, 2048//8))
    res_4 = identity_block(x, kernel_size=(1, 3, 3), filters=(512//8, 512//8, 2048//8))
    
    return Model(fast_frame_input, [res_0, res_1, res_2, res_3, res_4], name='fast_model')


def slow_pathway_model():
    slow_frame_input = Input(shape=(SLOW_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    x = slow_frame_input
    
    res_0_fast = Input(shape=(FAST_FRAMES, 53, 95, 8))
    res_1_fast = Input(shape=(FAST_FRAMES, 53, 95, 32))
    res_2_fast = Input(shape=(FAST_FRAMES, 27, 48, 64))
    res_3_fast = Input(shape=(FAST_FRAMES, 14, 24, 128))
    res_4_fast = Input(shape=(FAST_FRAMES, 7, 12, 256))
    
    x = Conv3D(filters=64, kernel_size=(1, 7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2))(x)
    res_0 = x
    res_0 = lateral_connection(res_0_fast, res_0)

    x = conv_block(res_0, kernel_size=(1, 3, 3), filters=(64, 64, 256), strides=(1, 1, 1), non_degenerate_temporal_conv=False)
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(64, 64, 256), non_degenerate_temporal_conv=False)
    res_1 = identity_block(x, kernel_size=(1, 3, 3), filters=(64, 64, 256), non_degenerate_temporal_conv=False)
    res_1 = lateral_connection(res_1_fast, res_1)
    
    x = conv_block(res_1, kernel_size=(1, 3, 3), filters=(128, 128, 512), non_degenerate_temporal_conv=False)
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(128, 128, 512), non_degenerate_temporal_conv=False)
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(128, 128, 512), non_degenerate_temporal_conv=False)
    res_2 = identity_block(x, kernel_size=(1, 3, 3), filters=(128, 128, 512), non_degenerate_temporal_conv=False)
    res_2 = lateral_connection(res_2_fast, res_2)
    
    x = conv_block(res_2, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    res_3 = identity_block(x, kernel_size=(1, 3, 3), filters=(256, 256, 1024))
    res_3 = lateral_connection(res_3_fast, res_3)

    x = conv_block(res_3, kernel_size=(1, 3, 3), filters=(512, 512, 2048))
    x = identity_block(x, kernel_size=(1, 3, 3), filters=(512, 512, 2048))
    res_4 = identity_block(x, kernel_size=(1, 3, 3), filters=(512, 512, 2048))
    res_4 = lateral_connection(res_4_fast, res_4)
    
    return Model([slow_frame_input, res_0_fast, res_1_fast, res_2_fast, res_3_fast, res_4_fast],
                 res_4, name='slow_model')


def slowfast_model(fast_model, slow_model):
    video_input = Input(shape=(FAST_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    slow_video_input = tf.gather(video_input, tf.range(0, FAST_FRAMES, 8), axis=1)
    
    res_0_fast, res_1_fast, res_2_fast, res_3_fast, res_4_fast = fast_model(video_input)
    res_4_slow = slow_model([slow_video_input, res_0_fast, res_1_fast, res_2_fast, res_3_fast, res_4_fast])
    
    fast_res = GlobalAveragePooling3D()(res_4_fast)
    slow_res = GlobalAveragePooling3D()(res_4_slow)
    
    out = Concatenate(axis=-1)([fast_res, slow_res])
    out = tf.expand_dims(out, axis=1)
    out = Dense(NUM_CLASSES)(out)
    
    return Model(video_input, out, name='slowfast_model')


def build_model():
    fast_model = fast_pathway_model()
    slow_model = slow_pathway_model()
    full_model = slowfast_model(fast_model, slow_model)
    
    return fast_model, slow_model, full_model


def benchmark_model(full_model, iters=30):
    frames = np.zeros(shape=(1, FAST_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH, 3))

    start = time.time()
    for i in range(iters):
        full_model.predict(frames)
    end = time.time()
    model_latency = (end - start) / iters
    
    return model_latency


def main():
    fast_model, slow_model, full_model = build_model()
    fast_model.summary()
    slow_model.summary()
    full_model.summary()
    
    print('Benchmarking model...')
    latency = benchmark_model(full_model)
    print(f'Speed: {latency:.4f}')


if __name__ == '__main__':
    main()

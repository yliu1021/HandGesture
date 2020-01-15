import os

import tensorflow as tf
import tfcoreml

import model
import data

import matplotlib.pyplot as plt
import cv2
import numpy as np
from constants import *
from matplotlib import animation
import time
from PIL import Image


def small_dataset():
    validation_data_generator = data.validation_dataset.data_generator(
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    for i in range(3):
        data = next(validation_data_generator)
        yield data[0]


training_dir = './training'
training_run = 'run8'
model_loc = os.path.join(training_dir, training_run, 'full_model.01.h5')

single_frame_encoder = model.single_frame_model()
multi_frame_encoder = model.multi_frame_model(num_frames=7)
single_frame_encoder.load_weights(model_loc, by_name=True)
multi_frame_encoder.load_weights(model_loc, by_name=True)
single_frame_encoder.save(os.path.join('./inference', training_run, 'single_frame.h5'))
multi_frame_encoder.save(os.path.join('./inference', training_run, 'multi_frame.h5'))

print('Loaded model')
single_frame_encoder.summary()
multi_frame_encoder.summary()


def convert_to_coreml():
    multi_frame_input_name = multi_frame_encoder.inputs[0].name.split(':')[0]
    keras_output_node_name = multi_frame_encoder.outputs[0].name.split(':')[0]
    m_graph_output_node_name = keras_output_node_name.split('/')[-1]
    print('Converting with input name: {}\noutput name: {}\ngraph output name: {}'.format(multi_frame_input_name,
                                                                                          keras_output_node_name,
                                                                                          m_graph_output_node_name))
    multi_frame_mlmodel = tfcoreml.convert(tf_model_path=os.path.join('./inference', training_run, 'multi_frame.h5'),
                             input_name_shape_dict={multi_frame_input_name: (1, 7, 4*6*2048)},
                             output_feature_names=[m_graph_output_node_name],
                             minimum_ios_deployment_target='13')

    inference_dir = './inference'
    mlmodel_loc = os.path.join(inference_dir, training_run)
    os.makedirs(mlmodel_loc, exist_ok=True)
    mlmodel_loc = os.path.join(mlmodel_loc, 'MultiFrame.mlmodel')
    multi_frame_mlmodel.save(mlmodel_loc)

    single_frame_input_name = single_frame_encoder.inputs[0].name.split(':')[0]
    keras_output_node_name = single_frame_encoder.outputs[0].name.split(':')[0]
    s_graph_output_node_name = keras_output_node_name.split('/')[-1]
    print('Converting with input name: {}\noutput name: {}\ngraph output name: {}'.format(single_frame_input_name,
                                                                                          keras_output_node_name,
                                                                                          s_graph_output_node_name))
    single_frame_mlmodel = tfcoreml.convert(tf_model_path=os.path.join('./inference', training_run, 'single_frame.h5'),
                             input_name_shape_dict={single_frame_input_name: (1, 108, 192, 3)},
                             image_input_names=single_frame_input_name,
                             is_bgr=True,
                             image_scale=1.0/255.0,
                             output_feature_names=[s_graph_output_node_name],
                             minimum_ios_deployment_target='13')

    inference_dir = './inference'
    mlmodel_loc = os.path.join(inference_dir, training_run)
    os.makedirs(mlmodel_loc, exist_ok=True)
    mlmodel_loc = os.path.join(mlmodel_loc, 'SingleFrame.mlmodel')
    single_frame_mlmodel.save(mlmodel_loc)
    
    # TESTING
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.figure.set_size_inches(10, 10)
    bars = ax.bar(data.labels, np.zeros(NUM_CLASSES, dtype=np.float32))
    plt.xticks(rotation=90)
    plt.tight_layout()

    num_frames = 7
    frame_time = 1/MIN_FPS * 1000
    
    single_frame_model = single_frame_mlmodel
    multi_frame_model = multi_frame_mlmodel

    cap = cv2.VideoCapture(0)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    prev = np.zeros((4*6*2048), dtype=np.float32)
    model_input = np.zeros((1, num_frames, 4*6*2048), dtype=np.float32)

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def animate(i):
        start = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("Couldn't read input")
            return
        
        frame = cv2.resize(frame, image_size)
        frame_img = Image.fromarray(frame)
        frame = frame.astype(np.float32) / 255.0

        frame_encoded = single_frame_model.predict({single_frame_input_name: frame_img})[s_graph_output_node_name]
        frame_encoded = frame_encoded.reshape(4*6*2048)
        frame_diff = frame_encoded - prev
        prev[:] = frame_encoded
        
        model_input[:-1] = model_input[1:]
        model_input[-1] = frame_diff
        
        pred = multi_frame_model.predict({multi_frame_input_name: model_input})[m_graph_output_node_name][0][0]
        pred = softmax(pred)

        for bar, p in zip(bars, pred):
            bar.set_height(p)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(max(1, int((time.time() - start)*1000 - frame_time))) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

        if pred.max() > 0.5:
            predictions = pred.argsort()
            print(data.labels[predictions[-1]])
            print(data.labels[predictions[-2]])
        return bars

    animation.FuncAnimation(fig, animate, frames=None, interval=1, blit=True)
    plt.show()


def convert_to_tflite():
    converter = tf.lite.TFLiteConverter.from_keras_model(multi_frame_model)
    converter.post_training_quantize = True
    converter.representative_dataset = tf.lite.RepresentativeDataset(small_dataset())
    print('Converting to tf lite...')
    tflite_model = converter.convert()
    print('Converted to tf lite')

    inference_dir = './inference'
    tflite_model_loc = os.path.join(inference_dir, training_run)
    os.makedirs(tflite_model_loc, exist_ok=True)
    tflite_model_loc = os.path.join(tflite_model_loc, 'multi_frame_model.tflite')
    print('Saving tf lite')
    try:
        with open(tflite_model_loc, 'wb+') as f:
            f.write(tflite_model)
    except Exception as e:
        print(f'Unable to save: {e}')
    else:
        print('Saved tf lite')


if __name__ == '__main__':
    convert_to_coreml()

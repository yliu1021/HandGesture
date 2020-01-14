import sys
import os
import time

from tensorflow.keras.models import load_model as keras_load_model
import cv2
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

import data
from constants import *
import train
import model


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def single_frame(model_loc):
    single_frame_model = model.single_frame_model()
    single_frame_model.load_weights(model_loc, by_name=True)
    cap = cv2.VideoCapture(0)
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, image_size)
        frame = frame.astype(np.float32) / 255.0
        pred = single_frame_model.predict(frame[None, ...])[0]
        
        encoded_frames = list()
        for i in range(40, 45):
            f = pred[..., i]
            encoded_frames.append(f)
        
        encoded_frames = np.array(encoded_frames)
        encoded_frames -= encoded_frames.min()
        encoded_frames /= encoded_frames.max()
        
        for i, f in enumerate(encoded_frames):
            cv2.imshow(f'frame_{i}', f)

        if cv2.waitKey(1) == ord('q'):
            break
        

def main(model_loc):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.figure.set_size_inches(10, 10)
    bars = ax.bar(data.labels, np.zeros(NUM_CLASSES, dtype=np.float32))
    plt.xticks(rotation=90)
    plt.tight_layout()

    num_frames = 8
    frame_time = 1/MIN_FPS * 1000
    
    single_frame_model = model.single_frame_model()
    multi_frame_model = model.multi_frame_model(single_frame_model, num_frames=num_frames)
    multi_frame_model.load_weights(model_loc)

    cap = cv2.VideoCapture(0)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    model_input = np.zeros((1, num_frames, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)

    def animate(i):
        start = time.time()
        ret, frame = cap.read()
        read_time = time.time() - start
        
        if not ret:
            print("Couldn't read input")
            return
        
        start = time.time()
        frame = cv2.resize(frame, image_size)
        frame = frame.astype(np.float32) / 255.0
        resize_time = time.time() - start
        
        start = time.time()
        model_input[0, :-1] = model_input[0, 1:]
        model_input[0, -1] = frame
        input_shift_time = time.time() - start
        
        start = time.time()
        pred = multi_frame_model.predict(model_input)[0]
        predict_time = time.time() - start
        
        start = time.time()
        pred = np.max(pred, axis=0)
        pred = softmax(pred)
        softmax_time = time.time() - start

        for bar, p in zip(bars, pred):
            bar.set_height(p)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(max(1, int((time.time() - start)*1000 - frame_time))) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

        print('Read: {:.3f}\nResize: {:.3f}\nShift: {:.3f}\nPredict: {:.3f}\nSoftmax: {:.3f}\nTotal: {:.3f}\n\n'.format(
                read_time,
                resize_time,
                input_shift_time,
                predict_time,
                softmax_time,
                read_time + resize_time + input_shift_time + predict_time + softmax_time
            )
        )
        if pred.max() > 0.5:
            predictions = pred.argsort()
            print(data.labels[predictions[-1]])
            print(data.labels[predictions[-2]])
        return bars

    animation.FuncAnimation(fig, animate, frames=None, interval=1, blit=True)
    plt.show()


if __name__ == '__main__':
    if 'single' in sys.argv:
        single_frame(sys.argv[1])
    else:
        main(sys.argv[1])

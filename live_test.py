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

    num_frames = 7
    frame_time = 1/MIN_FPS * 1000
    
    single_frame_model = model.single_frame_model()
    multi_frame_model = model.multi_frame_model(num_frames=num_frames)
    full_model = model.full_model(single_frame_model, multi_frame_model, num_frames=num_frames + 1)
    full_model.load_weights(model_loc)

    cap = cv2.VideoCapture(0)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    prev = np.zeros((4*6*2048), dtype=np.float32)
    model_input = np.zeros((1, num_frames, 4*6*2048), dtype=np.float32)

    def animate(i):
        start = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("Couldn't read input")
            return
        
        frame = cv2.resize(frame, image_size)
        frame = frame.astype(np.float32) / 255.0
        
        frame_encoded = single_frame_model.predict(frame[None, ...])[0]
        frame_encoded = frame_encoded.reshape(4*6*2048)
        frame_diff = frame_encoded - prev
        prev[:] = frame_encoded
        
        model_input[0, :-1] = model_input[0, 1:]
        model_input[0, -1] = frame_diff
        
        pred = multi_frame_model.predict(model_input)[0]
        
        pred = np.max(pred, axis=0)
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


if __name__ == '__main__':
    if 'single' in sys.argv:
        single_frame(sys.argv[1])
    else:
        main(sys.argv[1])

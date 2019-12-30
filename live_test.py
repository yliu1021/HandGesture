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


def main():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.figure.set_size_inches(10, 10)
    bars = ax.bar(data.labels, np.zeros(NUM_CLASSES, dtype=np.float32))
    plt.xticks(rotation=90)
    plt.tight_layout()

    num_frames = 8
    frame_time = 1/MIN_FPS * 1000
    
    single_frame_model = model.single_frame_model()
    multi_frame_model = model.multi_frame_model(single_frame_model, num_frames=num_frames, stateful=True)
    multi_frame_model.load_weights(os.path.join('training', 'run5', 'multi_frame_model.12.hdf5'))

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
        resize_time = time.time() - start
        
        start = time.time()
        model_input[0, :-1] = model_input[0, 1:]
        model_input[0, -1] = frame
        input_shift_time = time.time() - start
        
        start = time.time()
        pred = multi_frame_model.predict(model_input)
        predict_time = time.time() - start
        
        start = time.time()
        pred = softmax(pred[0, -1])
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
        return bars

    animation.FuncAnimation(fig, animate, frames=None, interval=1, blit=True)
    plt.show()


if __name__ == '__main__':
    main()

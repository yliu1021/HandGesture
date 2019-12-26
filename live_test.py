import os

from tensorflow.keras.models import load_model as keras_load_model
import cv2
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

import data
from constants import *
import train


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

    model = keras_load_model(os.path.join('training', 'run1', 'multi_frame_model.hdf5'), custom_objects={
        'temporal_crossentropy': train.temporal_crossentropy,
        'temporal_accuracy': train.temporal_accuracy,
        'temporal_top_k_accuracy': train.temporal_top_k_accuracy
    })
    cap = cv2.VideoCapture(0)

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    model_input = np.zeros((1, 6, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)

    def animate(i):
        ret, frame = cap.read()
        if not ret:
            print("Couldn't read input")
            return

        frame = cv2.resize(frame, image_size)
        model_input[0, :-1] = model_input[0, 1:]
        model_input[0, -1] = frame
        pred = model.predict(model_input)
        pred = softmax(pred[0, -1])

        for bar, p in zip(bars, pred):
            bar.set_height(p)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(int(1/MIN_FPS * 1000)) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)

        return bars

    animation.FuncAnimation(fig, animate, frames=None, interval=1, blit=True)
    plt.show()


if __name__ == '__main__':
    main()

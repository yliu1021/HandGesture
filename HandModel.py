import os
import time
from queue import Queue

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

import model


class KerasModel:
    def __init__(self, single_frame_model_loc, multi_frame_model_loc):
        self.single_frame_model = load_model(single_frame_model_loc)
        self.multi_frame_model = load_model(multi_frame_model_loc)
        
        self.frame_encoding_diffs = np.zeros(shape=(1, 8, 4 * 6 * 2048), dtype=np.float32)
        
        self.predicting = False

    def receive_frame(self, frame, should_predict=False):
        frame = cv2.resize(frame, (192, 108))
        frame = frame.astype(np.float32) / 255.0

        new_frame_encoding = self.single_frame_model.predict(frame[None, ...])
        
        new_frame_encoding = np.reshape(new_frame_encoding, 4*6*2048)
        
        self.frame_encoding_diffs[0, :-1] = self.frame_encoding_diffs[0, 1:]
        self.frame_encoding_diffs[0, -1] = new_frame_encoding
        
        if should_predict:
            pred = self.multi_frame_model.predict(self.frame_encoding_diffs)[0, 0]

            pred_exp = np.exp(pred)
            pred_exp /= np.sum(pred_exp)
            return pred_exp
    
    def start_predicting(self, threshold, callback_queue):
        cap = cv2.VideoCapture(0)
        self.predicting = True
        while self.predicting:
            start = time.time()
            ret, frame = cap.read()
            pred = self.receive_frame(frame, should_predict=True)
            pred = np.exp(pred - np.max(pred))
            pred /= pred.sum()
            if pred.max() > threshold:
                callback_queue.put(pred, block=False)
            if cv2.waitKey(1) == ord('q'):
                break
            end = time.time()
            cv2.waitKey(int(max(1, (1/8 - (end - start)) * 1000)))

    def stop_predicting(self):
        self.predicting = False


if __name__ == '__main__':
    model_dir = os.path.join('./inference', 'run9')
    single_frame_model_loc = os.path.join(model_dir, 'single_frame_model.tflite')
    multi_frame_model_loc = os.path.join(model_dir, 'multi_frame_model.tflite')
    model = Model(single_frame_model_loc, multi_frame_model_loc)


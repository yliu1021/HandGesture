import os
import queue
import threading
import time

from HandModel import KerasModel


GESTURES = [
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things"
]


def setup_model(callback_queue):
    model_dir = os.path.join('./inference', 'run10')
    single_frame_model_loc = os.path.join(model_dir, 'single_frame.h5')
    multi_frame_model_loc = os.path.join(model_dir, 'multi_frame.h5')
    model = KerasModel(single_frame_model_loc, multi_frame_model_loc)
    model.start_predicting(0.085, callback_queue)


def main():
    gesture_queue = queue.Queue()
    prediction_thread = threading.Thread(target=setup_model, args=(gesture_queue,))
    prediction_thread.start()
    
    previous_gestures = dict()
    for gesture in GESTURES:
        previous_gestures[gesture] = [(gesture, 0)]
    
    print('Looping')
    while True:
        gesture_pred = gesture_queue.get(block=True)
        gesture_time = time.time()
        
        gesture = GESTURES[gesture_pred.argmax()]
        prev_same_gestures = previous_gestures[gesture]
        if gesture_time - prev_same_gestures[-1][1] > 1.0:
            previous_gestures[gesture] = [(gesture, gesture_time)]
        else:
            previous_gestures[gesture].append((gesture, gesture_time))
        
        if len(previous_gestures[gesture]) == 3:
            confidence = gesture_pred.max()
            if gesture != 'No gesture' and gesture != 'Doing other things':
                print('{:.3f}: {}'.format(confidence, gesture))


if __name__ == '__main__':
    main()

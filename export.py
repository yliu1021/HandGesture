import os

import tensorflow as tf

import model


training_dir = './training'
training_run = 'run1'
model_loc = os.path.join(training_dir, training_run, 'multi_frame_predictor.10.hdf5')

single_frame_model = model.single_frame_model()
multi_frame_model = model.multi_frame_model(single_frame_model, num_frames=6)
multi_frame_model.load_weights(model_loc)
print('Loaded model')
multi_frame_model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(multi_frame_model)
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

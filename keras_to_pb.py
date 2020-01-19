import sys

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


K.set_learning_phase(0)

model = load_model(sys.argv[1])
print(model.input)
print(model.output)

tf.saved_model.save(model, f'pb_models/{model.name}')

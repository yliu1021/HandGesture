import os

import tensorflow as tf
import tfcoreml

import model
import data


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
training_run = 'run5'
model_loc = os.path.join(training_dir, training_run, 'multi_frame_model_weights.h5')

single_frame_model = model.single_frame_model()
multi_frame_model = model.multi_frame_model(single_frame_model, num_frames=8)
multi_frame_model.load_weights(model_loc)
print('Loaded model')
multi_frame_model.summary()
print(multi_frame_model.input)
print(multi_frame_model.output)


def convert_to_coreml():
    input_name = multi_frame_model.inputs[0].name.split(':')[0]
    keras_output_node_name = multi_frame_model.outputs[0].name.split(':')[0]
    graph_output_node_name = keras_output_node_name.split('/')[-1]
    print('Converting with input name: {}\noutput name: {}\ngraph output name: {}'.format(input_name,
                                                                                          keras_output_node_name,
                                                                                          graph_output_node_name))
    model = tfcoreml.convert(tf_model_path=model_loc,
                             input_name_shape_dict={input_name: (1, 8, 108, 192, 3)},
                             output_feature_names=[graph_output_node_name],
                             minimum_ios_deployment_target='13')

    inference_dir = './inference'
    tflite_model_loc = os.path.join(inference_dir, training_run)
    os.makedirs(tflite_model_loc, exist_ok=True)
    mlmodel_loc = os.path.join(tflite_model_loc, 'multi_frame_model.mlmodel')
    model.save(mlmodel_loc)


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

import os.path

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from tensorflow.lite.python.util import convert_bytes_to_c_source


def convert_tf_lite(tf_model_path, tf_lite_model_path):
    # Convert the model to the TensorFlow Lite format without quantization
    float_converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    float_tflite_model = float_converter.convert()
    float_tflite_model_size = open(tf_lite_model_path, "wb").write(float_tflite_model)
    print("Float model is %d bytes" % float_tflite_model_size)


def convert_tf_lite_quantized(tf_model_path, tf_lite_model_path, representative_dataset, mode='16x8'):
    # Convert the model to the TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    if mode == '16x8':  # activations are int16, weights are int8, input and output are int16
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
    elif mode == '8x8':  # activations and weights are int8, input and output are int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    elif mode == '8x8_full_integer':  # activations and weights are int8, input and output are float32
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32  # or tf.uint8, tf.int8
        converter.inference_output_type = tf.float32  # or tf.uint8, tf.int8

    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    tflite_model_size = open(tf_lite_model_path, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)


def convert_tf_lite_mtl(tf_model_path, tf_lite_model_path, representative_dataset, output_details):
    # Convert the model to the TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    converter.allow_custom_ops = True
    converter.output_details = output_details  # set output_details
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    tflite_model_size = open(tf_lite_model_path, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)


def convert_to_c(tflite_model, file_name):
    source_text, header_text = convert_bytes_to_c_source(tflite_model, file_name)

    file_name = os.path.join("../Models/", file_name)

    with open(file_name + '.h', 'w') as file:
        file.write(header_text)

    with open(file_name + '.cc', 'w') as file:
        file.write(source_text)


def predict_tf(tf_model_path, test_data, test_labels, loss="binary_crossentropy"):
    model = tf.keras.models.load_model(tf_model_path)
    if loss == "binary_crossentropy":
        y_pred = np.rint(model.predict(test_data))

        test_acc = accuracy_score(test_labels, y_pred) * 100
        test_precision = precision_score(test_labels, y_pred, average="binary") * 100
        test_recall = recall_score(test_labels, y_pred, average="binary") * 100
        test_f1 = f1_score(test_labels, y_pred, average="binary") * 100

        print('Full model accuracy is %f%% (Number of test samples=%d)' % (test_acc, len(test_data)))
        print('Full model precision is %f%% (Number of test samples=%d)' % (test_precision, len(test_data)))
        print('Full model recall is %f%% (Number of test samples=%d)' % (test_recall, len(test_data)))
        print('Full model f1-score is %f%% (Number of test samples=%d)' % (test_f1, len(test_data)))

    elif loss == "categorical_crossentropy":
        y_pred = np.argmax(model.predict(test_data), axis=1)
        print(classification_report(test_labels, y_pred, digits=4))

    else:
        raise Exception("Unknown loss function")

    return y_pred


def predict_tflite(tflite_model_path, test_data, test_labels, model_type="Float", loss="binary_crossentropy"):
    # Load test data
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # For quantized models, manually quantize the input data from float to integer
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        if (input_scale, input_zero_point) != (0.0, 0):
            test_data = test_data / input_scale + input_zero_point
            test_data = test_data.astype(input_details["dtype"])

    y_pred = []
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], test_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if loss == "binary_crossentropy":
            top_prediction = output
        elif loss == "categorical_crossentropy":
            top_prediction = output.argmax()
        else:
            raise Exception("Unknown loss function")
        y_pred.append(top_prediction)

    y_pred = np.asarray(y_pred)

    if loss == "binary_crossentropy":
        # If required, dequantized the output layer (from integer to float)
        output_scale, output_zero_point = output_details["quantization"]
        if (output_scale, output_zero_point) != (0.0, 0):
            y_pred = y_pred.astype(np.float32)
            y_pred = (y_pred - output_zero_point) * output_scale

        y_pred = np.rint(y_pred).astype(int)

        print('%s model accuracy is %f%% (Number of test samples=%d)' % (
            model_type, accuracy_score(test_labels, y_pred) * 100, len(test_data)))
        print('%s model precision is %f%% (Number of test samples=%d)' % (
            model_type, precision_score(test_labels, y_pred, average="binary") * 100, len(test_data)))
        print('%s model recall is %f%% (Number of test samples=%d)' % (
            model_type, recall_score(test_labels, y_pred, average="binary") * 100, len(test_data)))
        print('%s model f1-score is %f%% (Number of test samples=%d)' % (
            model_type, f1_score(test_labels, y_pred, average="binary") * 100, len(test_data)))

    elif loss == "categorical_crossentropy":
        print(classification_report(test_labels, y_pred, digits=4))

    return y_pred


def predict_tflite_mtl(tflite_model_path, test_data):
    # Load test data
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    y_pred_detection = []
    y_pred_classification = []

    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], test_data[i])
        interpreter.invoke()
        # Get the output data
        output_data = []
        for j in range(len(output_details)):
            output_data.append(interpreter.get_tensor(output_details[j]['index']))

        y_pred_detection.extend(output_data[0])
        y_pred_classification.extend(np.argmax(output_data[1], axis=1))

    output_scale, output_zero_point = output_details[0]["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        y_pred_detection = np.asarray(y_pred_detection).astype(np.float32)
        y_pred_detection = (y_pred_detection - output_zero_point) * output_scale

    y_pred_detection = np.rint(y_pred_detection).astype(int)
    y_pred_classification = np.asarray(y_pred_classification)

    return y_pred_detection, y_pred_classification


def compare_size(model_tf, batch_size, model_tflite_path, model_tflite_quantized_path):
    size_tf = keras_model_memory_usage_in_bytes(model_tf, batch_size=batch_size)
    size_no_quant_tflite = os.path.getsize(model_tflite_path)
    size_tflite = os.path.getsize(model_tflite_quantized_path)

    return pd.DataFrame.from_records(
        [["TensorFlow", f"{size_tf} bytes", ""],
         ["TensorFlow Lite", f"{size_no_quant_tflite} bytes ", f"(reduced by {size_tf - size_no_quant_tflite} bytes)"],
         ["TensorFlow Lite Quantized", f"{size_tflite} bytes",
          f"(reduced by {size_no_quant_tflite - size_tflite} bytes)"]],
        columns=["Model", "Size", ""], index="Model")


def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multiplied by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory

### MODULES ###

import json
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import time

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print(sys.argv)

if len(sys.argv) != 2:
    print("usage: python3 raspberrypi_cifar10_test_resnet_tflite.py [tflite model path]")
    exit(1)

### CIFAR-10 ###

print("Downloading CIFAR-10...")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

print("Done!")
# print(x_train[0])

# Path to TensorFlow Lite model file
tflite_model_file = sys.argv[1]

### INFERENCE PROCEDURE ###

def infer_with_TF_lite(interpreter, input_details, output_details, raw_image):
    # Get input size
    input_shape = input_details[0]['shape']
    # size = input_shape[:2] if len(input_shape) == 3 else input_shape[1:3]
    # print(input_shape)

    # Preprocess image
    # raw_image = raw_image.resize(size)
    img = np.array(raw_image, dtype=np.float32)
    # print(img.shape)

    # Normalize image
    img = img / 255.

    # Add a batch dimension and a dimension because we use grayscale format
    # Reshape from (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE) to (1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 1)
    input_data = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    
    # Point the data to be used for testing
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Obtain results
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    return predictions

# Create Interpreter (Load TFLite model).
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
# Allocate tensors
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# p = infer_with_TF_lite(interpreter, input_details, output_details, x_test[0])
# print(p)

### PREDICTIONS ###

i = 0
# n = 5
n = x_test.shape[0]
right_preds = 0
y_pred_label = []
results = {}
for raw_image in x_test[:n]:
    
    print(f"img {i}")

    inference_time = time.time()
    result = infer_with_TF_lite(interpreter, input_details, output_details, raw_image)
    inference_time = time.time() - inference_time
    
    # print(f"tfl esp32 image({i}) {inference_time}s: y_pred = {np.argmax(result)} | y_test = {y_test[i][0]} -> {np.argmax(result) == y_test[i][0]}")
    
    pred_label = np.argmax(result)
    y_pred_label.append(pred_label)

    if pred_label == y_test[i][0]:
        right_preds += 1

    # dict para o arquivo de dados de teste
    results[str(i)] = {
        "inference_time": float(inference_time),
        "pred_label": int(pred_label),
        "true_label": int(y_test[i][0])
    }

    i += 1

print("General Accuracy: ", right_preds/n)

### STATISTICS ###

y_true_label = np.reshape(y_test[:n], -1)
y_pred_label = np.array(y_pred_label)
print('Classification Report     : \n\n\n' , classification_report(y_true_label, y_pred_label))

# print(y_true_label, y_pred_label, np.concatenate((y_true_label, y_pred_label), axis=0))
all_y_labels = np.unique(np.concatenate((y_true_label, y_pred_label), axis=0))
cifar10_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
cvt_labels = [cifar10_labels[l] for l in all_y_labels]
cm = confusion_matrix(y_true_label, y_pred_label)
cm_display = ConfusionMatrixDisplay(cm, display_labels=cvt_labels)
cm_display.plot(xticks_rotation="vertical")

# Saving statistics

t_write = time.strftime("%Y%m%d-%H%M%S")
dir_path = f'./results/res-{t_write}/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

c_report = classification_report(y_true_label, y_pred_label, output_dict=True)
c_report_path = os.path.join(dir_path, f'creport-{t_write}.json')
with open(c_report_path, "w") as c_report_file:
    json.dump(c_report, c_report_file, indent=4)

results_path = os.path.join(dir_path, f'results-{t_write}.json')
with open(results_path, "w") as results_file:
    json.dump(results, results_file, indent=4)

cm_path = os.path.join(dir_path, f'cmatrix-{t_write}.png')
cm_display.figure_.savefig(cm_path, bbox_inches="tight")

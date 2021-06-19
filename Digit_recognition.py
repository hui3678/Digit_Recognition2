import cv2 as cv
import os
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + 'Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024*1024), 3)) + 'Megabytes')
    else:
        return print('File size: ' + str(size) + 'bytes')

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data):

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str



mnist = tf.keras.datasets.mnist
(image_train, digit_train), (image_test, digit_test) = mnist.load_data()

image_train = tf.keras.utils.normalize(image_train, axis=1)
image_test = tf.keras.utils.normalize(image_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(image_train, digit_train, epochs=3)

loss, accuracy = model.evaluate(image_test, digit_test)
print(loss)
print(accuracy)

model.save('digits.model')

for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The digit is {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

# write keras save file
keras_file = "Digit_recognition.h5"
keras.models.save_model(model, keras_file)
convert_bytes(get_file_size(keras_file), "MB")

# Convert keras file to tensorflow lite
TF_Lite_Model_file = "Digit_recognition.tflite"
TF_Lite_Converter = tf.lite.TFLiteConverter.from_keras_model(model)
TFLite_model = TF_Lite_Converter.convert()
TFLIte_model_name = TF_Lite_Model_file
open(TFLIte_model_name, "wb").write(TFLite_model)
convert_bytes(get_file_size(TF_Lite_Model_file), "KB")

# Check TF Lite shape
interpreter = tf.lite.Interpreter(model_path=TF_Lite_Model_file)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input shape: ", input_details[0]['shape'])
print("Input type: ", input_details[0]['dtype'])
print("Output shape: ", output_details[0]['shape'])
print("Output type: ", output_details[0]['dtype'])

# Resize TF Lite shape
interpreter.resize_tensor_input(input_details[0]['index'], (10000, 28, 28))
interpreter.resize_tensor_input(output_details[0]['index'], (10000, 10))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input shape: ", input_details[0]['shape'])
print("Input type: ", input_details[0]['dtype'])
print("Output shape: ", output_details[0]['shape'])
print("Output type: ", output_details[0]['dtype'])

# Check accuracy of TF Lite Model
print(image_test.dtype)
image_test_numpy = np.array(image_test, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], image_test_numpy)
interpreter.invoke()
TF_Lite_Model_Prediction = interpreter.get_tensor(output_details[0]['index'])
prediction_class = np.argmax(TF_Lite_Model_Prediction, axis=1)
accuracy = accuracy_score(prediction_class, digit_test)
print('Test accuracy TF lite model is: ', accuracy)

# Write TFLite model to a C source (or header) file
c_model_file = "Digit_recognition"
with open(c_model_file + '.h', 'w') as file:
    file.write(hex_to_c_array(TFLite_model, c_model_file))
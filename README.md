# Digit Recognition on STM32 with X-CUBE-AI
## Introduction
Tiny machine learning (TinyML) is a concept of embedding artificial intelligence on a small pieces of harware. With TinyML, it is possible to deploy the machine learning algorithm to train the network on device and shrink their size to an optimazation form for embedded device without the need of sending the data for cloud computing. Many problems regarding the significance of computing capabilities on data analyzing such as storage capacity, limited central processing unit (CPU) and reduced database performance can be solved through added latency from TinyML. In this tutorial, we will show you how we create a neutal network model using TensorFlow platform and deploy the pre-trained model into STM32F446 chip to run inference for digit recognition.

## Overview
Since microcontrollers have limitted resources compared to desktops, latops and servers, it is necessary to perform model training on a seperate computer before tranferring the model into the microcontroller. We create our model using python script with TensorFlow library and train our model in the computer’s CPU. After that, trained model is saved and converted into TensorFlow Lite and Keras format. The Keras or TensorFLow Lite model file is then imported into the X-CUBE-AI core engine which is the expansion software package of STM32CubeX to generate optimised C-model for STM32 board. Once the model is loaded onto the microcontroller, we can write code to perform inference for digit recognition.

![Overview](https://user-images.githubusercontent.com/82255334/121970873-0fe91500-cdaa-11eb-9848-0f94a02903e4.png)

## Model Training
As mentioned before, our model is trained using Tensor Flow platform. To train the model, sufficient enough of data is needed to be provided to the network for analyzing and prediction. MNISt dataset is used which is the database of handwritten digits that consist of a training set of 60,000 examples, and a test set of 10,000 examples. The digits have been size-normalized and centered in a fixed-size image.

```
mnist = tf.keras.datasets.mnist
(image_train, digit_train), (image_test, digit_test) = mnist.load_data()
```

Then, the input train and test image data is normalized into the range of 0 to 1, so that each input parameter has a similar data distribution to make convergence faster while training the network.

```
image_train = tf.keras.utils.normalize(image_train, axis=1)
image_test = tf.keras.utils.normalize(image_test, axis=1)
```

The model is created by starting to flatten the 28x28 pixels of imput image data, followed with two fully connected middle layers and fully connected output layer which consists of 10 neurons indicating 0 to 9 of output digits.

```
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
```

Then, the model is trained in 3 epochs which means 3 iterations have run for network training. The results are evaluated by printing the loss and accuracy of the output. Since, the loss and accuracy is 0.09 annd 0.97 respectively, the model is good enough to be used.

```
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(image_train, digit_train, epochs=3)

loss, accuracy = model.evaluate(image_test, digit_test)
print(loss)
print(accuracy)
``` 

![Trained_accuracy](https://user-images.githubusercontent.com/82255334/122112165-1d0e0e80-ce53-11eb-9616-0649d965fca4.png)

The trained model is saved and converted into Keras (.h5) and TensorFLow Lite (.tflite) format files.

```
model.save('digits.model')

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
```

## Install X-CUBE-AI
In STM32CubeIDE, click Help > Manage embedded software packages. In the pop-up window, select the STMicroelectronics tab. At X-CUBE-AI, click the drop down arrow and check the most recent version of the Artificial Intelligence package. Click Install Now.


![X-Cube_AI_Install](https://user-images.githubusercontent.com/82255334/122119375-af1a1500-ce5b-11eb-8e5a-600645f50fdd.png)


## Project Configuration
To start a new project, click File > New > STM32 Project. In the Target Selection window, click on the Board Selector tab and search for your development board. Select your board in the Board List.

![Board_select](https://user-images.githubusercontent.com/82255334/122120910-9448a000-ce5d-11eb-8b7a-f45b43f5c605.png)

Click Next. Give your project a name and leave the other options as default (we can use C with X-CUBE-AI).

![Create_Project](https://user-images.githubusercontent.com/82255334/122124211-8e54be00-ce61-11eb-9a2d-761da6658694.PNG)



# Digit Recognition on STM32 with X-CUBE-AI
## Introduction
Tiny machine learning (TinyML) is a concept of embedding artificial intelligence on a small pieces of harware. With TinyML, it is possible to deploy the machine learning algorithm to train the network on device and shrink their size to an optimazation form for embedded device without the need of sending the data for cloud computing. Many problems regarding the significance of computing capabilities on data analyzing such as storage capacity, limited central processing unit (CPU) and reduced database performance can be solved through added latency from TinyML. In this tutorial, we will show you how we create a neutal network model using TensorFlow platform and deploy the pre-trained model into STM32F446 chip to run inference for digit recognition.

## Overview
Since microcontrollers have limitted resources compared to desktops, latops and servers, it is necessary to perform model training on a seperate computer before tranferring the model into the microcontroller. We create our model using python script with TensorFlow library and train our model in the computer’s CPU. After that, we save our trained model and convert it into TensorFlow Lite and Keras format. We import the Keras or TensorFLow Lite model file into the X-CUBE-AI core engine which is the expansion software package of STM32CubeX to generate optimised C-model for STM32 board. Once the model is loaded onto the microcontroller, we can write code to perform inference for digit recognition.

![Overview](https://user-images.githubusercontent.com/82255334/121970873-0fe91500-cdaa-11eb-9848-0f94a02903e4.png)


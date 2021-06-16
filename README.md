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

## Generating C-model using X-CUBE-AI
The X-CUBE-AI core engine is part of the X-CUBE-AI Expansion Package. It provides an automatic and advanced NN mapping tool to generate and deploy an optimized and robust C-model implementation of a pre-trained Neural Network for the embedded systems with limited and constrained hardware resources. The generated STM32 NN library (both specialized
and generic parts) can be directly integrated in an IDE project or makefile-based build system 【】.

Refer to [this tutorial](https://www.digikey.com/en/maker/projects/tinyml-getting-started-with-stm32-x-cube-ai/f94e1c8bfc1e4b6291d0f672d780d2c0), install the X-CUBE-AI, and create the project. Then, we generate our C-model with STM32 NN library by importing the pre-trained Keras or TensorFlow Lite model file into the X-CUBE-AI core engine. After generating the C-model, we should see 5 files in our working directory. Since our network's name is digit_recognition_model, the 5 files generated are digit_recognition_model.h, digit_recognition_model.c, digit_recognition_model_data.h, digit_recognition_model_data.c and digit_recognition_model_config.h.

![C_model_generation](https://user-images.githubusercontent.com/82255334/122131748-38d1de80-ce6c-11eb-9d61-c510d943f4ac.PNG)

The header file digit_recognition_model.h consists of declaration on the input and ouput tensor size as well as tensor dimension (width, height and channel). It also consists of declaration of the main NN functions used to run reference. The digit_recognition_model.c consists of declaration on the weights and biases parameters of the network. The digit_recognition_model_data.c file is where our neural network which consists of huge array of weights stored in. The digit_recognition_model_data.h file consists of functions that initialize the pointer to our activations and weights.

## Run Inference for digit recognition
The following code is written in main.c to perform inference for digit recognition on NUCLEO-F446RE. The sample image data is converted into 1D array before feed into the data buffer for analyzing.

```
int main(void)
{
  /* USER CODE BEGIN 1 */
  char buf[50];
  
  //Sample image data
  //int8_t DIGIT_IMG_DATA_1[784] = {
//	  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,233,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,233,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,246,241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,246,241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,241,246,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,241,246,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,238,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,238,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
//	  };
  //int8_t DIGIT_IMG_DATA_2[784] = {
 // 255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,202,201,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,233,201,255,255,206,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,217,255,255,255,244,239,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,228,248,255,255,255,255,226,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,229,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,224,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,235,244,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,213,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,213,247,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,250,201,245,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,236,162,174,220,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,208,206,205,239,244,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,232,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
//  };
  int8_t DIGIT_IMG_DATA_3[784] = {
  255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,242,208,221,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,212,255,242,232,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,222,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,244,240,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,227,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,223,252,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,182,214,214,238,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,212,228,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,245,209,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,249,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,221,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,233,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,230,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,231,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,254,255,255,255,255,255,255,229,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,202,255,255,255,255,255,251,235,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,199,255,255,255,255,206,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,213,213,214,209,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255
  };
  
  int buf_len = 0;
  ai_error ai_err;
  ai_i32 nbatch;
  int width;
  int height;
  int channels;
  uint32_t timestamp;
  //float y_val;
  float max = 0.0f;
  int32_t imax = 0;
  int ii;
  float y_val;
  unsigned char x;

  // Chunk of memory used to hold intermediate values for neural network
  AI_ALIGNED(4) ai_u8 activations[AI_DIGIT_RECOGNITION_MODEL_DATA_ACTIVATIONS_SIZE];

  // Buffers used to store input and output tensors
  AI_ALIGNED(4) ai_i8 in_data[AI_DIGIT_RECOGNITION_MODEL_IN_1_SIZE_BYTES];
  AI_ALIGNED(4) ai_i8 out_data[AI_DIGIT_RECOGNITION_MODEL_OUT_1_SIZE_BYTES];

  // Pointer to our model
  ai_handle digit_recognition_model = AI_HANDLE_NULL;

  // Initialize wrapper structs that hold pointers to data and info about the
  // data (tensor height, width, channels)
  ai_buffer ai_input[AI_DIGIT_RECOGNITION_MODEL_IN_NUM] = AI_DIGIT_RECOGNITION_MODEL_IN;
  ai_buffer ai_output[AI_DIGIT_RECOGNITION_MODEL_OUT_NUM] = AI_DIGIT_RECOGNITION_MODEL_OUT;

  // Set working memory and get weights/biases from model
  ai_network_params ai_params = {
    AI_DIGIT_RECOGNITION_MODEL_DATA_WEIGHTS(ai_digit_recognition_model_data_weights_get()),
    AI_DIGIT_RECOGNITION_MODEL_DATA_ACTIVATIONS(activations)
  };

  // Set pointers wrapper structs to our data buffers
  ai_input[0].n_batches = 1;
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].n_batches = 1;
  ai_output[0].data = AI_HANDLE_PTR(out_data);
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_CRC_Init();
  MX_TIM14_Init();

  /* USER CODE BEGIN 2 */

  // Start timer/counter
  HAL_TIM_Base_Start(&htim14);

  // Greetings!
  buf_len = sprintf(buf, "\r\n\r\nSTM32 X-Cube-AI test\r\n");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

  // Create instance of neural network
  ai_err = ai_digit_recognition_model_create(&digit_recognition_model, AI_DIGIT_RECOGNITION_MODEL_DATA_CONFIG);
  if (ai_err.type != AI_ERROR_NONE)
  {
    buf_len = sprintf(buf, "Error: could not create NN instance\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    while(1);
  }

  // Initialize neural network
  if (!ai_digit_recognition_model_init(digit_recognition_model, &ai_params))
  {
    buf_len = sprintf(buf, "Error: could not initialize NN\r\n");
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    while(1);
  }

	 for (uint32_t i = 0; i < AI_DIGIT_RECOGNITION_MODEL_IN_1_SIZE; i++)
	 {
		//x = DIGIT_IMG_DATA_3[i];
		//x = (unsigned char) x >> 1;
	   ((ai_float *)in_data)[i] = (ai_float)x;
	 }

	 // Get current timestamp
	 timestamp = htim14.Instance->CNT;

	 // Perform inference
	 nbatch = ai_digit_recognition_model_run(digit_recognition_model, &ai_input[0], &ai_output[0]);
	 if (nbatch != 1) {
	   buf_len = sprintf(buf, "Error: could not run inference\r\n");
	   HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	 }

	  // Read output (predicted y) of neural network
	  for(ii=0;ii<AI_DIGIT_RECOGNITION_MODEL_OUT_1_SIZE;ii++) {
		y_val=((float*)out_data)[ii];
		buf_len = sprintf(buf, "Prob = %f | index = %d\r\n", y_val, ii);
		HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
		if(y_val > max ) { max = y_val; imax = ii; }
		HAL_Delay(500);
	  }

	buf_len = sprintf(buf, "Digit = %d | Inference time: %lu\r\n", imax, htim14.Instance->CNT - timestamp);
	HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	imax=0;
	max=0;
	HAL_Delay(500);
  //unsigned char *img = stbi_load("1.png", &width, &height, &channels, 0);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
	 // Fill input buffer (use test value)

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}
```




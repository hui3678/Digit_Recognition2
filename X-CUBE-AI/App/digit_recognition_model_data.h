/**
  ******************************************************************************
  * @file    digit_recognition_model_data.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri May 14 09:35:19 2021
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */

#ifndef DIGIT_RECOGNITION_MODEL_DATA_H
#define DIGIT_RECOGNITION_MODEL_DATA_H
#pragma once

#include "digit_recognition_model_config.h"
#include "ai_platform.h"

#define AI_DIGIT_RECOGNITION_MODEL_DATA_CONFIG           AI_HANDLE_NULL

#define AI_DIGIT_RECOGNITION_MODEL_DATA_ACTIVATIONS_SIZE     (1024)

#define AI_DIGIT_RECOGNITION_MODEL_DATA_WEIGHTS_SIZE         (473128)

#define AI_DIGIT_RECOGNITION_MODEL_DATA_ACTIVATIONS(ptr_)  \
  AI_BUFFER_OBJ_INIT( \
    AI_BUFFER_FORMAT_U8, \
    1, 1, AI_DIGIT_RECOGNITION_MODEL_DATA_ACTIVATIONS_SIZE, 1, \
    AI_HANDLE_PTR(ptr_) )

#define AI_DIGIT_RECOGNITION_MODEL_DATA_WEIGHTS(ptr_)  \
  AI_BUFFER_OBJ_INIT( \
    AI_BUFFER_FORMAT_U8|AI_BUFFER_FMT_FLAG_CONST, \
    1, 1, AI_DIGIT_RECOGNITION_MODEL_DATA_WEIGHTS_SIZE, 1, \
    AI_HANDLE_PTR(ptr_) )


AI_API_DECLARE_BEGIN

/*!
 * @brief Get network weights array pointer as a handle ptr.
 * @ingroup digit_recognition_model_data
 * @return a ai_handle pointer to the weights array
 */
AI_API_ENTRY
ai_handle ai_digit_recognition_model_data_weights_get(void);


AI_API_DECLARE_END

#endif /* DIGIT_RECOGNITION_MODEL_DATA_H */


/*
 * classifier.h
 *
 *  Created on: Apr 17, 2023
 *      Author: loren
 */

#ifndef INC_MULTI_TASK_NETWORK_H_
#define INC_MULTI_TASK_NETWORK_H_

#include "network.h"
#include "network_data.h"
#include "ai_platform.h"
#include "ai_datatypes_defines.h"
#include <stdio.h>
#include <arm_math.h>

void aiInit(void);

void aiConvertInputFloat_2_Int8(ai_float *in_f32, ai_i8 *out_int8);

void aiConvertOutputInt8_2_Float(ai_i8 *in_int8, ai_float *out_f32);

uint8_t roundInt(float32_t cnnOutput);

uint8_t aiArgmax(ai_i8* cnnOutput);

void multi_task_network(float32_t *input, uint8_t *detectPrediction, uint8_t *classPrediction);

#endif /* INC_MULTI_TASK_NETWORK_H_ */

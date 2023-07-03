/*
 * classifier.h
 *
 *  Created on: Apr 17, 2023
 *      Author: loren
 */

#ifndef INC_CLASSIFIER_H_
#define INC_CLASSIFIER_H_

#include "cnn.h"
#include "cnn_data.h"
#include "ai_platform.h"
#include "ai_datatypes_defines.h"
#include <stdio.h>
#include <arm_math.h>

void aiInit(void);

void aiRun(int8_t *pIn, int8_t *pOut);

void aiConvertInputFloat_2_Int8(ai_float *in_f32, ai_i8 *out_int8);

void aiConvertOutputInt8_2_Float(ai_i8 *in_int8, ai_float *out_f32);

uint8_t aiArgmax(ai_i8* cnnOutput);

void cnn_classifier(float32_t *pSpectrogram, uint8_t *prediction);

#endif /* INC_CLASSIFIER_H_ */

/*
 * bandpass_filter.h
 *
 *  Created on: Mar 23, 2023
 *      Author: lorenzo
 */

#ifndef BANDPASS_FILTER_H_
#define BANDPASS_FILTER_H_

#include "arm_math.h"

#if !defined (NUM_STAGES)
#define NUM_STAGES 2
#endif

void iir_filter(float32_t *inputBuffer, float32_t *outputBuffer, uint32_t inputSize);

void standardize(float32_t *inputBuffer, uint32_t inputSize);

void normalize(float32_t *inputBuffer, uint32_t inputSize);


#endif /* BANDPASS_FILTER_H_ */

/*
 * bandpass_filter.c
 *
 *  Created on: Mar 20, 2023
 *      Author: lorenzo
 */


#include "preprocessing.h"


void iir_filter(float32_t *inputBuffer, float32_t *outputBuffer, uint32_t inputSize)
{
	// second-order section (SOS) computed in Python with lowcut=200Hz and highcut=20KHz
	float32_t m_biquad_coeffs[NUM_STAGES * 5] = {
			0.796954f, 1.593909f, 0.796954f, -1.591040f, -0.661705f,
			1.000000f, -2.000000f, 1.000000f, 1.959704f, -0.960504f,
	};

	float32_t m_biquad_state[NUM_STAGES * 2]; //NUM_STAGES * 4

	arm_biquad_cascade_df2T_instance_f32 iirInstance;
	arm_biquad_cascade_df2T_init_f32(&iirInstance, NUM_STAGES, m_biquad_coeffs, m_biquad_state);

	arm_biquad_cascade_df2T_f32(&iirInstance, inputBuffer, outputBuffer, inputSize);

	//TODO OPTIMIZATION: overwrite inputBuffer, block size is WIN_SIZE the first time, than HOP_SIZE
}


void standardize(float32_t *inputBuffer, uint32_t inputSize)
{
	// zi = (xi - mean(x)) / std(x)

	float32_t mean, std;

	// Compute mean and standard deviation
	arm_mean_f32(inputBuffer, inputSize, &mean);
	arm_std_f32(inputBuffer, inputSize, &std);

    // Standardize input buffer
    arm_offset_f32(inputBuffer, -mean, inputBuffer, inputSize);
    arm_scale_f32(inputBuffer, 1.0f / std, inputBuffer, inputSize);
}


void normalize(float32_t *inputBuffer, uint32_t inputSize)
{
	// zi = (xi - min(x)) / (max(x) - min(x))

	float32_t min, max;
	uint32_t min_index, max_index;

	arm_min_f32(inputBuffer, inputSize, &min, &min_index);
	arm_max_f32(inputBuffer, inputSize, &max, &max_index);

	arm_offset_f32(inputBuffer, -min, inputBuffer, inputSize);
	arm_scale_f32(inputBuffer, 1.0f / (max - min), inputBuffer, inputSize);
}



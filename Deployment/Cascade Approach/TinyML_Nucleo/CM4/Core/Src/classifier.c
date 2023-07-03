/*
 * classifier.c
 *
 *  Created on: Mar 17, 2023
 *      Author: Lorenzo
 */


/* Includes ------------------------------------------------------------------*/
#include "classifier.h"

/* Global handle to reference the instance of the NN */
static ai_handle network = AI_HANDLE_NULL;
static ai_u8 activations[AI_CNN_DATA_ACTIVATIONS_SIZE];
static ai_buffer * ai_input;
static ai_buffer * ai_output;


/*
 * Init function to create and initialize a NN.
 */
void aiInit(void)
{
    ai_error err;

    /* 1 - Create a local array with the addresses of the activations buffers */
    const ai_handle act_addr[] = { activations };

    /* 2 - Create an instance of the NN */
    err = ai_cnn_create_and_init(&network, act_addr, NULL);

    if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type, err.code);
    }

    /* 3 - Initialize input and output buffer */
    ai_input = ai_cnn_inputs_get(network, NULL);
    ai_output = ai_cnn_outputs_get(network, NULL);
}

/*
 * Run function to execute an inference.
 */
void aiRun(int8_t *pIn, int8_t *pOut)
{
    ai_i32 nbatch;
    ai_error err;

    /* 1 - Update IO handlers with the data payload */
    ai_input[0].data = AI_HANDLE_PTR(pIn);
    ai_output[0].data = AI_HANDLE_PTR(pOut);

    /* 2 - Run inference*/
    nbatch = ai_cnn_run(network, ai_input, ai_output);

    if (nbatch != 1) {
    	err = ai_cnn_get_error(network);
    	printf("AI ai_network_run error - type=%d code=%d\r\n", err.type, err.code);
    }
}


void aiConvertInputFloat_2_Int8(ai_float *in_f32, ai_i8 *out_int8)
{
	ai_buffer_format format = ai_input->format;
	ai_size size  = ai_input->size;

	ai_float scale = 0.0;
	ai_float zero_point = 0;

	if (AI_BUFFER_FMT_TYPE_Q != AI_BUFFER_FMT_GET_TYPE(format) &&\
			! AI_BUFFER_FMT_GET_SIGN(format) && 8 != AI_BUFFER_FMT_GET_BITS(format))
	{
		printf("E: expected signed integer 8 bits\r\n");
		return;
	}

	if (AI_BUFFER_META_INFO_INTQ(ai_input->meta_info))
	{
		scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(ai_input->meta_info, 0);
		if (scale != 0.0F) {
			scale= 1.0F/scale ;
		}
		else {
			printf("E: division by zero\r\n");
			return;
		}
		zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(ai_input->meta_info, 0);
	}
	else {
		printf("E: no meta info\r\n");
		return;
	}

	for (uint32_t i = 0; i < size ; i++) {
		out_int8[i] = __SSAT((int32_t) roundf(zero_point + in_f32[i]*scale), 8);
	}
}


void aiConvertOutputInt8_2_Float(ai_i8 *in_int8, ai_float *out_f32)
{
	ai_buffer_format format = ai_output->format;
	ai_size size = ai_output->size;

	ai_float scale = 0.0;
	ai_float zero_point = 0;

	if (AI_BUFFER_FMT_TYPE_Q != AI_BUFFER_FMT_GET_TYPE(format) &&\
			! AI_BUFFER_FMT_GET_SIGN(format) && 8 != AI_BUFFER_FMT_GET_BITS(format))
	{
		printf("E: expected signed integer 8 bits\r\n");
		return;
	}

	if (AI_BUFFER_META_INFO_INTQ(ai_output->meta_info))
	{
		scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(ai_output->meta_info, 0);
		zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(ai_output->meta_info, 0);
	}
	else {
		printf("E: no meta info\r\n");
		return;
	}

	for (uint32_t i = 0; i < size ; i++) {
		out_f32[i] = scale * ((ai_float)(in_int8[i]) - zero_point);
	}
}


uint8_t aiArgmax(ai_i8* cnnOutput)
{
	/* ArgMax to associate NN output with the most likely classification label */
	uint8_t prediction = 0;
	ai_i8 max_out = cnnOutput[0];

	for (uint8_t i = 1; i < AI_CNN_OUT_1_SIZE; i++) {
		if (cnnOutput[i] > max_out) {
			max_out = cnnOutput[i];
			prediction = i;
		}
	}

	return prediction;
}


void cnn_classifier(float32_t *pSpectrogram, uint8_t *prediction)
{
	// input and output are quantized int8
	ai_i8 cnnInput[AI_CNN_IN_1_SIZE];
	ai_i8 cnnOutput[AI_CNN_OUT_1_SIZE];

	aiConvertInputFloat_2_Int8(pSpectrogram, cnnInput);

	aiRun(cnnInput, cnnOutput);

	*prediction = aiArgmax(cnnOutput);
}


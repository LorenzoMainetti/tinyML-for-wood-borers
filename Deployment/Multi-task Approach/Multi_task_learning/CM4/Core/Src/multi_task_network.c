/*
 * multi_task_network.c
 *
 *  Created on: Mar 17, 2023
 *      Author: Lorenzo
 */


/* Includes ------------------------------------------------------------------*/
#include <multi_task_network.h>

/* Global handle to reference the instance of the NN */
static ai_handle network = AI_HANDLE_NULL;
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

/* AI input/output handlers */
static ai_buffer *ai_inputs;
static ai_buffer *ai_outputs;

/* C-table to store the @ of the input buffer */
AI_ALIGNED(32)
static ai_i8 in_data[AI_NETWORK_IN_1_SIZE];

/* data buffer for the output buffers */
AI_ALIGNED(32)
static ai_i8 out_1_data[AI_NETWORK_OUT_1_SIZE];
AI_ALIGNED(32)
static ai_i8 out_2_data[AI_NETWORK_OUT_2_SIZE];

/* C-table to store the @ of the output buffers */
static ai_i8 *out_data[AI_NETWORK_OUT_NUM] = {
  &out_1_data[0],
  &out_2_data[0]
};


/*
 * Init function to create and initialize a NN.
 */
void aiInit(void) {
	/* 1 - Create and initialize network */
	ai_error err;

	const ai_handle act_addr[] = { activations };

	err = ai_network_create_and_init(&network, act_addr, NULL);

	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type, err.code);
	}

	/* 2 - Retrieve IO network infos */
	ai_inputs = ai_network_inputs_get(network, NULL);
	ai_outputs = ai_network_outputs_get(network, NULL);
}


void aiConvertInputFloat_2_Int8(ai_float *in_f32, ai_i8 *out_int8)
{
	ai_buffer_format format = ai_inputs->format;
	ai_size size  = ai_inputs->size;

	ai_float scale = 0.0;
	ai_float zero_point = 0;

	if (AI_BUFFER_FMT_TYPE_Q != AI_BUFFER_FMT_GET_TYPE(format) &&\
			! AI_BUFFER_FMT_GET_SIGN(format) && 8 != AI_BUFFER_FMT_GET_BITS(format))
	{
		printf("E: expected signed integer 8 bits\r\n");
		return;
	}

	if (AI_BUFFER_META_INFO_INTQ(ai_inputs->meta_info))
	{
		scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(ai_inputs->meta_info, 0);
		if (scale != 0.0F) {
			scale= 1.0F/scale ;
		}
		else {
			printf("E: division by zero\r\n");
			return;
		}
		zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(ai_inputs->meta_info, 0);
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
	ai_buffer_format format = ai_outputs->format;
	ai_size size = ai_outputs->size;

	ai_float scale = 0.0;
	ai_float zero_point = 0;

	if (AI_BUFFER_FMT_TYPE_Q != AI_BUFFER_FMT_GET_TYPE(format) &&\
			! AI_BUFFER_FMT_GET_SIGN(format) && 8 != AI_BUFFER_FMT_GET_BITS(format))
	{
		printf("E: expected signed integer 8 bits\r\n");
		return;
	}

	if (AI_BUFFER_META_INFO_INTQ(ai_outputs->meta_info))
	{
		scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(ai_outputs->meta_info, 0);
		zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(ai_outputs->meta_info, 0);
	}
	else {
		printf("E: no meta info\r\n");
		return;
	}

	for (uint32_t i = 0; i < size ; i++) {
		out_f32[i] = scale * ((ai_float)(in_int8[i]) - zero_point);
	}
}


uint8_t roundInt(float32_t cnnOutput)
{
	if (cnnOutput > 0.5) {
		return 1;
	}
	else {
		return 0;
	}
}


uint8_t aiArgmax(ai_i8 *cnnOutput)
{
	/* ArgMax to associate NN output with the most likely classification label */
	uint8_t prediction = 0;
	ai_i8 max_out = cnnOutput[0];

	for (uint8_t i = 1; i < AI_NETWORK_OUT_2_SIZE; i++) {
		if (cnnOutput[i] > max_out) {
			max_out = cnnOutput[i];
			prediction = i;
		}
	}

	return prediction;
}


void multi_task_network(float32_t *input, uint8_t *detectPrediction, uint8_t *classPrediction)
{
    ai_i32 nbatch;
    ai_error err;
    float32_t detectProb = 0.0;

	aiConvertInputFloat_2_Int8(input, in_data);
	ai_inputs[0].data = AI_HANDLE_PTR(&in_data[0]);

	/* Update the AI output handlers */
	ai_outputs[0].data = AI_HANDLE_PTR(out_data[0]);
	ai_outputs[1].data = AI_HANDLE_PTR(out_data[1]);

    nbatch = ai_network_run(network, ai_inputs, ai_outputs);

    if (nbatch != 1) {
    	err = ai_network_get_error(network);
    	printf("AI ai_network_run error - type=%d code=%d\r\n", err.type, err.code);
    }

	// Detection: convert back to float (probability)
	aiConvertOutputInt8_2_Float(out_data[0], &detectProb);
	*detectPrediction = roundInt(detectProb);

	// Classification: take the argmax of the int8 output
	*classPrediction = aiArgmax(out_data[1]);
}


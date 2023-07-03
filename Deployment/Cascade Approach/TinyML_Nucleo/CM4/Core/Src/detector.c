/*
 * detector.c
 *
 *  Created on: 21 mar 2023
 *      Author: Lorenzo
 */

#include "detector.h"


bool rolling_std_detector(float32_t *inputBuffer)
{
	float32_t curr_std;
	float32_t slice[ROLL_WIN_SIZE];

	for(uint16_t i = ROLL_WIN_SIZE; i < WIN_SIZE; i++)
	{
		memcpy(slice, inputBuffer + i - ROLL_WIN_SIZE, ROLL_WIN_SIZE * sizeof(float32_t));

		arm_std_f32(slice, ROLL_WIN_SIZE, &curr_std);

		if(curr_std > THRESHOLD)
		{
			return true;
		}

	}

	return false;
}


bool rolling_std_detector_v2(float32_t *inputBuffer)
{
	float32_t curr_std;
	float32_t slice[ROLL_WIN_SIZE];

    for(uint16_t i = ROLL_WIN_SIZE; i < WIN_SIZE; i++)
    {
		memcpy(slice, inputBuffer + i - ROLL_WIN_SIZE, ROLL_WIN_SIZE * sizeof(float32_t));

		arm_std_f32(slice, ROLL_WIN_SIZE, &curr_std);

		float32_t upper_bound = FACTOR * curr_std;
		float32_t lower_bound = -FACTOR * curr_std;

        if(inputBuffer[i] > upper_bound || inputBuffer[i] < lower_bound){
            return true;
        }
    }
    return false;
}




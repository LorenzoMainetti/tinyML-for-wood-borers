/*
 * detector.h
 *
 *  Created on: 13 apr 2023
 *      Author: loren
 */

#ifndef INC_DETECTOR_H_
#define INC_DETECTOR_H_

#include "arm_math.h"
#include "constants.h"
#include <stdbool.h>
#include <string.h>

#if !defined (ROLL_WIN_SIZE)
#define ROLL_WIN_SIZE 50
#endif

#if !defined (THRESHOLD)
#define THRESHOLD 0.00085f  //1.3f
#endif

#if !defined (FACTOR)
#define FACTOR 5.75f
#endif

bool rolling_std_detector(float32_t *inputBuffer);

bool rolling_std_detector_v2(float32_t *inputBuffer);

#endif /* INC_DETECTOR_H_ */

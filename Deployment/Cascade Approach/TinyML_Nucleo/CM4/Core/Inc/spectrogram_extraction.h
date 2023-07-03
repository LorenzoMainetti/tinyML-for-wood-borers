/*
 * spectrogram_extraction.h
 *
 *  Created on: 14 apr 2023
 *      Author: loren
 */

#ifndef INC_SPECTROGRAM_EXTRACTION_H_
#define INC_SPECTROGRAM_EXTRACTION_H_

#include "arm_math.h"
#include "feature_extraction.h"
#include "window.h"
#include "common_tables.h"
#include "constants.h"

#if !defined (SAMPLE_RATE)
#define SAMPLE_RATE 44100
#endif

#if !defined (NFFT)
#define NFFT 128
#endif

#if !defined (FRAME_LEN)
#define FRAME_LEN 128
#endif

#if !defined (HOP_LEN)
#define HOP_LEN 64
#endif

#if !defined (MIN_FREQ)
#define MIN_FREQ 500
#endif

#if !defined (MAX_FREQ)
#define MAX_FREQ 14000 //16000
#endif

// Number of rows (i.e. frequency bins) is (NFFT/2 + 1).
// After frequency selection is int(MAX_FREQ / freq_res) - int(MIN_FREQ / freq_res)
// where freq_res is SAMPLE_RATE / NFFT
#if !defined (SPECTROGRAM_ROWS)
#define SPECTROGRAM_ROWS 39 //45 (if MAX_FREQ = 16000)
#endif

// Number of columns (i.e. time frames) is given by int((len(input) - FRAME_LEN) / HOP_LEN) + 1
// where len(input) = WIN_SIZE
#if !defined (SPECTROGRAM_COLS)
#define SPECTROGRAM_COLS 33 //45 (if WIN_SIZE = 4410)
#endif


/*
 * @brief Initialization function of structs used with Middleware/ST/STM32_AI_AudioPreprocessing
 */
void spectrogram_init(void);

/*
 * @brief Audio preprocessing function for creating LogSpectrograms
 * @param 	*input points to audio input
 * @return 	None
 */
void spectrogram_run(float32_t *inputBuffer, float32_t *spectrogram);

/*
 * @brief Helper function to create a column of the spectrogram
 * @param 	*input points to spectrogram column
 * @return 	None
 */
void create_column(float32_t *input, float32_t *outputCol);

/*
 * @brief Helper function to convert spectrogram to Log-scale (dB)
 * @param 	*input points to spectrogram column
 * @return 	None
 */
void power_to_dB(float32_t *outputCol);


#endif /* INC_SPECTROGRAM_EXTRACTION_H_ */

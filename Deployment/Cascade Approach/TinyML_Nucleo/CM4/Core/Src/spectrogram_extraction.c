/*
 * spectrogram_extraction.c
 *
 *  Created on: 14 apr 2023
 *      Author: loren
 */

#include "spectrogram_extraction.h"


float32_t scratcher[FRAME_LEN];

static arm_rfft_fast_instance_f32 rfft;
static SpectrogramTypeDef S;


void spectrogram_init(void)
{
	  // Initializing real fourier transform instance (RFFT) struct
	  arm_rfft_fast_init_f32(&rfft, NFFT);

	  // Init Spectrogram struct
	  S.pRfft = &rfft;					         /*!< points to the real FFT instance */
	  S.Type = SPECTRUM_TYPE_POWER;         	 /*!< spectrum type */
	  S.pWindow = (float32_t *) hannWin_128;     /*!< points to the window function. The length must be equal to FrameLen. */
	  S.SampRate = SAMPLE_RATE;                        /*!< sampling rate of the input signal. */
	  S.FrameLen = FRAME_LEN;                    /*!< length of the input signal. */
	  S.FFTLen = NFFT;                           /*!< length of the real FFT. */
	  S.pScratch = (float32_t *) scratcher;		 /*!< point to the temporary calculation buffer of length FFTLen */
}


void spectrogram_run(float32_t *inputBuffer, float32_t *spectrogram)
{
	float32_t input[FRAME_LEN];
	float32_t outputCol[SPECTROGRAM_ROWS];

	uint8_t curr = 0;
	uint8_t SpectrColIndex = 0;

	// Overlapping window to build spectrogram matrix
	for (uint16_t i = 0; i < WIN_SIZE; i += HOP_LEN) {
		for (uint16_t j = i; j < i + FRAME_LEN; j++) {
			input[curr] = inputBuffer[j];	
			curr++;
		}
		
		if (SpectrColIndex < SPECTROGRAM_COLS) {
			create_column((float32_t *) input, (float32_t *) outputCol);

			for (uint8_t j = 0; j < SPECTROGRAM_ROWS; j++) {
			    spectrogram[j * SPECTROGRAM_COLS + SpectrColIndex] = outputCol[j];
			}
		}
		else {
			break;
		}

		curr = 0;
		SpectrColIndex++;
	}
}


void create_column(float32_t *input, float32_t *outputCol)
{
	// Output buffer of original size (with all the frequecy bins)
	float32_t output[(NFFT / 2) + 1];

	// Create spectrogram column
	SpectrogramColumn(&S, input, (float32_t *) output);
	//SpectrogramColumn(&S, input, outputCol);

	// Keep frequency band of interest [MIN_FREQ, MAX_FREQ]
	float32_t freq_res = (float32_t)SAMPLE_RATE / (float32_t)NFFT;
	uint8_t bin_min = (uint8_t)(MIN_FREQ / freq_res);

	memcpy(outputCol, &output[bin_min], SPECTROGRAM_ROWS * sizeof(float32_t));

	// Convert to log scale (dB)
	power_to_dB(outputCol);
}


void power_to_dB(float32_t *outputCol)
{
	float32_t ref = 1.0;
	float32_t topDB = 80.0;

	/* Scaling */
	for (uint8_t i = 0; i < SPECTROGRAM_ROWS; i++) {
		outputCol[i] /= ref;
	}

	/* Avoid log of zero or a negative number */
	for (uint8_t i = 0; i < SPECTROGRAM_ROWS; i++) {
		if (outputCol[i] <= 0.0f) {
			outputCol[i] = FLT_MIN;
		}
	}

	/* Convert power spectrogram to decibel */
	for (uint8_t i = 0; i < SPECTROGRAM_ROWS; i++) {
		outputCol[i] = 10.0f * log10f(outputCol[i]);
	}

	/* Threshold output to -top_dB dB */
	for (uint8_t i = 0; i < SPECTROGRAM_ROWS; i++) {
		outputCol[i] = (outputCol[i] < -topDB) ? (-topDB) : (outputCol[i]);
	}
}






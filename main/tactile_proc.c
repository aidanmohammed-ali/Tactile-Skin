/**
 * @file tactile_proc.c
 * @author Aidan Mohammed-Ali
 * @brief Signal processing logic for tactile skin.
 * @version 0.1
 * @date 2026-05-01
 */

#include "tactile_proc.h"
#include <stddef.h>

static proc_config_t processing_config;

/**
 * @brief Initialise the processing logic.
 * @param config Pointer to the structure containing processing information.
 */
void tactile_proc_init(proc_config_t *config) {
	if (config == NULL) {
		return;
	}
	
	processing_config = *config;
}

/**
 * @brief Simple zero-reference calibration.
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline_buffer Pointer to where the resting state values are saved.
 * @param size Total number of sensor points in the grid.
 */
void tactile_zero_calibration(const uint16_t *raw_frame, uint16_t *baseline_buffer, uint16_t size) {
	for (uint16_t i = 0; i < size; ++i) {
		baseline_buffer[i] = raw_frame[i];
	}
}

/**
 * @brief Performs 3-point quadratic fit logic.
 * @param x_samples Raw sensor inputs.
 * @param y_values Reference values.
 * @param size Total number of sensor points in the grid.
 */
void tactile_fit_curve(float *x_samples[3], float y_values[3], uint16_t size) {
	if (processing_config.curves == NULL || x_samples == NULL) {
		return;
	}
	
	for (uint16_t i = 0; i < size; ++i) {
		float x[3] = { x_samples[0][i], x_samples[1][i], x_samples[2][i] };
		float y[3] = { y_values[0], y_values[1], y_values[2] };
	
		float det = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]);
	
		// If Singular Default to Linear
		if (det == 0.0f) {
			processing_config.curves[i].a = 0.0f;
			processing_config.curves[i].b = 1.0f;
			processing_config.curves[i].c = 0.0f;
			continue;
		}
		
		float inv_det = 1.0f / det;
		processing_config.curves[i].a = ( (y[0] * (x[1] - x[2])) -
										   (y[1] * (x[0] - x[2])) +
										   (y[2] * (x[0] - x[1]))) * inv_det;
							   
		processing_config.curves[i].b = (-(y[0] * (x[1] * x[1] - x[2] * x[2])) +
										   (y[1] * (x[0] * x[0] - x[2] * x[2])) -
										   (y[2] * (x[0] * x[0] - x[1] * x[1]))) * inv_det;
							   
		processing_config.curves[i].c = ( (y[0] * (x[1] * x[1] * x[2] - x[2] * x[2] * x[1])) - 
										   (y[1] * (x[0] * x[0] * x[2] - x[2] * x[2] * x[0])) +
										   (y[2] * (x[0] * x[0] * x[1] - x[1] * x[1] * x[0]))) * inv_det;
		}
}

/**
 * @brief Process a raw frame into clean pressure data.
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline Pointer to where the resting state values are stored.
 * @param processed_frame Pointer to where the processed frame is stored.
 * @param size Total number of sensor points in the grid.
 */
void tactile_process_frame(const uint16_t *raw_frame, uint16_t *processed_frame, uint16_t size) {
	if (raw_frame == NULL || processed_frame == NULL || processing_config.curves == NULL) {
		return;
	}
	
	for (uint16_t i = 0; i < size; ++i) {
		float x = (float)raw_frame[i];
		
		float out = (processing_config.curves[i].a * x * x) +
					(processing_config.curves[i].b * x) +
					(processing_config.curves[i].c);
					
		if (out < processing_config.noise_threshold) {
			out = 0.0f;
		}
		
		if (out > processing_config.max_output) {
			out = processing_config.max_output;
		}
		
		processed_frame[i] = (uint16_t)out;
	}
}

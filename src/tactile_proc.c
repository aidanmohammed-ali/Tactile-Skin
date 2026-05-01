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
 * @brief Capture the resting state of the skin (no touch).
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline_buffer Pointer to where the resting state values are saved.
 * @param size Total number of sensor points in the grid.
 */
void tactile_calibration(const uint16_t *raw_frame, uint16_t *baseline_buffer, uint16_t size) {
	for (uint16_t i = 0; i < size; ++i) {
		baseline_buffer[i] = raw_frame[i]
	}
}

/**
 * @brief Process a raw frame into clean pressure data.
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline Pointer to where the resting state values are stored.
 * @param processed_frame Pointer to where the processed frame is stored.
 * @param size Total number of sensor points in the grid.
 */
void tactile_process_frame(const uint16_t *raw_frame, const uint16_t *baseline, uint16_t *processed_frame, uint16_t size) {
	for (uint16_t i = 0; i < size; ++i) {
		if (raw_frame[i] > baseline[i]) {
			uint16_t delta = raw_frame[i] - baseline[i];
			
			processed_frame[i] = (delta > processing_config.noise_threshold) ? delta : 0;
		} else {
			processed_frame[i] = 0;
		}
	}
}

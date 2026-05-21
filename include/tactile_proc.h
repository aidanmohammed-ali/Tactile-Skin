/**
 * @file tactile_proc.h
 * @author Aidan Mohammed-Ali
 * @brief Signal processing logic for tactile skin.
 * @version 0.1
 * @date 2026-05-01
 */

#ifndef TACTILE_PROC_H
#define TACTILE_PROC_H

#include <stdint.h>

/** Curve Fitting Coefficients **/
typedef struct {
	float a;
	float b;
	float c;
} curve_params_t;

/** Processing Configuration **/
typedef struct {
	uint16_t noise_threshold;
	uint16_t sensitivity;
	uint16_t max_output;
	curve_params_t *curves;
} proc_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialise the processing logic.
 * @param config Pointer to the structure containing processing information.
 */
void tactile_proc_init(proc_config_t *config);

/**
 * @brief Simple zero-reference calibration.
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline_buffer Pointer to where the resting state values are saved.
 * @param size Total number of sensor points in the grid.
 */
void tactile_zero_calibration(const uint16_t *raw_frame, uint16_t *baseline_buffer, uint16_t size);

/**
 * @brief Performs 3-point quadratic fit logic.
 * @param x_samples Raw sensor inputs.
 * @param y_values Reference values.
 * @param size Total number of sensor points in the grid.
 */
void tactile_fit_curve(uint16_t *x_samples[3], float y_values[3], uint16_t size);

/**
 * @brief Process a raw frame into clean pressure data.
 * @param raw_frame Pointer to the start of the array containing sensor data.
 * @param baseline Pointer to where the resting state values are stored.
 * @param processed_frame Pointer to where the processed frame is stored.
 * @param size Total number of sensor points in the grid.
 */
void tactile_process_frame(const uint16_t *raw_frame, uint16_t *processed_frame, uint16_t size);

#ifdef __cplusplus
}
#endif

#endif // TACTILE_PROC_H

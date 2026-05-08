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
 * @brief Processing initialisation function.
 */
void tactile_proc_init(proc_config_t *config);

/**
 * @brief Simple zero-reference calibration.
 */
void tactile_zero_calibration(const uint16_t *raw_frame, uint16_t *baseline_buffer, uint16_t size);

/**
 * @brief Performs 3-point quadratic fit.
 */
void tactile_fit_curve(float *x_samples[3], float y_values[3], uint16_t size);

/**
 * @brief Process a raw frame into clean pressure data.
 */
void tactile_process_frame(const uint16_t *raw_frame, uint16_t *processed_frame, uint16_t size);

#ifdef __cplusplus
}
#endif

#endif // TACTILE_PROC_H

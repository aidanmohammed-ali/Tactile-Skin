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

/** Processing Configuration **/
typedef struct {
	uint16_t noise_threshold;
	uint16_t sensitivity;
	uint16_t max_output;
} proc_config_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Processing initialisation function.
 */
void tactile_proc_init(proc_config_t *config);

/**
 * @brief Capture the resting state of the skin (no touch).
 */
void tactile_calibration(const uint16_t *raw_frame, uint16_t *baseline_buffer, uint16_t size);

/**
 * @brief Process a raw frame into clean pressure data.
 */
void tactile_process_frame(const uint16_t *raw_frame, const uint16_t *baseline, uint16_t *processed_frame, uint16_t size);

#ifdef __cplusplus
}
#endif

#endif // TACTILE_PROC_H

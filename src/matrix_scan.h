/**
 * @file matrix_scan.h
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @version 0.1
 * @date 2026-04-29
 */

#ifndef MATRIX_SCAN_H
#define MATRIX_SCAN_H

#include <stdint.h>

/** Hardware Configuration **/
// Row Mux Select Pins
#define PIN_ROW_S0 12
#define PIN_ROW_S1 13
#define PIN_ROW_S2 14
#define PIN_ROW_S3 15
#define PIN_ROW_S4 16

// Row Mux Select Pins
#define PIN_COL_S0 17
#define PIN_COL_S1 18
#define PIN_COL_S2 19
#define PIN_COL_S3 20
#define PIN_COL_S4 21

/** Matrix Dimensions **/
#define MATRIX_ROWS		32
#define MATRIX_COLS		32
#define MATRIX_SIZE		(MATRIX_ROWS * MATRIX_COLS)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs a full scan of the matrix.
 */
void matrix_scan_grid(uint16_t *buffer);

/**
 * @brief Sets the hardware row multiplexer to a specific address.
 */
void set_mux_row(uint8_t addr);

/**
 * @brief Sets the hardware column multiplexer to a specific address.
 */
void set_mux_col(uint8_t addr);

/**
 * @brief Bridge function to set physical GPIO states.
 */
void set_gpio_state(uint8_t gpio_pin, uint8_t state);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_SCAN_H

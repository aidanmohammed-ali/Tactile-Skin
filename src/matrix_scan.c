/**
 * @file matrix_scan.c
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @version 0.1
 * @date 2026-04-29
 */

#include "matrix_scan.h"
#include <stddef.h>

/**
 * @brief Implementation of the grid scanning logic.
 * @param buffer Pointer to the memory where values will be stored.
 * This function iterates through the rows and columns defined in the header.
 */
void matrix_scan_grid(uint16_t *buffer) {
	if (buffer == NULL) {
		return;
	}
	
	for (uint8_t r = 0; r < MATRIX_ROWS; ++r) {
		set_mux_row(r);
		
		for (uint8_t c = 0; c < MATRIX_COLS; ++c) {
			set_mux_col(c);
			
			// TODO: Read Value
			
			// Calculate index for buffer
			size_t index = (r * MATRIX_COLS) + c;
			buffer[index] = 0; // Placeholder
		}
	}
}

static const uint8_t row_pins[] = {
	PIN_ROW_S0, PIN_ROW_S1, PIN_ROW_S2, PIN_ROW_S3, PIN_ROW_S4
};

static const uint8_t col_pins[] = {
	PIN_COL_S0, PIN_COL_S1, PIN_COL_S2, PIN_COL_S3, PIN_COL_S4
};

/**
 * @brief Logic to translate a row number into pin states.
 * @param addr The row address
 */
void set_mux_row(uint8_t addr) {
	for (uint8_t i = 0; i < 5; ++i) {
		uint8_t state = (addr >> i) & 0x01;
		
		set_gpio_state(row_pins[i], state);
	}
}

/**
 * @brief Logic to translate a column number into pin states.
 * @param addr The column address
 */
void set_mux_col(uint8_t addr) {
	for (uint8_t i = 0; i < 5; ++i) {
		uint8_t state = (addr >> i) & 0x01;
		
		set_gpio_state(col_pins[i], state);
	}
}

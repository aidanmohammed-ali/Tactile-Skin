/**
 * @file matrix_scan.h
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @version 0.1
 * @date 2026-04-29
 */

#include "matrix_scan.h"

/**
 * @brief Implementation of the grid scanning logic.
 * This function iterates through the rows and columns defined in the header.
 */
void matrix_scan_grid(uint16_t *buffer) {
	for (uint8_t r = 0; r < MATRIX_ROWS; r++) {
		// TODO: Set ROW Multiplexer Address to 'r'
		
		for (uint8_t c = 0; c < MATRIX_COLS; c++) {
			// TODO: Set COLUMN Multiplexer Address to 'c'
			// TODO: Read Value
			
			// Calculate index for buffer
			size_t index = (r * MATRIX_COLS) + c;
		}
	}
}

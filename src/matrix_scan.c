/**
 * @file matrix_scan.c
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @version 0.1
 * @date 2026-04-29
 */

#include "matrix_scan.h"
#include <stddef.h>

static matrix_config_t board_config;

/**
 * @brief Initialise the hardware pins on the microcontroller.
 * @param config Pointer to the structure containing hardware pin assignments and dimensions.
 */
void matrix_init(matrix_config_t* config) {
	if (config == NULL) {
		return;
	}
	
	board_config = *config;
	uint8_t r_channels = (1 << board_config.num_row_addr_pins);
	uint8_t c_channels = (1 << board_config.num_col_addr_pins);
	
	board_config.active_rows = r_channels * (board_config.num_row_en_pins > 0 ? board_config.num_row_en_pins : 1);
	board_config.active_cols = c_channels * (board_config.num_col_en_pins > 0 ? board_config.num_col_en_pins : 1);
	
	if (board_config.active_rows > MAX_DIMENSION ||
			board_config.active_cols > MAX_DIMENSION) {
		return;	
	}
}

/**
 * @brief Implementation of the grid scanning logic.
 * @param buffer Pointer to the memory where values will be stored.
 * This function iterates through the rows and columns defined in the header.
 */
void matrix_scan_grid(uint16_t *buffer) {
	if (buffer == NULL) {
		return;
	}
	
	for (uint16_t r = 0; r < board_config.active_rows; ++r) {
		set_mux_row(r);
		
		for (uint16_t c = 0; c < board_config.active_cols; ++c) {
			set_mux_col(c);
			
			size_t index = (r * board_config.active_cols) + c;
			buffer[index] = get_sensor_value();
		}
	}
}

/**
 * @brief Logic to translate a row number into pin states.
 * @param addr The row address
 */
void set_mux_row(uint8_t addr) {
	uint8_t channels_per_chip = (1 << board_config.num_row_addr_pins);
	
	uint8_t target_chip = addr / channels_per_chip;
	uint8_t local_addr = addr % channels_per_chip;
	
	for (uint8_t i = 0; i < board_config.num_row_en_pins; ++i) {
		// Active Low (0 = ON, 1 = OFF)
		uint8_t state = (i == target_chip) ? 0 : 1;
		set_gpio_state(board_config.row_en_pins[i], state);
	}
	
	for (uint8_t i = 0; i < board_config.num_row_addr_pins; ++i) {
		uint8_t state = (local_addr >> i) & 0x01;
		set_gpio_state(board_config.row_addr_pins[i], state);
	}
}

/**
 * @brief Logic to translate a column number into pin states.
 * @param addr The column address
 */
void set_mux_col(uint8_t addr) {
	uint8_t channels_per_chip = (1 << board_config.num_col_addr_pins);
	
	uint8_t target_chip = addr / channels_per_chip;
	uint8_t local_addr = addr % channels_per_chip;
	
	for (uint8_t i = 0; i < board_config.num_col_en_pins; ++i) {
		// Active Low (0 = ON, 1 = OFF)
		uint8_t state = (i == target_chip) ? 0 : 1;
		set_gpio_state(board_config.col_en_pins[i], state);
	}
	
	for (uint8_t i = 0; i < board_config.num_col_addr_pins; ++i) {
		uint8_t state = (local_addr >> i) & 0x01;
		set_gpio_state(board_config.col_addr_pins[i], state);
	}
}

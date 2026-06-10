/**
 * @file matrix_scan.h
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @date 2026-04-29
 */

#ifndef MATRIX_SCAN_H
#define MATRIX_SCAN_H

#include <stdint.h>

/** Hardware Configuration **/
typedef struct {
	// Hardware Control
	void (*set_row_func)(uint8_t addr);
	void (*set_col_func)(uint8_t addr);
	
	// Synchronisation Hooks
	void (*trigger_scan_func)(void);
	void (*wait_ready_func)(void);
	void (*clear_interrupt_func)(void);
	
	uint8_t use_parallel_scan;
	
	// Address Pins
	uint8_t row_addr_pins[8];
	uint8_t col_addr_pins[8];
	uint8_t num_row_addr_pins;
	uint8_t num_col_addr_pins;
	
	// Enable Pins
	uint8_t row_en_pins[8];
	uint8_t col_en_pins[8];
	uint8_t num_row_en_pins;
	uint8_t num_col_en_pins;
	
	// Analog Pins
	uint8_t analog_pins[8];
	uint8_t num_analog_pins;
	
	// Dimensions
	uint16_t active_rows;
	uint16_t active_cols;
	
	// Settling Time (in microseconds)
	uint16_t settle_time_us;
} matrix_config_t;

/** Maximum Hardware Capacity **/
#define MAX_DIMENSION		16
#define MAX_BUFFER_SIZE		(MAX_DIMENSION * MAX_DIMENSION)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialise the hardware pins on the microcontroller.
 * @param config Pointer to the structure containing hardware pin assignments and dimensions.
 */
void matrix_init(matrix_config_t* config);

/**
 * @brief Implementation of the grid scanning logic.
 * @param buffer Pointer to the memory where values will be stored.
 * This function iterates through the rows and columns defined in the header.
 */
void matrix_scan_grid(uint16_t *buffer);

/**
 * @brief Logic to translate a row number into pin states.
 * @param addr The row address
 */
void set_mux_row(uint8_t addr);

/**
 * @brief Logic to translate a column number into pin states.
 * @param addr The column address
 */
void set_mux_col(uint8_t addr);

/**
 * @brief Bridge function to set physical GPIO states.
 * @param gpio_pin Number of the pin being used.
 * @param state Value being applied to the pin.
 */
void set_gpio_state(uint8_t gpio_pin, uint8_t state);

/**
 * @brief Bridge function to read physical analog pins.
 * @retval 16-bit value read from analog pins.
 */
uint16_t read_analog(void);

/**
 * @brief Bridge function to read two sensors at once.
 * @param val_a Pointer to location of first value to be read.
 * @param val_b Pointer to location of second value to be read.
 */
void get_sensor_pair(uint16_t *val_a, uint16_t *val_b);

/**
 * @brief Performs a parallel scan.
 * @param buffer Pointer to the memory where values will be stored.
 */
void matrix_scan_parallel(uint16_t* buffer);

/**
 * @brief Bridge function for microsecond delays.
 * @param us Delay time in microseconds.
 */
void delay_us(uint16_t us);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_SCAN_H

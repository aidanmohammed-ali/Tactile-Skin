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
typedef struct {
	// Hardware Control
	void (*set_row_func)(uint8_t addr);
	void (*set_col_func)(uint8_t addr);
	
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
 * @brief Hardware initialisation function.
 */
void matrix_init(matrix_config_t* config);

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

/**
 * @brief Bridge function to read physical analog pins.
 */
uint16_t read_analog(void);

/**
 * @brief Bridge function to read two sensors at once.
 */
void get_sensor_pair(uint16_t *val_a, uint16_t *val_b);

/**
 * @brief Performs a parallel scan.
 */
void matrix_scan_parallel(uint16_t* buffer);

/**
 * @brief Bridge function for microsecond delays.
 */
void delay_us(uint16_t us);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_SCAN_H

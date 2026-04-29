/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Hardware-specific implementation for tactile skin.
 * * This file acts as the Hardware Driver Layer.
 * @version 0.1
 * @date 2026-04-29
 */

#include <Arduino.h>

extern "C" {
	#include "matrix_scan.h"
}

/**
 * @brief Logic called by matrix_scan.c to toggle pins.
 */
void set_gpio_state(uint8_t gpio_pin, uint8_t state) {
	digitalWrite(gpio_pin, state);
}

/**
 * @brief Placeholder for reading logic.
 */
uint16_t read_value(void) {
	return 0;
}

void setup() {
	Serial.begin(115200);
	while (!Serial);
	
	Serial.println("Tactile Skin Initialising");
	
	// Initialise all Mux Select Pins as OUTPUT
	const uint8_t row_mux[] = {PIN_ROW_S0, PIN_ROW_S1, PIN_ROW_S2, PIN_ROW_S3, PIN_ROW_S4};
	const uint8_t col_mux[] = {PIN_COL_S0, PIN_COL_S1, PIN_COL_S2, PIN_COL_S3, PIN_COL_S4};
	
	for (int i = 0; i < 5; ++i) {
		pinMode(row_mux[i], OUTPUT);
		pinMode(col_mux[i], OUTPUT);
	}
}

void loop() {
	static uint16_t sensor_data[MATRIX_ROWS * MATRIX_COLS];
	
	matrix_scan_grid(sensor_data);
	
	delay(100);
}

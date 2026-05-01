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
	#include "tactile_proc.h"
}

matrix_config_t skin_config;
proc_config_t proc_config;

/**
 * @brief Logic called by matrix_scan.c to toggle pins.
 */
void set_gpio_state(uint8_t gpio_pin, uint8_t state) {
	digitalWrite(gpio_pin, state);
}

/**
 * @brief Bridge logic to retrieve data.
 */
uint16_t get_sensor_value(void) {
	// TODO: Insert reading logic.
	return 0;
}

/**
 * @brief Bridge logic for delay.
 */
void delay_us(uint16_t us) {
	delayMicroseconds(us);
}

uint16_t *sensor_data = NULL;
uint16_t *baseline = NULL;

void setup() {
	Serial.begin(115200);
	while (!Serial);
	
	Serial.println("Tactile Skin Initialising");
	
	// Initialise all Mux Select Pins as OUTPUT
	// TODO: Define Row and Column Hardware.
	
	for (int i = 0; i < skin_config.num_row_addr_pins; ++i) {
		pinMode(skin_config.row_addr_pins[i], OUTPUT);
	}
	
	for (int i = 0; i < skin_config.num_col_addr_pins; ++i) {
		pinMode(skin_config.col_addr_pins[i], OUTPUT);
	}
	
	uint16_t grid_size = skin_config.active_rows * skin_config.active_cols;
	
	sensor_data = (uint16_t*)malloc(grid_size * sizeof(uint16_t));
	baseline = (uint16_t*)malloc(grid_size * sizeof(uint16_t));
	
	if (sensor_data == NULL || baseline == NULL) {
		Serial.println("Fatal: Out of Memory");
		while(1);
	}
	
	matrix_init(&skin_config);
	tactile_proc_init(&proc_config);
	
	Serial.print("Grid Initialised: ");
	Serial.print(skin_config.active_rows);
	Serial.print("x");
	Serial.println(skin_config.active_cols);
}

void loop() {
	static bool is_calibrated = false;
	uint16_t grid_size = skin_config.active_rows * skin_config.active_cols;
	
	matrix_scan_grid(sensor_data);
	
	if (!is_calibrated) {
		tactile_calibration(sensor_data, baseline, grid_size);
		is_calibrated = true;
		Serial.println("Calibration Complete");
	}
	
	tactile_process_frame(sensor_data, baseline, sensor_data, grid_size);
	
	delay(50);
}

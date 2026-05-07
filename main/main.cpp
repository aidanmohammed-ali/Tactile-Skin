/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Hardware-specific implementation for tactile skin.
 * * This file acts as the Hardware Driver Layer.
 * @version 0.1
 * @date 2026-04-29
 */

#include <stdio.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_adc/adc_oneshot.h"

extern "C" {
	#include "matrix_scan.h"
	#include "tactile_proc.h"
}

static const char *TAG = "TactileSkin";

matrix_config_t skin_config = {};
proc_config_t proc_config = {};

adc_oneshot_unit_handle_t adc_handle;

/**
 * @brief Map function to convert GPIO to ADC Channel.
 */
adc_channel_t map_gpio_to_adc(uint8_t gpio_pin) {
	switch (gpio_pin) {
		case 36:
			return ADC_CHANNEL_0;
		case 37:
			return ADC_CHANNEL_1;
		case 38:
			return ADC_CHANNEL_2;
		case 39:
			return ADC_CHANNEL_3;
		case 32:
			return ADC_CHANNEL_4;
		case 33:
			return ADC_CHANNEL_5;
		case 34:
			return ADC_CHANNEL_6;
		case 35:
			return ADC_CHANNEL_7;
		default:
			return (adc_channel_t)-1;
	}
}

/**
 * @brief Logic called by matrix_scan.c to toggle pins.
 */
void set_gpio_state(uint8_t gpio_pin, uint8_t state) {
	gpio_set_level((gpio_num_t)gpio_pin, state);
}

/**
 * @brief Generic analog GPIO read function.
 */
uint16_t read_analog(uint8_t gpio_pin) {
	int raw_out = 0;
	
	adc_channel_t channel = map_gpio_to_adc(gpio_pin);
	
	if (channel != (adc_channel_t)-1) {
		adc_oneshot_read(adc_handle, channel, &raw_out);
	}
	
	return (uint16_t)raw_out;
}

/**
 * @brief Bridge logic to retrieve data.
 */
uint16_t get_sensor_value(void) {
	return read_analog(skin_config.analog_pins[0]);
}

/**
 * @brief Bridge logic for delay.
 */
void delay_us(uint16_t us) {
	esp_rom_delay_us(us);
}

uint16_t *sensor_data = NULL;
uint16_t *baseline = NULL;

#ifdef __cplusplus
extern "C" {
#endif

void app_main(void) {
	ESP_LOGI(TAG, "Tactile Skin Initialising");
	
	// Initialise ADC Unit
	adc_oneshot_unit_init_cfg_t init_config = {
		.unit_id = ADC_UNIT_1,
		.ulp_mode = ADC_ULP_MODE_DISABLE,
	};
	ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config, &adc_handle));
	
	// Initialise all Mux Select Pins as OUTPUT
	skin_config.num_row_addr_pins = 3;
	skin_config.row_addr_pins[0] = 12;
	skin_config.row_addr_pins[1] = 13;
	skin_config.row_addr_pins[2] = 14;

	skin_config.num_col_addr_pins = 3;
	skin_config.col_addr_pins[0] = 25;
	skin_config.col_addr_pins[1] = 26;
	skin_config.col_addr_pins[2] = 27;
	
	skin_config.num_analog_pins = 2;
	skin_config.analog_pins[0] = 36;
	skin_config.analog_pins[1] = 37;
	
	skin_config.settle_time_us = 10;
	
	matrix_init(&skin_config);
	
	adc_oneshot_chan_cfg_t adc_cfg = {
		.bitwidth = ADC_BITWIDTH_DEFAULT,
		.atten = ADC_ATTEN_DB_12,
	};
	
	for (int i = 0; i < skin_config.num_analog_pins; ++i) {
		adc_channel_t channel = map_gpio_to_adc(skin_config.analog_pins[i]);
		if (channel != (adc_channel_t)-1) {
			ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, channel, &adc_cfg));
		}
	}
	
	for (int i = 0; i < skin_config.num_row_addr_pins; ++i) {
		gpio_reset_pin((gpio_num_t)skin_config.row_addr_pins[i]);
		gpio_set_direction((gpio_num_t)skin_config.row_addr_pins[i], GPIO_MODE_OUTPUT);
	}
	
	for (int i = 0; i < skin_config.num_col_addr_pins; ++i) {
		gpio_reset_pin((gpio_num_t)skin_config.col_addr_pins[i]);
		gpio_set_direction((gpio_num_t)skin_config.col_addr_pins[i], GPIO_MODE_OUTPUT);
	}
	
	// Memory Allocation
	uint16_t grid_size = skin_config.active_rows * skin_config.active_cols;
	
	sensor_data = (uint16_t*)malloc(grid_size * sizeof(uint16_t));
	baseline = (uint16_t*)malloc(grid_size * sizeof(uint16_t));
	
	if (sensor_data == NULL || baseline == NULL) {
		ESP_LOGE(TAG, "Fatal: Out of Memory");
		return;
	}
	
	ESP_LOGI(TAG, "Grid Initialised: %dx%d", skin_config.active_rows, skin_config.active_cols);
	
	tactile_proc_init(&proc_config);
	
	bool is_calibrated = false;
	
	// Main Loop
	while (1) {
		matrix_scan_grid(sensor_data);
		
		if (!is_calibrated) {
			tactile_calibration(sensor_data, baseline, grid_size);
			is_calibrated = true;
			ESP_LOGI(TAG, "Calibration Complete");
		}
		
		tactile_process_frame(sensor_data, baseline, sensor_data, grid_size);
		
		vTaskDelay(pdMS_TO_TICKS(50));
	}
}

#ifdef __cplusplus
}
#endif

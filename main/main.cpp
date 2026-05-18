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
#include <stm32f4xx_hal.h>

extern "C" {
	#include "matrix_scan.h"
	#include "tactile_proc.h"
}

SPI_HandleTypeDef hspi1; // CDC A
SPI_HandleTypeDef hspi2; // CDC B

volatile uint8_t current_column = 0;

// Hardware Initialisation
void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_SPI1_Init(void);
void MX_SPI2_Init(void);

/**
 * @brief Bridge function to set physical GPIO states.
 */
extern "C" void set_gpio_state(uint8_t gpio_pin, uint8_t state) {
	GPIO_PinState s = (state) ? GPIO_PIN_SET : GPIO_PIN_RESET;
	
	if (gpio_pin == 0) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, s);
	if (gpio_pin == 1) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, s);
	if (gpio_pin == 2) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_2, s);
}

/**
 * @brief Bridge function to read two sensors at once.
 */
extern "C" void get_sensor_pair(uint16_t *val_a, uint16_t *val_b) {
	uint16_t reg_addr = 0x00B + current_column;
	uint16_t tx_command = 0xE000 | 0x0400 | reg_addr;
	
	// Read sensor A
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
	HAL_SPI_Transmit(&hspi1, &tx_command, 1, 10);
	HAL_SPI_Receive(&hspi1, val_a, 1, 10);
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
	
	// Read sensor B
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_15, GPIO_PIN_RESET);
	HAL_SPI_Transmit(&hspi2, &tx_command, 1, 10);
	HAL_SPI_Receive(&hspi2, val_b, 1, 10);
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_15, GPIO_PIN_SET);
}

/**
 * @brief Set the active CDC column channel for the matrix scanner.
 * @param col_addr The column index requested by the library.
 */
extern "C" void set_cdc_channel(uint8_t col_addr) {
	current_column = col_addr;
}

/**
 * @brief Initialise the hardware DWT cycle counter for precision delays.
 */
void DWT_Delay_Init(void) {
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	DWT->CYCCNT = 0;
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/**
 * @brief Microsecond delay block.
 * @param us Number of microseconds to pause.
 */
void delay_us(uint16_t us) {
	uint32_t start_tick = DWT->CYCCNT;
	
	uint32_t target_ticks = (uint32_t)us * (SystemCoreClock / 1000000);
	while ((DWT->CYCCNT - start_tick) < target_ticks);
}

int main(void) {
	// Initialise STM32
	HAL_Init();
	SystemClock_Config();	
	DWT_Delay_Init();
	
	// Initialise peripheral ports
	MX_GPIO_Init();
	MX_SPI1_Init();
	MX_SPI2_Init();
	
	// TODO: AD7142_Init();
	
	// Configure tactile geometry
	matrix_config_t skin_config = {};
	skin_config.num_row_addr_pins = 3;
	skin_config.num_col_addr_pins = 4;
	skin_config.num_row_en_pins = 0;
	skin_config.num_col_en_pins = 0;
	
	skin_config.row_addr_pins[0] = 0;
	skin_config.row_addr_pins[1] = 1;
	skin_config.row_addr_pins[2] = 2;
	
	skin_config.set_row_func = set_mux_row;
	skin_config.set_col_func = set_cdc_channel;
	
	matrix_init(&skin_config);
	
	// Configure signal processing parameters
	curve_params_t skin_curves[128];
	
	proc_config_t skin_proc = {};
	// TODO: Assign actual values
	skin_proc.noise_threshold = 50;
	skin_proc.sensitivity = 100;
	skin_proc.max_output = 4095;
	skin_proc.curves = skin_curves;
	
	tactile_proc_init(&skin_proc);
	
	// Sensor calibration
	uint16_t *weight_low = (uint16_t*)malloc(128 * sizeof(uint16_t));
	uint16_t *weight_mid = (uint16_t*)malloc(128 * sizeof(uint16_t));
	uint16_t *weight_high = (uint16_t*)malloc(128 * sizeof(uint16_t));
	
	// TODO: Handle prompting to user
	
	float y_targets[3] = { 0.0f, 100.0f, 500.0f };
	uint16_t *x_samples[3] = { weight_low, weight_mid, weight_high };
	tactile_fit_curve(x_samples, y_targets, 128);
	
	free(weight_low);
	free(weight_mid);
	free(weight_high);
	
	// Frame buffers
	uint16_t sensor_data[128] = {0};
	uint16_t processed_data[128] = {0};
	
	while (1) {
		matrix_scan_parallel(sensor_data);
		tactile_process_frame(sensor_data, processed_data, 128);
		HAL_Delay(10);
	}
}

/**
 * @brief Configure the system clock source and bus dividers.
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = {0};
	RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
	
	// Enable power interface clock
	__HAL_RCC_PWR_CLK_ENABLE();
	__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
	
	// Set physical 25MHz external crystal as clock source
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	
	// Enable PLL
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;	
	
	// Multiplication math
	RCC_OscInitStruct.PLL.PLLM = 25;
	RCC_OscInitStruct.PLL.PLLN = 200;
	RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
	
	// Physically apply configuration
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		while(1);
	}
	
	// Configure clock
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
									RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	
	// Set dividers
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;	
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
	
	// Physically apply configuration
	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK) {
		while(1);
	}
}

/**
 * @brief Initialise GPIO pins.
 */
void MX_GPIO_Init(void) {
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	
	// Enable internal clocks for GPIOA and GPIOB
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();
	
	// Set default starting states for MUX
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2, GPIO_PIN_RESET);
	
	GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	
	// Physically apply configuration
	HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
	
	// Set default starting states for CS
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4 | GPIO_PIN_15, GPIO_PIN_SET);
	
	GPIO_InitStruct.Pin = GPIO_PIN_4 | GPIO_PIN_15;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	
	// Physically apply configuration
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
	
	// Configure INT
	GPIO_InitStruct.Pin = GPIO_PIN_3;
	GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	
	// Physically apply configuration
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
	
	// Listen to interrupt channel
	HAL_NVIC_SetPriority(EXTI3_IRQn, 5, 0);
	HAL_NVIC_Enable(EXTI3_IRQn);
}

/**
 * @brief Initialise SPI1 for CDC1.
 */
void MX_SPI1_Init(void) {
	hspi1.Instance = SPI1;
	
	// Configuration for SPI
	hspi1.Init.Mode = SPI_MODE_MASTER;
	hspi1.Init.Direction = SPI_DIRECTION_2LINES;
	hspi1.Init.DataSize = SPI_DATASIZE_16BIT;
	hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
	hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
	hspi1.Init.NSS = SPI_NSS_SOFT;
	
	// Set speed limit
	hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;
	hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
	hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
	hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	
	// Physically apply configuration
	if (HAL_SPI_Init(&hspi1) != HAL_OK) {
		while(1);
	}
}

/**
 * @brief Initialise SPI2 for CDC2.
 */
void MX_SPI2_Init(void) {
	hspi2.Instance = SPI2;
	
	// Configuration for SPI
	hspi2.Init.Mode = SPI_MODE_MASTER;
	hspi2.Init.Direction = SPI_DIRECTION_2LINES;
	hspi2.Init.DataSize = SPI_DATASIZE_16BIT;
	hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
	hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
	hspi2.Init.NSS = SPI_NSS_SOFT;
	
	// Set speed limit
	hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;
	hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
	hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
	hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	
	// Physically apply configuration
	if (HAL_SPI_Init(&hspi2) != HAL_OK) {
		while(1);
	}
}

/**
 * @brief System tick timer heartbeat.
 */
extern "C" void SysTick_Handler(void) {
	HAL_IncTick();
}

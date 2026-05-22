/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Hardware-specific implementation for tactile skin.
 * * This file acts as the Hardware Driver Layer.
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
volatile uint8_t cdc_conversion_complete = 0;

/**
 * @brief Structure to pair an AD7142 register address with its configuration value.
 */
typedef struct {
	uint16_t reg_addr;
	uint16_t reg_val;
} ad7142_reg_config_t;

// Hardware initialisation
void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_SPI1_Init(void);
void MX_SPI2_Init(void);
void AD7142_Init(void);
void Clear_CDC_Interrupts(void);

/**
 * @brief Bridge function to set physical GPIO states.
 * @param gpio_pin Number of the pin being used.
 * @param state Value being applied to the pin.
 */
extern "C" void set_gpio_state(uint8_t gpio_pin, uint8_t state) {
	GPIO_PinState s = (state) ? GPIO_PIN_SET : GPIO_PIN_RESET;
	
	if (gpio_pin == 0) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, s);
	if (gpio_pin == 1) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, s);
	if (gpio_pin == 2) HAL_GPIO_WritePin(GPIOB, GPIO_PIN_2, s);
}

/**
 * @brief Helper function to write a 16-bit value to a specific AD7142 register.
 * @param hspi Pointer to the SPI handler structure.
 * @param cs_port Pointer to the GPIO port instance for Chip Select.
 * @param cs_pin GPIO Pin number for Chip Select.
 * @param reg_addr Target register address on the AD7142.
 * @param data_val 16-bit data value to write to the register.
 */
void AD7142_Write_Reg(SPI_HandleTypeDef *hspi, GPIO_TypeDef *cs_port, uint16_t cs_pin, uint16_t reg_addr, uint16_t data_val) {
	uint16_t tx_command = 0xE000 | (reg_addr & 0x03FF);
	
	HAL_GPIO_WritePin(cs_port, cs_pin, GPIO_PIN_RESET);
	HAL_SPI_Transmit(hspi, &tx_command, 1, 10);
	HAL_SPI_Transmit(hspi, &data_val, 1, 10);
	HAL_GPIO_WritePin(cs_port, cs_pin, GPIO_PIN_SET);
}

/**
 * @brief Helper function to read a 16-bit value from a specific AD7142 register.
 * @param hspi Pointer to the SPI handler structure.
 * @param cs_port Pointer to the GPIO port instance for Chip Select.
 * @param cs_pin GPIO Pin number for Chip Select.
 * @param reg_addr Target register address on the AD7142.
 * @retval Value read from the AD7142 register.
 */
uint16_t AD7142_Read_Reg(SPI_HandleTypeDef *hspi, GPIO_TypeDef *cs_port, uint16_t cs_pin, uint16_t reg_addr) {
	uint16_t tx_command = 0xE400 | (reg_addr & 0x03FF);
	uint16_t rx_val = 0;
	
	HAL_GPIO_WritePin(cs_port, cs_pin, GPIO_PIN_RESET);
	HAL_SPI_Transmit(hspi, &tx_command, 1, 10);
	HAL_SPI_Receive(hspi, &rx_val, 1, 10);
	HAL_GPIO_WritePin(cs_port, cs_pin, GPIO_PIN_SET);
	
	return rx_val;
}

/**
 * @brief Bridge function to read two sensors at once.
 * @param val_a Pointer to location of first value to be read.
 * @param val_b Pointer to location of second value to be read.
 */
extern "C" void get_sensor_pair(uint16_t *val_a, uint16_t *val_b) {
	uint16_t reg_addr = 0x00B + current_column;
	uint16_t tx_command = 0xE400 | (reg_addr & 0x03FF);
	
	// Read sensor A
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
	HAL_SPI_Transmit(&hspi1, &tx_command, 1, 10);
	HAL_SPI_Receive(&hspi1, val_a, 1, 10);
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
	
	// Read sensor B
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_15, GPIO_PIN_RESET);
	HAL_SPI_Transmit(&hspi2, &tx_command, 1, 10);
	HAL_SPI_Receive(&hspi2, val_b, 1, 10);
	HAL_GPIO_WritePin(GPIOA, GPIO_PIN_15, GPIO_PIN_SET);
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
extern "C" void delay_us(uint16_t us) {
	uint32_t start_tick = DWT->CYCCNT;
	
	uint32_t target_ticks = (uint32_t)us * (SystemCoreClock / 1000000);
	while ((DWT->CYCCNT - start_tick) < target_ticks);
}

/**
 * @brief Scan the tactile matrix multiple times and saves the averaged data.
 * @param buffer Pointer to the calibration destination array.
 * @param num_averages Number of frame to average (use power of 2).
 */
void capture_calibration_frame(uint16_t *buffer, uint8_t num_averages) {
	// Clear buffer
	for (int i = 0; i < 128; ++i) {
		buffer[i] = 0;
	}
	
	// Allocate temporary buffers
	uint32_t *accumulator = (uint32_t*)calloc(128, sizeof(uint32_t));
	uint16_t *single_frame = (uint16_t*)malloc(128 * sizeof(uint16_t));
	
	// Scan matrix
	for (uint8_t i = 0; i < num_averages; ++i) {
		matrix_scan_parallel(single_frame);
		for (int j = 0; j < 128; ++j) {
			accumulator[j] += single_frame[j];
		}
		HAL_Delay(5);
	}
	
	// Compute average
	for (int i = 0; i < 128; ++i) {
		buffer[i] = (uint16_t)(accumulator[i] / num_averages);
	}
	
	free(accumulator);
	free(single_frame);
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
	
	AD7142_Init();
	
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
	
	HAL_Delay(3000);
	capture_calibration_frame(weight_low, 16);
	
	HAL_Delay(5000);
	capture_calibration_frame(weight_mid, 16);
	
	HAL_Delay(5000);
	capture_calibration_frame(weight_high, 16);
	
	float y_targets[3] = { 0.0f, 100.0f, 500.0f };
	uint16_t *x_samples[3] = { weight_low, weight_mid, weight_high };
	tactile_fit_curve(x_samples, y_targets, 128);
	
	free(weight_low);
	free(weight_mid);
	free(weight_high);
	
	// Frame buffers
	uint16_t sensor_data[128] = {0};
	uint16_t processed_data[128] = {0};
	
	// Clear initial startup triggers
	Clear_CDC_Interrupts();
	cdc_conversion_complete = 0;
	
	while (1) {
		while (cdc_conversion_complete == 0) {
			// Idle until callback
		}
		
		// Ensure both CDCs are fully done converting
		uint8_t cdc_a_done = 0;
		uint8_t cdc_b_done = 0;
		
		while (!cdc_a_done || !cdc_b_done) {
			if (!cdc_a_done) {
				uint16_t status_a = AD7142_Read_Reg(&hspi1, GPIOA, GPIO_PIN_4, 0x00A);
				if (status_a & 0x0080) {
					cdc_a_done = 1;
				}
			}
			if (!cdc_b_done) {
				uint16_t status_b = AD7142_Read_Reg(&hspi2, GPIOA, GPIO_PIN_15, 0x00A);
				if (status_b & 0x0080) {
					cdc_b_done = 1;
				}
			}
		}
		
		cdc_conversion_complete = 0;
		
		matrix_scan_parallel(sensor_data);
		tactile_process_frame(sensor_data, processed_data, 128);
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
	
	// Multiplication math (96MHz clock)
	RCC_OscInitStruct.PLL.PLLM = 25;
	RCC_OscInitStruct.PLL.PLLN = 192;
	RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
	RCC_OscInitStruct.PLL.PLLQ = 4;
	
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
	
	// Configure SPI1 hardware
	GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
	GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
	
	// Physically apply configuration
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
	
	// Configure SPI2 hardware
	GPIO_InitStruct.Pin = GPIO_PIN_13 | GPIO_PIN_14 | GPIO_PIN_15;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
	GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
	
	// Physically apply configuration
	HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
	
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
 * @brief Initialise the AD7142.
 */
void AD7142_Init(void) {
	// Bank 1
	ad7142_reg_config_t bank1_table[] = {
		{0x000, 0x0170}, // PWR_CONTROL
		{0x001, 0x00FF}, // STAGE_CAL_EN
		{0x002, 0x0FF0}, // AMB_COMP_CTRL0 (default)
		{0x003, 0x0140}, // AMB_COMP_CTRL1 (default)
		{0x004, 0xFFFF}, // AMB_COMP_CTRL2 (default)
		{0x005, 0x0000}, // STAGE_LOW_INT_EN (default)
		{0x006, 0x0000}, // STAGE_HIGH_INT_EN (default)
		{0x007, 0x0080}  // STAGE_COMPLETE_INT_EN
	};
	
	const int num_registers = sizeof(bank1_table) / sizeof(bank1_table[0]);
	
	for (int i = 0; i < num_registers; ++i) {
		AD7142_Write_Reg(&hspi1, GPIOA, GPIO_PIN_4, bank1_table[i].reg_addr, bank1_table[i].reg_val);
		AD7142_Write_Reg(&hspi2, GPIOA, GPIO_PIN_15, bank1_table[i].reg_addr, bank1_table[i].reg_val);
	}
	
	// Bank 2
	uint16_t used_reg_vals[] = {
		0x0001, // STAGEx_CONNECTION[6:0]
		0x4000, // STAGEx_CONNECTION[13:7]
		0x0000, // STAGEx_AFE_OFFSET
		0x2424, // STAGEx_SENSITIVITY
		0x0640, // STAGEx_OFFSET_LOW
		0x0640, // STAGEx_OFFSET_HIGH
		0x07D0, // STAGEx_OFFSET_HIGH_CLAMP
		0x07D0  // STAGEx_OFFSET_LOW_CLAMP
	};
	
	uint16_t unused_reg_vals[] = {
		0x3FFF, // STAGEx_CONNECTION[6:0]
		0xFFFF, // STAGEx_CONNECTION[13:7]
		0x0000, // STAGEx_AFE_OFFSET
		0x0000, // STAGEx_SENSITIVITY
		0x0000, // STAGEx_OFFSET_LOW
		0x0000, // STAGEx_OFFSET_HIGH
		0x0000, // STAGEx_OFFSET_HIGH_CLAMP
		0x0000  // STAGEx_OFFSET_LOW_CLAMP
	};
	
	int stage = 0;
	for (uint16_t addr = 0x080; addr < 0x0E0; ) {
		for (uint16_t i = 0; i < 8; ++i) {
			if (stage < 8) {
				uint16_t val;
				
				switch (i) {
					case 0:
						val = 0x3FFF ^ (used_reg_vals[i] << (2 * stage));
						val = 0x3FFF & val;
						break;
						
					case 1:
						val = 0x3FFF ^ (used_reg_vals[i] >> (2 * stage));
						val = 0x3FFE | val;
						val = 0x3FFF & val;
						break;
						
					case 2:
					case 3:						
					case 4:
					case 5:
					case 6:
					case 7:
						val = used_reg_vals[i];
						break;
				}

				AD7142_Write_Reg(&hspi1, GPIOA, GPIO_PIN_4, addr, val);
				AD7142_Write_Reg(&hspi2, GPIOA, GPIO_PIN_15, addr, val);
			} else {
				AD7142_Write_Reg(&hspi1, GPIOA, GPIO_PIN_4, addr, unused_reg_vals[i]);
				AD7142_Write_Reg(&hspi2, GPIOA, GPIO_PIN_15, addr, unused_reg_vals[i]);
			}
			addr++;
		}
		stage++;
	}
}

/**
 * @brief Reads the completion status register on both CDCs to clear.
 */
void Clear_CDC_Interrupts(void) {
	uint16_t status_a = AD7142_Read_Reg(&hspi1, GPIOA, GPIO_PIN_4, 0x00A);
	uint16_t status_b = AD7142_Read_Reg(&hspi2, GPIOA, GPIO_PIN_15, 0x00A);
	
	(void)status_a;
	(void)status_b;
}

/**
 * @brief System tick timer heartbeat.
 */
extern "C" void SysTick_Handler(void) {
	HAL_IncTick();
}

/**
 * @brief Hardware Interrupt Vector for EXTI Line 3.
 */
extern "C" void EXTI3_IRQHandler(void) {
	HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_3);
}

/**
 * @brief STM32 External Interrupt Callback hook.
 * @param Interrupt GPIO Pin.
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
	if (GPIO_Pin == GPIO_PIN_3) {
		cdc_conversion_complete = 1;
	}
}

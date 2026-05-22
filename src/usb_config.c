/**
 * @file usb_config.c
 * @author Aidan Mohammed-Ali
 * @brief Configuration interface for USB CDC Serial.
 * @date 2026-05-21
 */

#include <stm32f4xx_hal.h>
#include <stddef.h>
#include "usbd_core.h"
#include "usbd_cdc.h"

PCD_HandleTypeDef hpcd_USB_OTG_FS;
USBD_HandleTypeDef hUsbDeviceFS;

// Private transmission buffer for CDC data
static uint8_t UserTxBufferFS[2048];
// Private reception buffer for CDC data
static uint8_t UserRxBufferFS[2048];

// Standard device descriptor
static const uint8_t DeviceDescriptorFS[18] = {
	0x12, 			// bLength
	0x01,			// bDescriptorType
	0x00, 0x02, 	// bcdUSB
	0x02, 			// bDeviceClass
	0x00,			// bDeviceSubClass
	0x00,			// bDeviceProtocol
	0x40,			// bMaxPacketSize0
	0x83, 0x04, 	// idVendor
	0x40, 0x57, 	// idProduct
	0x00, 0x01, 	// bcdDevice
	0x01, 			// iManufacturer
	0x02, 			// iProduct
	0x03, 			// iSerialNumber
	0x01			// bNumConfigurations
};

// String descriptor 0: Language ID
static const uint8_t LangIDDescriptor[4] = {
	0x04,			// bLength
	0x03,			// bDescriptorType
	0x09, 0x08		// wLANGID[0]
};

// String descriptor 1: Manufacturer String
static const uint8_t ManufacturerStrDescriptor[28] = {
	0x1C,			// bLength
	0x03,			// bDescriptorType
	
	// "3YGP26-Spiers" encoded in UTF-16LE
	'3', 0x00, 'Y', 0x00, 'G', 0x00, 'P', 0x00, '2', 0x00, '6', 0x00,
	'-', 0x00,
	'S', 0x00, 'p', 0x00, 'i', 0x00, 'e', 0x00, 'r', 0x00, 's', 0x00
};

// String descriptor 2: Product String
static const uint8_t ProductStrDescriptor[26] = {
	0x1A,			// bLength
	0x03,			// bDescriptorType
	
	// "Tactile-Skin" encoded in UTF-16LE
	'T', 0x00, 'a', 0x00, 'c', 0x00, 't', 0x00, 'i', 0x00, 'l', 0x00, 'e', 0x00,
	'-', 0x00,
	'S', 0x00, 'k', 0x00, 'i', 0x00, 'n', 0x00
};

// String descriptor 3: Serial Number String
static const uint8_t SerialStrDescriptor[26] = {
	0x1A,			// bLength
	0x03,			// bDescriptorType
	
	// "000000000001" encoded in UTF16-LE
	'0', 0x00, '0', 0x00, '0', 0x00, '0', 0x00, '0', 0x00, '0', 0x00,
	'0', 0x00, '0', 0x00, '0', 0x00, '0', 0x00, '0', 0x00, '1', 0x00
};

/**
 * @brief Returns the physical device identity array to the core library.
 * @param speed Current bus speed (unused but required by ST library).
 * @param length Pointer to tell the library how big array is.
 * @retval A raw memory pointer to the start of the identity array.
 */
static uint8_t* GetDeviceDesc(USBD_SpeedTypeDef speed, uint16_t *length) {
	(void)speed;
	*length = sizeof(DeviceDescriptorFS);
	return (uint8_t*)DeviceDescriptorFS;
}

/**
 * @brief Returns the language ID descriptor array to the core library.
 * @param speed Current bus speed (unused but required by ST library).
 * @param length Pointer to tell the library how big array is.
 * @retval A raw memory pointer to the start of the language ID array.
 */
static uint8_t* GetLangIDDesc(USBD_SpeedTypeDef speed, uint16_t *length) {
	(void)speed;
	*length = sizeof(LangIDDescriptor);
	return (uint8_t*)LangIDDescriptor;
}

/**
 * @brief Returns the manufacturer text string descriptor array to the core library.
 * @param speed Current bus speed (unused but required by ST library).
 * @param length Pointer to tell the library how big array is.
 * @retval A raw memory pointer to the start of the manufacturer string array.
 */
static uint8_t* GetManufacturerStrDesc(USBD_SpeedTypeDef speed, uint16_t *length) {
	(void)speed;
	*length = sizeof(ManufacturerStrDescriptor);
	return (uint8_t*)ManufacturerStrDescriptor;
}

/**
 * @brief Returns the product text string descriptor array to the core library.
 * @param speed Current bus speed (unused but required by ST library).
 * @param length Pointer to tell the library how big array is.
 * @retval A raw memory pointer to the start of the product string array.
 */
static uint8_t* GetProductStrDesc(USBD_SpeedTypeDef speed, uint16_t *length) {
	(void)speed;
	*length = sizeof(ProductStrDescriptor);
	return (uint8_t*)ProductStrDescriptor;
}

/**
 * @brief Returns the hardware serial number text string descriptor array to the core library.
 * @param speed Current bus speed (unused but required by ST library).
 * @param length Pointer to tell the library how big array is.
 * @retval A raw memory pointer to the start of the serial number string array.
 */
static uint8_t* GetSerialStrDesc(USBD_SpeedTypeDef speed, uint16_t *length) {
	(void)speed;
	*length = sizeof(SerialStrDescriptor);
	return (uint8_t*)SerialStrDescriptor;
}

// Routing table
USBD_DescriptorsTypeDef FS_Desc = {
	GetDeviceDesc,
	GetLangIDDesc,
	GetManufacturerStrDesc,
	GetProductStrDesc,
	GetSerialStrDesc,
	NULL,
	NULL
};

/**
 * @brief Initialises the low-level hardware resources.
 * @param hpcd Pointer to the USB Peripheral Control Driver handle.
 */
void HAL_PCD_MspInit(PCD_HandleTypeDef *hpcd) {
	GPIO_InitTypeDef GPIO_InitStruct = {0};
	
	if (hpcd->Instance == USB_OTG_FS) {
		// Enable the power clocks to GPIO Port A
		__HAL_RCC_GPIOA_CLK_ENABLE();
		
		// Configure physical pins
		GPIO_InitStruct.Pin = GPIO_PIN_11 | GPIO_PIN_12;
		GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
		GPIO_InitStruct.Pull = GPIO_NOPULL;
		GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
		GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
		HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
		
		// Enable the main core clock to the USB OTG peripheral hardware block
		__HAL_RCC_USB_OTG_FS_CLK_ENABLE();
		
		// Enable the global USB interrupt channel
		HAL_NVIC_SetPriority(OTG_FS_IRQn, 0, 0);
		HAL_NVIC_EnableIRQ(OTG_FS_IRQn);
	}
}

/**
 * @brief Initialise the CDC media low layer.
 * @retval USBD_OK if all operations are OK else USBD_FAIL.
 */
static int8_t CDC_Init_FS(void) {
	USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
	USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS, sizeof(UserRxBufferFS));
	return (USBD_OK);
}

/**
 * @brief De-initialise the CDC media low layer.
 * @retval USBD_OK if all operations are OK else USBD_FAIL.
 */
static int8_t CDC_DeInit_FS(void) {
	return (USBD_OK);
}

/**
 * @brief Handles class-specific control requests.
 * @param cmd Command code identifier.
 * @param pbuf Buffer containing command data arguments.
 * @param length Number of data bytes in the command packet.
 * @retval USBD_OK if the control packet was processed successfully.
 */
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length) {
	(void)pbuf;
	(void)length;
	
	switch (cmd) {
		case CDC_SEND_ENCAPSULATED_COMMAND:
			break;
		case CDC_GET_ENCAPSULATED_RESPONSE:
			break;
		case CDC_SET_COMM_FEATURE:
			break;
		case CDC_GET_COMM_FEATURE:
			break;
		case CDC_CLEAR_COMM_FEATURE:
			break;
		case CDC_SET_LINE_CODING:
			break;
		case CDC_GET_LINE_CODING:
			break;
		case CDC_SET_CONTROL_LINE_STATE:
			break;
		case CDC_SEND_BREAK:
			break;
		default:
			break;
	}
	return (USBD_OK);
}

/**
 * @brief Callback triggered when data is received from the host laptop.
 * @param pbuf Pointer to the buffer containing incoming bytes.
 * @param len Pointer to the variable tracking number of received bytes.
 * @retval USBD_OK if processing succeeded.
 */
static int8_t CDC_Receive_FS(uint8_t *pbuf, uint32_t *len) {
	USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &pbuf[0], sizeof(UserRxBufferFS));
	USBD_CDC_ReceivePacket(&hUsbDeviceFS);
	return (USBD_OK);
}

// Master CDC interface routing table
USBD_CDC_ItfTypeDef USBD_Interface_fops_FS = {
	CDC_Init_FS,
	CDC_DeInit_FS,
	CDC_Control_FS,
	CDC_Receive_FS
};

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
#include "usbd_conf.h"

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
	USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
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
	USBD_CDC_SetRxBuffer(&hUsbDeviceFS, pbuf);
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

/**
 * @defgroup USB_LL_Interface Low-Level Link Layer Interface
 * @brief Hardware routing functions binding the ST USB Stack to STM32 HAL peripheral drivers.
 * @{
 */

/**
 * @brief Initialise the low-level hardware peripheral controller driver (PCD).
 * @param pdev Device handle instance mapping the core USB stack.
 * @retval USBD_OK if peripheral is up, USBD_FAIL otherwise.
 */
USBD_StatusTypeDef USBD_LL_Init(USBD_HandleTypeDef *pdev) {
	hpcd_USB_OTG_FS.pData = pdev;
	pdev->pData = &hpcd_USB_OTG_FS;
	
	hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
	hpcd_USB_OTG_FS.Init.dev_endpoints = 4;
	hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
	hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
	hpcd_USB_OTG_FS.Init.Sof_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.vbus_sensing_enable = DISABLE;
	hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
	
	if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK) {
		return USBD_FAIL;
	}
	
	HAL_PCDEx_SetRxFiFo(&hpcd_USB_OTG_FS, 128);
	HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 0, 64);
	HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 1, 128);
	HAL_PCDEx_SetTxFiFo(&hpcd_USB_OTG_FS, 2, 64);
	
	return USBD_OK;
}

/**
 * @brief De-initialise the low-level hardware peripheral controller driver.
 * @param pdev Device handle instance mapping the core USB stack.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_DeInit(USBD_HandleTypeDef *pdev) {
	HAL_PCD_DeInit((PCD_HandleTypeDef*)pdev->pData);
	return USBD_OK;
}

/**
 * @brief Start the low-level USB hardware driver.
 * @param pdev Device handle instance mapping the core USB stack.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_Start(USBD_HandleTypeDef *pdev) {
	HAL_PCD_Start((PCD_HandleTypeDef*)pdev->pData);
	return USBD_OK;
}

/**
 * @brief Stop the low-level USB hardware driver.
 * @param pdev Device handle instance mapping the core USB stack.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_Stop(USBD_HandleTypeDef *pdev) {
	HAL_PCD_Stop((PCD_HandleTypeDef*)pdev->pData);
	return USBD_OK;
}

/**
 * @brief Open a designated USB endpoint device channel for structural transactions
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @param ep_type Selected transaction profile token (Control, Bulk, Interrupt).
 * @param ep_mps Maximum Packet Size bounds in total raw bytes.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_OpenEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t ep_type, uint16_t ep_mps) {
	HAL_PCD_EP_Open((PCD_HandleTypeDef*)pdev->pData, ep_addr, ep_mps, ep_type);
	return USBD_OK;
}

/**
 * @brief Closes a designated USB endpoint device channel.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_CloseEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	HAL_PCD_EP_Close((PCD_HandleTypeDef*)pdev->pData, ep_addr);
	return USBD_OK;
}

/**
 * @brief Flushes pending transaction data buffer inside a designated hardware endpoint FIFO.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_FlushEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	HAL_PCD_EP_Flush((PCD_HandleTypeDef*)pdev->pData, ep_addr);
	return USBD_OK;
}

/**
 * @brief Activates a hardware stall condition on a target endpoint channel.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_StallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	HAL_PCD_EP_SetStall((PCD_HandleTypeDef*)pdev->pData, ep_addr);
	return USBD_OK;
}

/**
 * @brief Clears an activate hardware stall condition on a target endpoint channel.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @retval Always return USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_ClearStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	HAL_PCD_EP_ClrStall((PCD_HandleTypeDef*)pdev->pData, ep_addr);
	return USBD_OK;
}

/**
 * @brief Evaluates if a specific hardware endpoint is currently stalled.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Hardware physical endpoint address index.
 * @retval Returns 1 if stalled, 0 if operational.
 */
uint8_t USBD_LL_IsStallEP(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	PCD_HandleTypeDef *hpcd = (PCD_HandleTypeDef*)pdev->pData;
	if ((ep_addr & 0x80) == 0x80) {
		return hpcd->IN_ep[ep_addr & 0x7F].is_stall;
	} else {
		return hpcd->OUT_ep[ep_addr & 0x7F].is_stall;
	}
}

/**
 * @brief Programs the assigned device network address into the internal USB hardware peripheral.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param dev_addr Network enumeration address assigned by host controller.
 * @retval Always returns USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_SetUSBAddress(USBD_HandleTypeDef *pdev, uint8_t dev_addr) {
	HAL_PCD_SetAddress((PCD_HandleTypeDef*)pdev->pData, dev_addr);
	return USBD_OK;
}


///////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Initiates an asynchronous data packet transmission sequence on an IN hardware endpoint channel.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Target physical IN endpoint address index.
 * @param pbuf Pointer to the source memory buffer holding the raw byte array payload.
 * @param size Total length of payload packet boundary constraints in bytes.
 * @retval Always returns USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_Transmit(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t *pbuf, uint32_t size) {
	HAL_PCD_EP_Transmit((PCD_HandleTypeDef*)pdev->pData, ep_addr, pbuf, size);
	return USBD_OK;
}

/**
 * @brief Configures and prepares an OUT hardware endpoint channel block buffer to receive incoming packets.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Target physical OUT endpoint address index.
 * @param pbuf Pointer to target destination buffer allocated in system memory RAM.
 * @param size Maximum incoming transfer packet allocation bounds in bytes.
 * @retval Always returns USBD_OK.
 */
USBD_StatusTypeDef USBD_LL_PrepareReceive(USBD_HandleTypeDef *pdev, uint8_t ep_addr, uint8_t *pbuf, uint32_t size) {
	HAL_PCD_EP_Receive((PCD_HandleTypeDef*)pdev->pData, ep_addr, pbuf, size);
	return USBD_OK;
}

/**
 * @brief Returns the total count of received bytes pulled off the wire into an OUT endpoint buffer channel.
 * @param pdev Device handle instance mapping the core USB stack.
 * @param ep_addr Target physical OUT endpoint address index.
 * @retval Total packet load count read from peripheral driver status registers.
 */
uint32_t USBD_LL_GetRxDataSize(USBD_HandleTypeDef *pdev, uint8_t ep_addr) {
	return HAL_PCD_EP_GetRxCount((PCD_HandleTypeDef*)pdev->pData, ep_addr);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when a SETUP packet is captured on Endpoint 0.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_SetupStageCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_SetupStage((USBD_HandleTypeDef*)hpcd->pData, (uint8_t*)hpcd->Setup);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when an OUT channel transaction completes.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 * @param Active target physical endpoint identifier channel index.
 */
void HAL_PCD_DataOutStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) {
	USBD_LL_DataOutStage((USBD_HandleTypeDef*)hpcd->pData, epnum, hpcd->OUT_ep[epnum].xfer_buff);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when an IN channel transaction finishes clearing FIFO buffers.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 * @param Active target physical endpoint identifier channel index.
 */
void HAL_PCD_DataInStageCallback(PCD_HandleTypeDef *hpcd, uint8_t epnum) {
	USBD_LL_DataInStage((USBD_HandleTypeDef*)hpcd->pData, epnum, hpcd->IN_ep[epnum].xfer_buff);
}

/**
 * @brief Event hook triggered by peripheral controller hardware upon capturing a Start of Frame (SOF) packet sequence.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_SOFCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_SOF((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when a bus reset state condition is encountered.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_ResetCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_SetSpeed((USBD_HandleTypeDef*)hpcd->pData, USBD_SPEED_FULL);
	USBD_LL_Reset((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when a USB suspend bus state becomes active.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_SuspendCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_Suspend((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when a host resume signal is asserted on the bus lines.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_ResumeCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_Resume((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when connection to the host is physically negotiated.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_ConnectCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_DevConnected((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @brief Event hook triggered by peripheral controller hardware when the physical USB cable connection is severed.
 * @param hpcd Hardware peripheral driver context structure tracking low-level controller registers.
 */
void HAL_PCD_DisconnectCallback(PCD_HandleTypeDef *hpcd) {
	USBD_LL_DevDisconnected((USBD_HandleTypeDef*)hpcd->pData);
}

/**
 * @}
 */

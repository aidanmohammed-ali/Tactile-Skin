/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Visualisation tool.
 * * This file takes the sensor outputs and displays.
 * @date 2026-05-23
 */

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include "raylib.h"

// Cross-Platform OS Serial Communication
#if defined(_WIN32)
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN // Exclude unneeded legacy Windows headers
	#endif
	#define NOGDI
	#define NOUSER
	#include <windows.h>
#else
	#include <unistd.h>
	#include <fcntl.h>
	#include <termios.h>
#endif

// Geometry configuration 
const int ROWS = 8;
const int COLS = 16;
const int CELL_SIZE = 75;
const int WINDOW_WIDTH = COLS * CELL_SIZE;
const int WINDOW_HEIGHT = ROWS * CELL_SIZE;
const int BAR_HEIGHT = 60;
const int RIGHT_MARGIN = 25;
const int TEXT_Y = 19;
const int STATUS_TEXT_SIZE = 18;

// Calibration step tracker
enum CalibrationWizardStep {
	STEP_READY = 0,
	STEP_WAITING_LOW,
	STEP_WAITING_MID,
	STEP_WAITING_HIGH,
	STEP_COMPLETE
};

CalibrationWizardStep wizard_step = STEP_READY;

// Firmware data structure
#pragma pack(push, 1)
typedef struct {
	uint16_t channels[128];
} TactileFrame;
#pragma pack(pop)

// Cross-Platform Serial Handlers
#if defined(_WIN32)
	typedef HANDLE SerialHandle;
	const SerialHandle INVALID_SERIAL = INVALID_HANDLE_VALUE;
#else
	typedef int SerialHandle;
	const SerialHandle INVALID_SERIAL = -1;
#endif

/**
 * @brief Establishes a raw cross-platform connection to the specified hardware serial interface.
 * @param portName The system file descriptor path
 * @retval A functional native system hook, or INVALID_SERIAL if initialisation fails.
 */
SerialHandle OpenSerialPort(const char *portName) {
#if defined(_WIN32)
	SerialHandle handle = CreateFileA(portName, GENERIC_READ | GENERIC_WRITE,
										0, nullptr, OPEN_EXISTING, 0, nullptr);
	if (handle == INVALID_HANDLE_VALUE) {
		return INVALID_SERIAL;
	}
	
	DCB dcbSerialParams = {0};
	dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
	if (!GetCommState(handle, &dcbSerialParams)) {
		CloseHandle(handle);
		return INVALID_SERIAL;
	}
	
	dcbSerialParams.BaudRate = CBR_115200;
	dcbSerialParams.ByteSize = 8;
	dcbSerialParams.StopBits = ONESTOPBIT;
	dcbSerialParams.Parity = NOPARITY;
	
	if (!SetCommState(handle, &dcbSerialParams)) {
		CloseHandle(handle);
		return INVALID_SERIAL;
	}
	
	COMMTIMEOUTS timeouts = {0};
	timeouts.ReadIntervalsTimeout = MAXWORD;
	timeouts.ReadTotalTimeoutConstant = 0;
	timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    
    if (!SetCommTimeouts(handle, &timeouts)) {
        CloseHandle(handle);
        return INVALID_SERIAL;
    }
	
	return handle;
#else
	SerialHandle handle = open(portName, O_RDONLY | O_NOCTTY | O_NDELAY);
	if (handle == -1) {
		return INVALID_SERIAL;
	}
	
	fcntl(handle, F_SETFL, 0);
	
	struct termios tty;
	if (tcgetattr(handle, &tty) != 0) {
		close(handle);
		return INVALID_SERIAL;
	}
	
	cfsetispeed(&tty, B115200);
	tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
	tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
	tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
	tty.c_oflag &= ~OPOST;
	
	tty.c_cc[VMIN] = 0;
	tty.c_cc[VTIME] = 1;
	
	tty.c_cflag |= (CLOCAL | CREAD);
	
	if (tcsetattr(handle, TCSANOW, &tty) != 0) {
		close(handle);
		return INVALID_SERIAL;
	}
	return handle;
#endif
}

/**
 * @brief Read a complete, raw binary data packet from the active serial interface.
 * @param handle The active operating system serial connection handle.
 * @param frame A reference to the target stucture where incoming sensor values are stored.
 * @retval true If a complete 256-byte frame was successfully read without corruption
 * * false If the stream timed out, threw an error, or the buffer was empty.
 */
bool ReadSerialFrame(SerialHandle handle, TactileFrame &frame) {
	if (handle == INVALID_SERIAL) {
		return false;
	}
	
	// Map a raw byte pointer directly to data layout
	uint8_t *buffer = reinterpret_cast<uint8_t*>(&frame);
	uint32_t bytesToRead = sizeof(TactileFrame);
	
#if defined(_WIN32)
	DWORD errors;
	COMSTAT status;
	
	ClearCommError(handle, &errors, &status);
	
	if (status.cbInQue < bytesToRead) {
		return false;
	}
	
	DWORD bytesRead;
	if (ReadFile(handle, buffer, bytesToRead, &bytesRead, nullptr)) {
		return (bytesRead == bytesToRead);
	}
	return true;
#else
	uint32_t totalBytesRead = 0;
	while (totalBytesRead < bytesToRead) {
		ssize_t bytesRead = read(handle, buffer + totalBytesRead, bytesToRead - totalBytesRead);
		
		if (bytesRead > 0) {
			totalBytesRead += bytesRead;
		} else if (bytesRead == 0) {
			return false;
		} else {
			if (errno == EAGAIN || errno == EWOULDBLOCK) {
				return false;
			}
			return false;
		}
	}
	return true;
#endif
}

/**
 * @brief Sends a single byte command down the active serial interface.
 * @param handle The active operating system serial connection handle.
 * @param cmd The 8-bit command to transmit.
 * @retval true If the byte was successfully sent without corruption
 * * false Otherwise.
 */
bool WriteSerialByte(SerialHandle handle, uint8_t cmd) {
#if defined(_WIN32)
	DWORD bytesWritten;
	if (WriteFile(handle, &cmd, 1, &bytesWritten, nullptr)) {
		return (bytesWritten == 1);
	}
	return false;
#else
	ssize_t bytesWritten = write(handle, &cmd, 1);
	return (bytesWritten == 1);
#endif
}

int main() {
	// Start the application completely disconnected
	SerialHandle serial = INVALID_SERIAL;
	bool hardware_online = false;
	
	std::cout << "[INT] Visualiser starting. Please select a port from the menu." << std::endl;
	
	// Initialise Raygui
	const char *port_list = "SIMULATOR;/dev/ttyACM0;/dev/ttyACM1;/dev/ttyACM2;COM7;COM8;COM9";
	int dropdown_active_index = 0;
	int last_selected_index = 0;
	
	// Initialise Raylib Accelerated Window Engine
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT + (2 * BAR_HEIGHT), "Tactile Skin Visualiser");
	SetTargetFPS(60);
	
	// Style extension
	GuiSetStyle(DEFAULT, TEXT_SIZE, 18);
	GuiSetStyle(DEFAULT, BACKGROUND_COLOR, ColorToInt({ 30, 30, 30, 255 }));
	GuiSetStyle(DEFAULT, LINE_COLOR, ColorToInt(GRAY));
	GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
	
	// Hover state across lists
	GuiSetStyle(DEFAULT, TEXT_COLOR_FOCUSED, ColorToInt(ORANGE));
	GuiSetStyle(DEFAULT, BORDER_COLOR_FOCUSED, ColorToInt(ORANGE));
	
	// Idle State
	GuiSetStyle(COMBOBOX, BASE_COLOR_NORMAL, ColorToInt({ 45, 45, 45, 255 }));
	GuiSetStyle(COMBOBOX, BORDER_COLOR_NORMAL, ColorToInt(GRAY));
	GuiSetStyle(COMBOBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
	
	// Hovered State
	GuiSetStyle(COMBOBOX, BASE_COLOR_FOCUSED, ColorToInt({ 65, 65, 65, 255 }));
	GuiSetStyle(COMBOBOX, BORDER_COLOR_FOCUSED, ColorToInt(ORANGE));
	GuiSetStyle(COMBOBOX, TEXT_COLOR_FOCUSED, ColorToInt(ORANGE));
	
	// Pressed/Expanded
	GuiSetStyle(DROPDOWNBOX, BASE_COLOR_NORMAL, ColorToInt({ 35, 35, 35, 255 }));
	GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_NORMAL, ColorToInt(WHITE));
	GuiSetStyle(DROPDOWNBOX, BASE_COLOR_FOCUSED, ColorToInt({ 60, 60, 60, 255 }));
	GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_FOCUSED, ColorToInt(ORANGE));
	GuiSetStyle(DROPDOWNBOX, BASE_COLOR_PRESSED, ColorToInt({ 80, 80, 80, 255 }));
	GuiSetStyle(DROPDOWNBOX, TEXT_COLOR_PRESSED, ColorToInt(LIME));
	
	// Button
	GuiSetStyle(COMBOBOX, COMBO_BUTTON_WIDTH, 46);
	GuiSetStyle(DEFAULT, BORDER_COLOR_NORMAL, ColorToInt(GRAY));
	GuiSetStyle(DEFAULT, BASE_COLOR_NORMAL, ColorToInt({ 55, 55, 55, 255 }));
		
	TactileFrame current_frame = {0};
	float simulation_time = 0.0f;
	
	// Run loop
	while (!WindowShouldClose()) {
		if (wizard_step != STEP_COMPLETE) {
			if (wizard_step == STEP_READY && IsKeyPressed(KEY_C)) {
				wizard_step = STEP_WAITING_LOW;
				std::cout << "[WIZARD] Calibration Initiated. STEP 1: Clear sensor matrix for low-range capture." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] when ready..." << std::endl;
			} else if (wizard_step == STEP_WAITING_LOW && IsKeyPressed(KEY_ENTER)) {
				uint8_t cmd = 0x10;
				if (hardware_online && serial != INVALID_SERIAL) {
					std::cout << "[LINK] Sent trigger byte 0x10 (Low Tare) to MCU." << std::endl;
					WriteSerialByte(serial, cmd);
				} else {
					std::cout << "[SIM] Simulating STEP 1: Low-range baseline locked." << std::endl;
				}
				wizard_step = STEP_WAITING_MID;
				std::cout << "[WIZARD] STEP 2: Load mid-weight reference onto matrix." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] when load is secure and stable..." << std::endl;
			} else if (wizard_step == STEP_WAITING_MID && IsKeyPressed(KEY_ENTER)) {
				uint8_t cmd = 0x11;
				if (hardware_online && serial != INVALID_SERIAL) {
					WriteSerialByte(serial, cmd);
					std::cout << "[LINK] Sent trigger byte 0x11 (Mid Weight) to MCU." << std::endl;
				} else {
					std::cout << "[SIM] Simulating STEP 2: Mid-weight curve calculation locked." << std::endl;
				}
				wizard_step = STEP_WAITING_HIGH;
				std::cout << "[WIZARD] STEP 3: Load high-weight reference onto matrix layers." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] when load is secure and stable..." << std::endl;
			} else if (wizard_step == STEP_WAITING_HIGH && IsKeyPressed(KEY_ENTER)) {
				uint8_t cmd = 0x12;
				if (hardware_online && serial != INVALID_SERIAL) {
					WriteSerialByte(serial, cmd);
					std::cout << "[LINK] Sent trigger byte 0x12 (High Weight) to MCU." << std::endl;
				} else {
					std::cout << "[SIM] Simulating STEP 3: High-weight curve parameters finalized." << std::endl;
				}
				wizard_step = STEP_COMPLETE;
				std::cout << "[SUCCESS] Hardware calibration sequence complete and active! (Press 'R' to clear profiles)" << std::endl;
			}
		} else {
			if (IsKeyPressed(KEY_R)) {
				uint8_t cmd = 0x1F;
				if (hardware_online) {
					WriteSerialByte(serial, cmd);
				}
				wizard_step = STEP_READY;
				std::cout << "[WIZARD] Calibration settings reset to default. System back to idle." << std::endl;
			}
		}
		
		if (hardware_online) {
			ReadSerialFrame(serial, current_frame);
		} else {
			// Simulation Mode
			simulation_time += GetFrameTime();
			
			float target_c = 7.5f + std::sin(simulation_time * 1.2f) * 5.0f;
			float target_r = 3.5f + std::cos(simulation_time * 0.8f) * 2.5f;
			
			for (int r = 0; r < ROWS; ++r) {
				for (int c = 0; c < COLS; ++c) {
					float dr = r - target_r;
					float dc = c - target_c;
					float distance_squared = (dr * dr) + (dc * dc);
					
					// Gaussian distribution
					float intensity_curve = std::exp(-distance_squared / 3.5f);
					
					current_frame.channels[r * COLS + c] = static_cast<uint16_t>(intensity_curve * 4000.0f);
				}
			}
		}
		
		// Rendering
		BeginDrawing();
		ClearBackground(BLACK);
		
		// Draw matrix
		for (int r = 0; r < ROWS; ++r) {
			for (int c = 0; c < COLS; ++c) {
				uint16_t raw_val = current_frame.channels[r * COLS + c];
				if (raw_val > 4095) {
					raw_val = 4095;
				}
				
				uint8_t intensity = (raw_val * 255) / 4095;
				
				uint8_t red_channel = intensity;
				uint8_t green_channel = 255 - intensity;
				
				Color cell_color = { red_channel, green_channel, 0, 255 };
				
				DrawRectangle(c * CELL_SIZE, (r * CELL_SIZE) + BAR_HEIGHT, CELL_SIZE - 2, CELL_SIZE - 2, cell_color);
			}
		}
		
		// Draw calibration instructions
		DrawRectangle(0, 0, WINDOW_WIDTH, BAR_HEIGHT, Fade(BLACK, 0.95f));
		DrawLine(0, BAR_HEIGHT - 1, WINDOW_WIDTH, BAR_HEIGHT - 1, GRAY);
		
		if (wizard_step == STEP_READY) {
			DrawText("SYSTEM OPERATIONAL", 25, 16, 22, ORANGE);
			
			const char *txt = "Press [C] to initiate Multi-Point Calibration Wizard";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (wizard_step == STEP_WAITING_LOW) {
			DrawText("CALIBRATION STEP 1/3: ZERO WEIGHT", 25, 16, 22, ORANGE);
			
			const char *txt = "Clear sensor completely. Press [ENTER] when cleared...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (wizard_step == STEP_WAITING_MID) {
			DrawText("CALIBRATION STEP 2/3: MID WEIGHT", 25, 16, 22, ORANGE);
			
			const char *txt = "Place mid-range reference load onto sensor. Press [ENTER] when stable...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (wizard_step == STEP_WAITING_HIGH) {
			DrawText("CALIBRATION STEP 3/3: HIGH WEIGHT", 25, 16, 22, ORANGE);
			
			const char *txt = "Place high-range reference load onto sensor. Press [ENTER] to compute curve...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (wizard_step == STEP_COMPLETE) {
			DrawText("CALIBRATION PROFILES ACTIVE", 25, 16, 22, LIME);
			
			const char *txt = "Matrix hardware curve computed. Press [R] to clear mapping and reset.";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		}
		
		// Draw link status bar
		if (hardware_online) {
			DrawRectangle(0, WINDOW_HEIGHT + BAR_HEIGHT, WINDOW_WIDTH, BAR_HEIGHT, ColorAlpha(BLACK, 0.9f));
			DrawRectangle(0, WINDOW_HEIGHT + BAR_HEIGHT, 12, BAR_HEIGHT, GREEN);
			DrawText("HARDWARE LIVE", 25, WINDOW_HEIGHT + BAR_HEIGHT + 16, 22, WHITE);
		} else {
			DrawRectangle(0, WINDOW_HEIGHT + BAR_HEIGHT, WINDOW_WIDTH, BAR_HEIGHT, ColorAlpha(BLACK, 0.9f));
			DrawRectangle(0, WINDOW_HEIGHT + BAR_HEIGHT, 12, BAR_HEIGHT, RED);
			DrawText("SIMULATION MODE", 25, WINDOW_HEIGHT + BAR_HEIGHT + 16, 22, WHITE);
		}
		
		// Draw dropdown menu
		GuiComboBox({ (float)(WINDOW_WIDTH - 425), (float)(WINDOW_HEIGHT + BAR_HEIGHT + 7), 400, 46 }, port_list, &dropdown_active_index);
		if (dropdown_active_index != last_selected_index) {
			last_selected_index = dropdown_active_index;

			if (serial != INVALID_SERIAL) {
#if defined(_WIN32)
				CloseHandle(serial);
#else
				close(serial);
#endif
				serial = INVALID_SERIAL;
				hardware_online = false;
			}

			const char *target_path = nullptr;
			switch (dropdown_active_index) {
				case 1:
					target_path = "/dev/ttyACM0";
					break;
				case 2:
					target_path = "/dev/ttyACM1";
					break;
				case 3:
					target_path = "/dev/ttyACM2";
					break;
				case 4:
					target_path = "COM7";
					break;
				case 5:
					target_path = "COM8";
					break;
				case 6:
					target_path = "COM9";
					break;
				default:
					target_path = nullptr;
					break;
			}

			if (target_path != nullptr) {
				std::cout << "[SERIAL LINK] Opening connection to " << target_path << "..." << std::endl;
				serial = OpenSerialPort(target_path);
				hardware_online = (serial != INVALID_SERIAL);
				if (!hardware_online) {
					std::cout << "[ERROR] Connection failed. Falling back to Simulation Mode." << std::endl;
				}
			} else {
				hardware_online = false;
				std::cout << "[SERIAL LINK] Forced Simulation Mode via menu." << std::endl;
			}
		}
		
		EndDrawing();
	}
	
	// De-allocation Phase
	std::cout << "[SHUTDOWN] Closing graphics pipeline modules..." << std::endl;
	CloseWindow();
	
	if (serial != INVALID_SERIAL) {
#if defined(_WIN32)
	CloseHandle(serial);
#else
	close(serial);
#endif
	std::cout << "[SHUTDOWN] Serial interface closed cleanly." << std::endl;
	}
	
	return 0;
}

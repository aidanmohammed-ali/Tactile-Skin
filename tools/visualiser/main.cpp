/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Visualisation tool.
 * * This file takes the sensor outputs and displays it.
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
#include "processor.hpp"

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
	#include <sys/ioctl.h>
#endif

// Geometry configuration 
const int ROWS = 8;
const int COLS = 16;
const int CELL_SIZE = 95;
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
	uint8_t magic_header[4];
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
 * @retval true If a complete 260-byte frame was successfully synchronised and read without corruption
 * * false If the stream timed out, threw an error, or the buffer was empty.
 */
bool ReadSerialFrame(SerialHandle handle, TactileFrame &frame) {
	if (handle == INVALID_SERIAL) {
		return false;
	}
	
	const uint32_t TOTAL_PACKET_SIZE = sizeof(TactileFrame);
	
#if defined(_WIN32)
	DWORD errors;
	COMSTAT status;	
	ClearCommError(handle, &errors, &status);
	
	if (status.cbInQue < TOTAL_PACKET_SIZE) {
		return false;
	}
#else
	int bytesAvailable = 0;
	
	if (ioctl(handle, FIONREAD, &bytesAvailable) < 0 ||
				bytesAvailable < static_cast<int>(TOTAL_PACKET_SIZE)) {
		return false;
	}
#endif

	uint8_t match_buffer[4] = { 0 };
	
	while (true) {
#if defined(_WIN32)
		DWORD bytesRead;
		if (!ReadFile(handle, &match_buffer[0], 1, &bytesRead, nullptr) || bytesRead != 1) {
			return false;
		}
#else
		ssize_t bytesRead = read(handle, &match_buffer[0], 1);
		if (bytesRead != 1) {
			return false;
		}
#endif

		if (match_buffer[0] == 0xDE) {
#if defined(_WIN32)
			if (!ReadFile(handle, &match_buffer[1], 3, &bytesRead, nullptr) || bytesRead != 3) {
				continue;
			}
#else
			bytesRead = read(handle, &match_buffer[1], 3);
			if (bytesRead != 3) {
				continue;
			}
#endif
			if (match_buffer[1] == 0xAD && match_buffer[2] == 0xBE && match_buffer[3] == 0xEF) {
				std::memcpy(frame.magic_header, match_buffer, 4);
				
				uint8_t *data_destination = reinterpret_cast<uint8_t*>(frame.channels);
				uint32_t bytes_remaining = 256;
				
				while (bytes_remaining > 0) {
#if defined(_WIN32)
					if (!ReadFile(handle, data_destination, bytes_remaining, &bytesRead, nullptr) ||
						bytesRead <= 0) {
							return false;
					}
#else
					bytesRead = read(handle, data_destination, bytes_remaining);
					if (bytesRead <= 0) {
						return false;
					}
#endif
					bytes_remaining -= bytesRead;
					data_destination += bytesRead;
				}
				return true;
			}
		}
	}
	return false;
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
		
	TactileFrame current_frame = { 0 };
	float simulation_time = 0.0f;

	Processor processor(128);
	float processed_frame[128] = { 0.0f };
	
	float popup_timer = 0.0f;
	char popup_message[256] = { 0 };
	
	bool config_edit_mode = false;
	char config_filepath[256] = "config.ini";
	
	// Run loop
	while (!WindowShouldClose()) {
		/** LEGACY CURVE CALIBRATION
		if (processor.m_state != Processor::STATE_READY) {
			if (processor.m_state == Processor::STATE_UNCALIBRATED && IsKeyPressed(KEY_C)) {
				processor.AdvanceWizard(current_frame.channels);
				std::cout << "[WIZARD] STEP 1/3: Baseline context requested. Clear matrix." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] to log low reference..." << std::endl;
			} else if (processor.m_state == Processor::STATE_CAPTURE_LOW && IsKeyPressed(KEY_ENTER)) {
				processor.AdvanceWizard(current_frame.channels);
				std::cout << "[WIZARD] STEP 2/3: Mid weight context requested. Apply intermediate force." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] to log mid reference..." << std::endl;
			} else if (processor.m_state == Processor::STATE_CAPTURE_MID && IsKeyPressed(KEY_ENTER)) {
				processor.AdvanceWizard(current_frame.channels);
				std::cout << "[WIZARD] STEP 3/3: Max weight context requested. Apply ceiling force." << std::endl;
				std::cout << "[WIZARD] Press [ENTER] to solve matrices..." << std::endl;
			} else if (processor.m_state == Processor::STATE_CAPTURE_HIGH && IsKeyPressed(KEY_ENTER)) {
				processor.AdvanceWizard(current_frame.channels);
				std::cout << "[SUCCESS] Mathematical profiles locked down into local storage vectors. (Press 'R' to reset)" << std::endl;
			}
		} else {
			if (IsKeyPressed(KEY_R)) {
				processor.ResetCalibration();
				std::cout << "[WIZARD] Context configurations cleared. System running on raw pass-through." << std::endl;
			}
		}
		**/
		
		if (!config_edit_mode) {
			if (IsKeyPressed(KEY_C)) {
				processor.Tare(current_frame.channels);
				std::cout << "[TARE] Baseline captured." << std::endl;
			}
			if (IsKeyPressed(KEY_R)) {
				processor.ResetCalibration();
				std::cout << "[RESET] Returning to raw passthrough." << std::endl;
			}
		}
		
		// Drag and drop config logic
		if (IsFileDropped()) {
			FilePathList droppedFiles = LoadDroppedFiles();
			
			if (droppedFiles.count > 0) {
				ProcessingConfig newConfig;
				
				char *clean_path = droppedFiles.paths[0];
				
				if (strncmp(clean_path, "file://", 7) == 0) {
					clean_path += 7;
				}
				
				clean_path[strcspn(clean_path, "\r\n")] = 0;
				
				if (newConfig.loadFromIni(clean_path)) {
					processor.UpdateConfig(newConfig);
					std::cout << "[CONFIG] Success: Engine updated with " << clean_path << std::endl;
					
					popup_timer = 3.0f;
					snprintf(popup_message, sizeof(popup_message), "CONFIG LOADED: %s", GetFileName(clean_path));
				} else {
					std::cout << "[CONFIG ERROR] Failed to parse " << clean_path << std::endl;
					
					popup_timer = 3.0f;
					snprintf(popup_message, sizeof(popup_message), "ERROR: INVALID CONFIG FILE");
				}
			}
			UnloadDroppedFiles(droppedFiles);
		}
		
		if (hardware_online) {
			ReadSerialFrame(serial, current_frame);
		} else {
			simulation_time += GetFrameTime();
			
			float target_c = 7.5f + std::sin(simulation_time * 1.2f) * 5.0f;
			float target_r = 3.5f + std::cos(simulation_time * 0.8f) * 2.5f;
			
			for (int r = 0; r < ROWS; ++r) {
				for (int c = 0; c < COLS; ++c) {
					float dr = r - target_r;
					float dc = c - target_c;
					float distance_squared = (dr * dr) + (dc * dc);
					
					float intensity_curve = std::exp(-distance_squared / 3.5f);
					current_frame.channels[r * COLS + c] = static_cast<uint16_t>(intensity_curve * 65535.0f);
				}
			}
		}
		
		processor.ProcessFrame(current_frame.channels, processed_frame);
		
		// Rendering
		BeginDrawing();
		ClearBackground(BLACK);
		
		// Draw matrix
		for (int r = 0; r < ROWS; ++r) {
			for (int c = 0; c < COLS; ++c) {
				int index = r * COLS + c;

				if (index < 0 || index >= 128) {
					continue;
				}

				float cell_value = processed_frame[index];
				
				uint8_t intensity = static_cast<uint8_t>(cell_value * 255.0f);
				
				uint8_t blue_channel = intensity;
				
				Color cell_color = { 0, 0, blue_channel, 255 };
				
				DrawRectangle(c * CELL_SIZE, (r * CELL_SIZE) + BAR_HEIGHT, CELL_SIZE - 2, CELL_SIZE - 2, cell_color);

				const char *val_text = TextFormat("%.2f", cell_value);
				int text_w = MeasureText(val_text, 20);
				DrawText(val_text, (c * CELL_SIZE) + (CELL_SIZE / 2) - (text_w / 2), 
									(r * CELL_SIZE) + BAR_HEIGHT + (CELL_SIZE / 2) - 10, 20, WHITE);
			}
		}
		
		// Draw calibration instructions
		DrawRectangle(0, 0, WINDOW_WIDTH, BAR_HEIGHT, Fade(BLACK, 0.95f));
		DrawLine(0, BAR_HEIGHT - 1, WINDOW_WIDTH, BAR_HEIGHT - 1, GRAY);
		
		/** LEGACY CURVE CALIBRATION
		if (processor.m_state == Processor::STATE_UNCALIBRATED) {
			DrawText("SYSTEM OPERATIONAL", 25, 16, 22, ORANGE);
			const char *txt = "Press [C] to initiate Multi-Point Calibration Wizard";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (processor.m_state == Processor::STATE_CAPTURE_LOW) {
			DrawText("CALIBRATION STEP 1/3: ZERO WEIGHT", 25, 16, 22, ORANGE);
			const char *txt = "Clear sensor completely. Press [ENTER] when cleared...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (processor.m_state == Processor::STATE_CAPTURE_MID) {
			DrawText("CALIBRATION STEP 2/3: MID WEIGHT", 25, 16, 22, ORANGE);
			const char *txt = "Place mid-range reference load onto sensor. Press [ENTER] when stable...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (processor.m_state == Processor::STATE_CAPTURE_HIGH) {
			DrawText("CALIBRATION STEP 3/3: HIGH WEIGHT", 25, 16, 22, ORANGE);
			const char *txt = "Place high-range reference load onto sensor. Press [ENTER] to compute curve...";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (processor.m_state == Processor::STATE_READY) {
			DrawText("CALIBRATION PROFILES ACTIVE", 25, 16, 22, LIME);
			const char *txt = "Matrix hardware curve computed. Press [R] to clear mapping and reset.";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		}
		**/

		if (processor.m_state == Processor::STATE_UNCALIBRATED) {
			DrawText("SYSTEM OPERATIONAL", 25, 16, 22, ORANGE);
			const char *txt = "Press [C] to tare sensor. Press [R] to reset.";
			int text_width = MeasureText(txt, STATUS_TEXT_SIZE);
			DrawText(txt, WINDOW_WIDTH - text_width - RIGHT_MARGIN, TEXT_Y, STATUS_TEXT_SIZE, LIGHTGRAY);
		} else if (processor.m_state == Processor::STATE_READY) {
			DrawText("TARED", 25, 16, 22, LIME);
			const char *txt = "Baseline active. Press [R] to reset.";
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
		float box_width = 250;
		float box_x = (float)(WINDOW_WIDTH - box_width - 10);
		float box_y = (float)(WINDOW_HEIGHT + BAR_HEIGHT + 7);
		GuiComboBox({ box_x, box_y, box_width, 46 }, port_list, &dropdown_active_index);
		
		// The Text Input Box
		if (GuiTextBox({ (float)(WINDOW_WIDTH - 600), (float)(WINDOW_HEIGHT + BAR_HEIGHT + 7), 240, 46 }, config_filepath, 255, config_edit_mode)) {
			config_edit_mode = !config_edit_mode;
		}
		
		// The Load Button
		if (GuiButton({ (float)(WINDOW_WIDTH - 350), (float)(WINDOW_HEIGHT + BAR_HEIGHT + 7), 80, 46 }, "LOAD")) {
			ProcessingConfig newConfig;
			
			// Load whatever path is currently typed in the text box
			if (newConfig.loadFromIni(config_filepath)) {
				processor.UpdateConfig(newConfig);
				std::cout << "[CONFIG] Success: Engine updated with " << config_filepath << std::endl;
				
				popup_timer = 3.0f;
				snprintf(popup_message, sizeof(popup_message), "CONFIG LOADED: %s", GetFileName(config_filepath));
			} else {
				std::cout << "[CONFIG ERROR] Failed to parse " << config_filepath << std::endl;
				
				popup_timer = 3.0f;
				snprintf(popup_message, sizeof(popup_message), "ERROR: INVALID FILE");
			}
		}
		
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
		
		// Draw Popup
		if (popup_timer > 0.0f) {
			popup_timer -= GetFrameTime();
			
			float alpha = (popup_timer < 1.0f) ? popup_timer : 1.0f;
			
			int msg_width = MeasureText(popup_message, 20);
			int box_width = msg_width + 40;
			int box_x = (WINDOW_WIDTH - box_width) / 2;
			int box_y = BAR_HEIGHT + 20;
			
			DrawRectangle(box_x, box_y, box_width, 40, Fade(BLACK, alpha * 0.85f));
			DrawRectangleLines(box_x, box_y, box_width, 40, Fade(LIME, alpha));
			
			DrawText(popup_message, box_x + 20, box_y + 10, 20, Fade(LIME, alpha));
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

/**
 * @file main.cpp
 * @author Aidan Mohammed-Ali
 * @brief Visualisation tool.
 * * This file takes the sensor outputs and displays.
 * @date 2026-05-23
 */

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
	#include <windows.h>
#else
	#include <unistd.h>
	#include <fcntl.h>
	#include <termios.h>
#endif

// Geometry configuration 
const int ROWS = 8;
const int COLS = 16;
const int CELL_SIZE = 150;
const int WINDOW_WIDTH = COLS * CELL_SIZE;
const int WINDOW_HEIGHT = ROWS * CELL_SIZE;

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
	SerialHandle handle = CreateFileA(portName, GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
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
	uint32_t totalBytesRead = 0;
	
#if defined(_WIN32)
	DWORD bytesRead;
	while (totalBytesRead < bytesToRead) {
		if (ReadFile(handle, buffer + totalBytesRead, bytesToRead - totalBytesRead, &bytesRead, nullptr)) {
			if (bytesRead == 0) {
				return false;
			}
			totalBytesRead += bytesRead;
		} else {
			return false;
		}
	}
	return true;
#else
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

int main() {
	// Cross-Platform Port Name Resolution
#if defined(_WIN32)
	const char *default_port = "\\\\.\\COM3";
#elif defined(__APPLE__)
	const char *default_port = "/dev/cu.usbmodem101";
#else
	const char *default_port = "/dev/ttyACM0";
#endif

	// Initialise Operating System Serial Connection
	std::cout << "[INIT] Connecting to tactile matrix on " << default_port << "..." << std::endl;
	SerialHandle serial = OpenSerialPort(default_port);
	bool hardware_online = (serial != INVALID_SERIAL);
	
	if (hardware_online) {
		std::cout << "[SUCCESS] Hardware online. Streaming telemetry..." << std::endl;
	} else {
		std::cout << "[WARN] Hardware offline or permission denied. Running in Simulation Mode." << std::endl;
	}
	
	// Initialise Raylib Accelerated Window Engine
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Tactile Skin Visualiser");
	SetTargetFPS(60);
	
	TactileFrame current_frame = {0};
	float simulation_time = 0.0f;
	
	// Run loop
	while (!WindowShouldClose()) {
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
				
				DrawRectangle(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE - 2, CELL_SIZE - 2, cell_color);
			}
		}
		
		EndDrawing();
	}
	
	// De-allocation Phase
	std::cout << "[SHUTDOWN] Closing graphics pipeline modules..." << std::endl;
	CloseWindow();
	
	if (hardware_online) {
#if defined(_WIN32)
	CloseHandle(serial);
#else
	close(serial);
#endif
	std::cout << "[SHUTDOWN] Serial interface closed cleanly." << std::endl;
	}
	
	return 0;
}

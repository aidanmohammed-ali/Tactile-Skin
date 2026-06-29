# Visualiser

A high-performance, cross-platform desktop utility built with C++ and Raylib to map, calibrate, and simulate physical data from the Tactile Skin hardware matrix.

<img width="2418" height="1391" alt="visualiser" src="https://github.com/user-attachments/assets/48c41b93-6515-4664-a24f-1f82fb56eff1" />

## Compilation Guide

### Prerequisites
Before compiling, you must install the native development tools and **Raylib** setup for your operating system.

### Windows (MSYS2 / MinGW-w64)
The most reliable way to compile the C++ Raylib environment on Windows is via the MSYS2 UCRT64 subsystem.

1. Download and install [MSYS2](https://www.msys2.org/).
2. Open the **MSYS2 UCRT64** terminal from the Windows Start menu (do not use the standard MSYS terminal).
3. Update the package database and install the GCC compiler alongside the pre-compiled Raylib package:

   ```bash
   pacman -Syu
   pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-raylib
4. Navigate to the project directory.
5. Compile the program. Windows requires explicitly linking  the underlying OS graphics and media subsystems:

   ```bash
   g++ main.cpp ../common/config.cpp -o visualiser.exe -O2 -Wall -I../common -lraylib -lopengl32 -lgdi32 -lwinmm
6. Run the executable directly from terminal:

   ```bash
   ./visualiser.exe

### Linux (Ubuntu / Debian)
Linux natively supports the POSIX serial architecture and only requires the standard build toolchain.

1. Install the required development packages and Raylib:

   ```bash
   sudo apt update
   sudo apt install build-essential libraylib-dev
   ```
2. Navigate to the project directory and compile:

   ```bash
   g++ main.cpp ../common/config.cpp -o visualiser -O2 -Wall -I../common -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
   ```
3. Run the executable:

   ```bash
   ./visualiser
   ```

### Linux (Fedora / RHEL)
Fedora uses the `dnf`  manager and slightly different package names. But the compilation process is identical to Debian-based systems.

1. Install the GCC compiler suite and Raylib development headers:

   ```bash
   sudo dnf install gcc-c++ raylib-devel
   ```

2. Navigate to the project directory and compile:

   ```bash
   g++ main.cpp ../common/config.cpp -o visualiser -O2 -Wall -I../common -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
   ```
3. Run the executable:

   ```bash
   ./visualiser
   ```

### macOS (Apple Silicon & Intel)
Because macOS is UNIX-based, the application utilises the exact same non-blocking POSIX serial logic as Linux. The easiest way to install dependencies is via Homebrew.

1. Install the Xcode Command Line Tools and Raylib via Homebrew:

   ```bash
   xcode-select --install
   brew install raylib
   ```
2. Navigate to the project directory. macOS requires linking Apple's native rendering frameworks:

   ```bash
   clang++ main.cpp ../common/config.cpp -o visualiser -O2 -Wall -I../common -lraylib -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL
   ```
3. Run the executable:

   ```bash
   ./visualiser
   ```

## Hardware Connection & Usage

When connecting the physical Tactile Skin matrix via USB, ensure the correct hardware node is selected from the menu to avoid falling back to Simulation Mode.

* Windows: Open **Device Manager** and expand _Ports_ to find your assigned port (e.g., `COM7`). Select this exact port from the visualiser's menu.
* Linux (Ubuntu/Fedora): The hardware will typically mount to `/dev/ttyACM0` or `/dev/ttyACM1`.
  * _Note: If the application throws an immediate connection error, ensure the user has serial port permissions by running: `sudo usermod -a -G dialout $USER` (requires a logout/login to take effect)._
* macOS: Apple handles the serial devices differently. Look for paths like `/dev/cu.usbmodem` or `/dev/tty.usbmodem`. (This path may need to be manually typed into the codebase if it is not present in the default menu).

## Calibration Wizard

Once the hardware link is live (indicated by the green status bar), clear the sensor and press [C] to Tare. Press [R] at any time to clear the mapping and reset to default.

## Configuration (.ini) Setup

The application features a built-in configuration profile manager. Users can type the specific name or path of an `.ini` file into the UI text entry box and click the Load button to instantly apply different profiles.

If no custom file is specified, the application defaults to the hardcoded values. Configuration files must be stored in the same directory as the visualiser executable. Example `.ini` files can be found in this repository.

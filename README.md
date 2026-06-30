# Tactile-Skin

This project focuses on creating a thin, flexible tactile skin that measures mechanical pressure, rather than just proximity. This repository contains the firmware and signal processing logic to convert physical deformation and capacitance into usable digital data and the schematics and Gerber files for the requisite hardware.

## Repository Structure

### Firmware
The core software running on the microcontroller, built using PlatformIO. It handles high-speed analog acquisition from the skin matrix and packs it for serial transmission.
* `/src`: Main firmware source files (`.c` & `.cpp`).
* `/include`: Global header files (`.h`).
* `platformio.ini`: The core PlatformIO configuration script.

### Tools (`/tools`)
* `/common`: Shared, hardware-agnostic C++ data engine containing the core signal processing logic and GUI rendering requirements. Folder required to compile both visualiser and raw-streaming.
* `/visualiser`: A high-performance, cross-platform desktop utility built with Raylib. Features include:
  * **Multi-Instance Isolation:** Run multiple parallel instances side-by-side to monitor independent microcontrollers simultaneously.
  * **Dynamic Port Mapping:** Hot-swap hardware link contexts on-the-fly via an integrated UI selector.
  * **Zero-Lag Simulation Mode:** Automated fallback to a high-fidelity Gaussian simulation model when running offline.
  * **Dynamic Profile Loading:** Real-time parameter loading via local `.ini` configuration files to calibrate individual sensor variations dynamically without recompiling the codebase.
* `/raw-streaming`: A utility based on the visualiser for streaming the raw sensor data without any signal processing. Raw data across a specified time frame can be saved in `.csv` format and used for data analysis.

### Hardware (`/hardware`)
* `/fpc`: Flexible Printed Circuit manufacturing files.
  * Complete production-ready Gerber packages.
* `/pcb`: Main rigid Printed Circuit Board design histories.
  * Main interface between the sensor and microcontroller.

## Contribution Guidelines

To keep the project history clean and professional, please follow these conventions:

### Git Commit Prefixes
Use **Conventional Commits**. Start every commit message with one of these:
* **`Docs:`** Documentation changes or code comments (e.g., `Docs: update README`).
* **`Feat:`** New features or logic (e.g., `Feat: implementation of scan loop`).
* **`Fix:`** Bug fixes (e.g., `Fix: corrected index overflow`).
* **`Chore:`** Maintenance tasks, library imports, or project scaffolding (e.g., `Chore: import USB core libraries`).

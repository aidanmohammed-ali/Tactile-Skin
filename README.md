# Tactile-Skin

This project focuses on creating a thin, flexible tactile skin that measures mechanical pressure, rather than just proximity. This repository contains the firmware and signal processing logic to convert physical deformation into usable digital data.

## System Overview
* **Sensing Mechanism:**
* **Materials:**
* **Goal:**

## Repository Structure

### Firmware
The core software running on the microcontroller, built using PlatformIO. It handles high-speed analog acquisition from the skin matrix and packs it for serial transmission.
* `/src`: Main firmware source files (`.c` & `.cpp`).
* `/include`: Global header files (`.h`).
* `platformio.ini`: The core PlatformIO configuration script.

### Tools
* `/visualiser`: A high-performance, cross-platform desktop utility built with Raylib. Features include:
  * **Multi-Instance Isolation:** Run multiple parallel instances side-by-side to monitor independent microcontrollers simultaneously.
  * **Dynamic Port Mapping:** Hot-swap hardware link contexts on-the-fly via an integrated UI selector.
  * **Multi-Point Calibration Wizard:** Interactive step-by-step UI to trigger hardware tares (Low, Mid, High) directly on the MCU.
  * **Zero-Lag Simulation Mode:** Automated fallback to a high-fidelity Gaussian simulation model when running offline.

### Hardware
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

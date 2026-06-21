/**
 * @file config.cpp
 * @author Aidan Mohammed-Ali
 * @brief Implementation of the runtime configuration parser for the tactile skin processing engine
 * @date 2026-06-12
 */

#include <stdio.h>
#include <string.h>
#include "config.hpp"

/**
 * @brief Reads settings from an INI file.
 * @param filepath The location of the file on disk.
 * @return true if successful, false if the file couldn't be opened.
 */
bool ProcessingConfig::loadFromIni(const char *filepath) {
	// Open the file
	FILE *file = fopen(filepath, "r");
	if (!file) {
		printf("Engine Error: Could not open INI file %s\n", filepath);
		return false;
	}
	
	char line[256];
	char currentSection[64] = "";
	
	// Read the file line-by-line
	while (fgets(line, sizeof(line), file)) {
		// Skip empty lines or comments
		if (line[0] == '\n' || line[0] == '\r' || line[0] == ';' || line[0] == '#') {
			continue;
		}
		
		// Detect if the line is a [Section]
		if (line[0] == '[') {
			sscanf(line, "[%63[^]]]", currentSection);
			continue;
		}
		
		// Parse the Key = Value strings
		char key[64];
		float val;
		
		if (sscanf(line, "%63[^= ] = %f", key, &val) == 2) {
			// Map the parsed float to the correct struct variable
			if (strcmp(currentSection, "Filters") == 0) {
				if (strcmp(key, "FilterAlpha") == 0) {
					filterAlpha = val;
				} else if (strcmp(key, "SpatialCenterWeight") == 0) {
					spatialCenterWeight = val;
				} else if (strcmp(key, "SharpenAmount") == 0) {
					sharpenAmount = val;
				}
			} else if (strcmp(currentSection, "Thresholds") == 0) {
				if (strcmp(key, "NoiseThreshold") == 0) {
					noiseThreshold = val;
				} else if (strcmp(key, "RawDeltaThreshold") == 0) {
					rawDeltaThreshold = val;
				} else if (strcmp(key, "MaxDelta") == 0) {
					maxDelta = val;
				}
			}
		}
	}
	
	fclose(file);
	return true;
}

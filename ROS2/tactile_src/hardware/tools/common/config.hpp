/**
 * @file config.hpp
 * @author Aidan Mohammed-Ali
 * @brief Header file for config.cpp containing the structure of tuned values.
 * @date 2026-06-12
 */

/**
 * @brief Stores the tuned values for processing.
 */
struct ProcessingConfig {
	float filterAlpha = 0.09f;
	float spatialCenterWeight = 0.90f;
	float sharpenAmount = 2.0f;
	float noiseThreshold = 0.16f;
	float rawDeltaThreshold = 800.0f;
	float maxDelta = 45000.0f;
	
	/**
	 * @brief Reads settings from an INI file.
	 * @param filepath The location of the file on disk.
	 * @return true if successful, false if the file couldn't be opened.
	 */
	bool loadFromIni(const char *filepath);
};

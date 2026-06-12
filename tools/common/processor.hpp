/**
 * @file processor.hpp
 * @author Aidan Mohammed-Ali
 * @brief PC-side calibration and signal processing engine for tactile skin.
 * @date 2026-06-04
 */

#pragma once
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "config.hpp"

/**
 * @brief Handles tactile matrix low-pass filtering and multi-point quadratic curve fitting calibration.
 * @note Desgined with a flat data layout for high-performance array streaming.
 */
class Processor {
public:
	/**
	 * @brief Tracks the active execution step of the calibration wizard state machine.
	 */
	enum WizardState {
		STATE_UNCALIBRATED = 0,
		STATE_CAPTURE_LOW,
		STATE_CAPTURE_MID,
		STATE_CAPTURE_HIGH,
		STATE_READY
	};
	
	/**
	 * @brief Stores the calculated quadratic coefficients for a single sensor element.
	 */
	struct CurveParams {
		float a = 0.0f;
		float b = 1.0f;
		float c = 0.0f;
	};

public:
	int m_numTaxels;
	WizardState m_state;
	ProcessingConfig m_config;
		
	std::vector<CurveParams> m_curves;
	std::vector<float> m_baselines;
	std::vector<float> m_smoothedData;
	
	std::vector<float> m_history0;
	std::vector<float> m_history1;
	std::vector<float> m_history2;
	std::vector<float> m_history3;
	std::vector<float> m_history4;
	
	std::vector<uint16_t> m_weightLow;
	std::vector<uint16_t> m_weightMid;
	std::vector<uint16_t> m_weightHigh;
	
	std::vector<float> m_spatialTemp;

private:
	static constexpr int GRID_WIDTH = 16;
	static constexpr int GRID_HEIGHT = 8;
	
public:
	/**
	 * @brief Initialise data buffers and configure uncalibrated tracking defaults.
	 * @param num Total number of operational sensor nodes in the conected matrix.
	 * @param config Optional initial configuration parameters.
	 */
	Processor(int num = 128, const ProcessingConfig& config = ProcessingConfig()) {
		m_numTaxels = num;
		m_state = STATE_UNCALIBRATED;
		
		m_config = config;
		
		m_curves.resize(m_numTaxels);
		m_baselines.resize(m_numTaxels, 0.0f);
		m_smoothedData.resize(m_numTaxels, 0.0f);
		
		m_history0.resize(m_numTaxels, 0.0f);
		m_history1.resize(m_numTaxels, 0.0f);
		m_history2.resize(m_numTaxels, 0.0f);
		m_history3.resize(m_numTaxels, 0.0f);
		m_history4.resize(m_numTaxels, 0.0f);
		
		m_weightLow.resize(m_numTaxels, 0);
		m_weightMid.resize(m_numTaxels, 0);
		m_weightHigh.resize(m_numTaxels, 0);
		
		m_spatialTemp.resize(m_numTaxels, 0.0f);
		
		ResetCalibration();
	}
	
	/**
	 * @brief Dynamically update settings parameters from the GUI at runtime.
	 * @param newConfig New configuration parameters.
	 */
	void UpdateConfig(const ProcessingConfig& newConfig) {
		m_config = newConfig;
	}
	
	/**
	 * @brief Steps the calibration sequence forward and snapshots incoming raw data frames.
	 * @param rawFrame Pointer to the start of the 128-element uint16_t array received from the MCU.
	 */
	void AdvanceWizard(const uint16_t *rawFrame) {
		if (rawFrame == nullptr) {
			return;
		}
		
		switch(m_state) {
			case STATE_UNCALIBRATED:
				m_state = STATE_CAPTURE_LOW;
				break;
			
			case STATE_CAPTURE_LOW:
				std::memcpy(m_weightLow.data(), rawFrame, m_numTaxels * sizeof(uint16_t));
				m_state = STATE_CAPTURE_MID;
				break;
			
			case STATE_CAPTURE_MID:
				std::memcpy(m_weightMid.data(), rawFrame, m_numTaxels * sizeof(uint16_t));
				m_state = STATE_CAPTURE_HIGH;
				break;
			
			case STATE_CAPTURE_HIGH:
				std::memcpy(m_weightHigh.data(), rawFrame, m_numTaxels * sizeof(uint16_t));
				CalculateCurve();
				m_state = STATE_READY;
				break;
				
			case STATE_READY:
				ResetCalibration();
				break;
		}
	}
	
	/**
	 * @brief Resets all coefficients back to a 1:1 uncalibrated state.
	 */
	void ResetCalibration() {
		m_state = STATE_UNCALIBRATED;
		for (int i = 0; i < m_numTaxels; ++i) {
			m_curves[i].a = 0.0f;
			m_curves[i].b = 1.0f;
			m_curves[i].c = 0.0f;
			
			m_baselines[i] = 0.0f;
		}
	}
	
	/**
	 * @brief Filters raw sensor frames and applies active calibration coefficients.
	 * @param rawFrame Pointer to the incoming raw buffer.
	 * @param processedFrame Pointer to the output destination array to be filled with clean floats.
	 */
	void ProcessFrame(const uint16_t *rawFrame, float *processedFrame) {
		if (rawFrame == nullptr || processedFrame == nullptr) {
			return;
		}
		
		FilterFrame(rawFrame);	
		std::vector<float> processedData(m_numTaxels, 0.0f);
		
		for (int i = 0; i < m_numTaxels; ++i) {			
			float zeroed = m_smoothedData[i] - m_baselines[i];
			
			if (zeroed < m_config.rawDeltaThreshold) {
				zeroed = 0.0f;
			}
			
			float out = zeroed / m_config.maxDelta;
			
			if (out < 0.0f) {
				out = 0.0f;
			}
			
			if (out < m_config.noiseThreshold) {
				out = 0.0f;
			}
			
			if (out > 1.0f) {
				out = 1.0f;
			}
			
			processedData[i] = out;
		}
		
		// Spatial smoothing
		const float neighbourWeight = (1.0f - m_config.spatialCenterWeight) / 4.0f;
		
		for (int i = 0; i < m_numTaxels; ++i) {
			int row = i / GRID_WIDTH;
			int col = i % GRID_WIDTH;
			
			float currentVal = processedData[i];
			float spatialSum = currentVal * m_config.spatialCenterWeight;
			float missingWeight = 0.0f;
			
			if (col > 0) {
				spatialSum += processedData[i - 1] * neighbourWeight;
			} else {
				missingWeight += neighbourWeight;
			}
			
			if (col < GRID_WIDTH - 1) {
				spatialSum += processedData[i + 1] * neighbourWeight;
			} else {
				missingWeight += neighbourWeight;
			}
			
			if (row > 0) {
				spatialSum += processedData[i - GRID_WIDTH] * neighbourWeight;
			} else {
				missingWeight += neighbourWeight;
			}
			
			if (row < GRID_HEIGHT - 1) {
				spatialSum += processedData[i + GRID_WIDTH] * neighbourWeight;
			} else {
				missingWeight += neighbourWeight;
			}
			
			if (missingWeight > 0.0f) {
				spatialSum += currentVal * missingWeight;
			}
			
			processedFrame[i] = spatialSum;
		}
		
		// Unsharp Masking
		for (int i = 0; i < m_numTaxels; ++i) {
			float original = processedData[i];
			float blurred = processedFrame[i];
			
			float sharpened = original + m_config.sharpenAmount * (original - blurred);
			
			if (sharpened < 0.0f) {
				sharpened = 0.0f;
			}
			
			if (sharpened > 1.0f) {
				sharpened = 1.0f;
			}
			
			processedFrame[i] = sharpened;
		}
	}

	/**
	 * @brief Tare the sensor to reset the baseline to zero.
	 * @param rawFrame Pointer to the incoming raw buffer.
	 */
	void Tare(const uint16_t *rawFrame) {
		if (rawFrame == nullptr) {
			return;
		}
		
		FilterFrame(rawFrame);
		
		for (int i = 0; i < m_numTaxels; ++i) {
			m_baselines[i] = m_smoothedData[i];
		}
	}

private:
	/**
	 * @brief Core filtering engine using 5-tap median filter and an EMA.
	 * @param rawFrame Pointer to the incoming raw buffer.
	 */
	void FilterFrame(const uint16_t *rawFrame) {
		for (int i = 0; i < m_numTaxels; ++i) {
			float rawVal = (float)rawFrame[i];
			
			// 5-Tap Median filter
			m_history4[i] = m_history3[i];
			m_history3[i] = m_history2[i];
			m_history2[i] = m_history1[i];
			m_history1[i] = m_history0[i];
			m_history0[i] = rawVal;
			
			float h0 = m_history0[i];
			float h1 = m_history1[i];
			float h2 = m_history2[i];
			float h3 = m_history3[i];
			float h4 = m_history4[i];
			float medianVal;
			
			if (h0 > h1) std::swap(h0, h1);
			if (h2 > h3) std::swap(h2, h3);
			if (h0 > h2) std::swap(h0, h2);
			if (h1 > h3) std::swap(h1, h3);
			if (h1 > h2) std::swap(h1, h2);
			if (h0 > h4) std::swap(h0, h4);
			if (h2 > h4) std::swap(h2, h4);
			if (h1 > h2) std::swap(h1, h2);
			if (h3 > h4) std::swap(h3, h4);
			
			medianVal = h2;
			
			// Apply low-pass EMA filter
			m_smoothedData[i] = (m_config.filterAlpha * medianVal) + ((1.0f - m_config.filterAlpha) * m_smoothedData[i]);
		}
	}
	
	/**
	 * @brief Solves the Cramer's rule matrix system to generate curve coefficients.
	 */
	void CalculateCurve() {
		const float y0 = 0.0f;
		const float y1 = 0.5;
		const float y2 = 1.0f;
		
		for (int i = 0; i < m_numTaxels; ++i) {
			double x0 = (double)m_weightLow[i] / 65535.0;
			double x1 = (double)m_weightMid[i] / 65535.0;
			double x2 = (double)m_weightHigh[i] / 65535.0;
			
			double X1 = x1 - x0;
			double X2 = x2 - x0;
			
			double Y1 = (double)y1 - (double)y0;
			double Y2 = (double)y2 - (double)y0;
			
			double det = (X1 * X1 * X2) - (X1 * X2 * X2);;
			
			if (std::abs(det) < 1e-2) {
				m_curves[i].a = 0.0f;
				m_curves[i].b = 1.0f;
				m_curves[i].c = (float)x0;
				continue;
			}
			
			double inv_det = 1.0 / det;
			
			double a = ((Y1 * X2) - (Y2 * X1)) * inv_det;
			double b = ((Y2 * X1 * X1) - (Y1 * X2 * X2)) * inv_det;
			
			m_curves[i].a = (float)a;
			m_curves[i].b = (float)b;
			m_curves[i].c = (float)x0;
		}
	}
};

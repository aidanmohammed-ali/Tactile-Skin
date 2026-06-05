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
		float b = 0.0f;
		float c = 0.0f;
	};

public:
	int m_numTaxels;
	WizardState m_state;
	float m_noiseThreshold;
		
	std::vector<CurveParams> m_curves;
	std::vector<float> m_smoothedData;
	
	std::vector<uint16_t> m_weightLow;
	std::vector<uint16_t> m_weightMid;
	std::vector<uint16_t> m_weightHigh;
	
public:
	/**
	 * @brief Initialise data buffers and configure uncalibrated tracking defaults.
	 * @param num Total number of operational sensor nodes in the conected matrix.
	 */
	Processor(int num = 128) {
		m_numTaxels = num;
		m_state = STATE_UNCALIBRATED;
		m_noiseThreshold = 0.2f;
		
		m_curves.resize(m_numTaxels);
		m_smoothedData.resize(m_numTaxels, 0.0f);
		m_weightLow.resize(m_numTaxels, 0);
		m_weightMid.resize(m_numTaxels, 0);
		m_weightHigh.resize(m_numTaxels, 0);
		
		ResetCalibration();
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
		
		const float alpha = 0.02f;
		
		for (int i = 0; i < m_numTaxels; ++i) {
			// Apply low-pass EMA filter
			m_smoothedData[i] = (alpha * (float)rawFrame[i]) + ((1.0f - alpha) * m_smoothedData[i]);
			float x = m_smoothedData[i] / 65535.0f;
			
			if (m_state == STATE_READY) {
				float zeroed = m_smoothedData[i] - m_curves[i].c;
				float out = zeroed * m_curves[i].b;
				
				if (out < m_noiseThreshold) {
					out = 0.0f;
				}
				if (out > 1.0f) {
					out = 1.0f;
				}
				
				processedFrame[i] = out;
			} else {
				processedFrame[i] = x;
			}
		}
	}

	/**
	 * @brief Tare the sensor to reset the baseline to zero.
	 */
	void Tare(const uint16_t *rawFrame) {
		for (int i = 0; i < m_numTaxels; ++i) {
			m_curves[i].c = (float)rawFrame[i];
			m_curves[i].b = 1.0f / (65535.0f - m_curves[i].c);
			m_curves[i].a = 0.0f;
		}
		m_state = STATE_READY;
	}

private:
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

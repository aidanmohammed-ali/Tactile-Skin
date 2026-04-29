/**
 * @file matrix_scan.h
 * @author Aidan Mohammed-Ali
 * @brief High-speed scanning logic for tactile skin.
 * @version 0.1
 * @date 2026-04-29
 */

#ifndef MATRIX_SCAN_H
#define MATRIX_SCAN_H

#include <stdint.h>

/** Matrix Dimensions **/
#define MATRIX_ROWS		32
#define MATRIX_COLS		32
#define MATRIX_SIZE		(MATRIX_ROWS * MATRIX_COLS)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs a full scan of the matrix.
 * * @param buffer Pointer to the memory where values will be stored.
 */
void matrix_scan_grid(uint16_t *buffer);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_SCAN_H

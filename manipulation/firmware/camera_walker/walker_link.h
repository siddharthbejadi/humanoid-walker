/* Laptop-controlled "walker" mode: receives 44-byte frames on USART1 (ST-LINK virtual COM port),
 * clamps them to the calibrated limits and drives the two PCA9685 boards.
 *
 * Modes:  CONSOLE (default, servo calibration console, outputs untouched)
 *         ARMED   (after 'run confirm': waiting up to 120 s for the first valid frame; outputs untouched)
 *         RUN     (frames drive the servos; no valid frame for 100 ms -> relax + back to CONSOLE)
 */
#ifndef WALKER_LINK_H
#define WALKER_LINK_H

#include <stdbool.h>
#include "main.h"

typedef enum { WALKER_CONSOLE = 0, WALKER_ARMED = 1, WALKER_RUN = 2 } WalkerMode;

void Walker_Init(I2C_HandleTypeDef *i2c, UART_HandleTypeDef *uart);
void Walker_Process(void);            /* call every main-loop pass */
WalkerMode Walker_Mode(void);
bool Walker_PopByte(uint8_t *byte);   /* serial bytes buffered by the interrupt (used by the console) */
bool Walker_Arm(bool dry);            /* 'run confirm' (live) or 'run dry' (frames checked, PCA outputs stay OFF) */
void Walker_Relax(const char *reason);   /* RELEASE: all servos limp (only on an explicit request) */
void Walker_Hold(const char *reason);    /* HOLD: stop taking frames but keep every servo at its last pulse (stays stiff) */
void ImuStream_Set(uint8_t enable);   /* defined in main.c: IMU text rows on/off */

#endif

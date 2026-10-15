#ifndef SERVO_CALIBRATION_H
#define SERVO_CALIBRATION_H

#include <stdbool.h>
#include "main.h"

void ServoCal_Init(I2C_HandleTypeDef *i2c, UART_HandleTypeDef *uart);
void ServoCal_Process(void);
/* Used by walker_link.c */
bool ServoCal_GetLimits(unsigned index, uint8_t *address, uint8_t *channel, uint16_t *min_us, uint16_t *max_us);
void ServoCal_RelaxAll(void);
void ServoCal_NoteCommand(unsigned index, uint16_t pulse_us);   /* record a pulse the walker holds, so the console knows the pose */
bool ServoCal_JointCommanded(unsigned index, uint16_t *last_us);   /* true if the console has commanded this joint since the last release */
bool ServoCal_JointUnused(unsigned index);   /* neutral 0 in walker_limits.h = not driven in walker mode */

#endif

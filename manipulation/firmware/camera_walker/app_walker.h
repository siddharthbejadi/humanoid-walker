#ifndef APP_WALKER_H
#define APP_WALKER_H
/* Laptop-controlled servo mode (walker) running as a FreeRTOS task inside the camera-capture firmware:
 * the board streams UVC video AND accepts 44-byte servo frames on USART1 (ST-LINK virtual COM port). */
void app_walker_start(void);
#endif

#include "app_walker.h"

#include "FreeRTOS.h"
#include "task.h"
#include "stm32n6xx_hal.h"
#include "servo_calibration.h"
#include "walker_link.h"

extern UART_HandleTypeDef huart1;      /* initialised by main.c CONSOLE_Config() (115200 8N1, PE5/PE6) */
static I2C_HandleTypeDef hi2c1;        /* servo drivers PCA9685 0x40 / 0x41 on I2C1 (PC1 SDA, PH9 SCL); the camera sensor uses I2C2 */

static StaticTask_t walker_task;
static StackType_t walker_stack[1024];

/* The walker modules call this to switch the IMU text stream; there is no IMU in the camera build. */
void ImuStream_Set(uint8_t enable) { (void)enable; }

/* Only I2C1 is handled here; the BSP's I2C2 (camera sensor) keeps its own init. */
void HAL_I2C_MspInit(I2C_HandleTypeDef *hi2c)
{
  GPIO_InitTypeDef gpio = {0};
  RCC_PeriphCLKInitTypeDef clk = {0};

  if (hi2c->Instance != I2C1) return;
  clk.PeriphClockSelection = RCC_PERIPHCLK_I2C1;
  clk.I2c1ClockSelection = RCC_I2C1CLKSOURCE_PCLK1;
  (void)HAL_RCCEx_PeriphCLKConfig(&clk);
  __HAL_RCC_I2C1_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  gpio.Mode = GPIO_MODE_AF_OD;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Alternate = GPIO_AF4_I2C1;
  gpio.Pin = GPIO_PIN_1;  HAL_GPIO_Init(GPIOC, &gpio);     /* I2C1_SDA */
  gpio.Pin = GPIO_PIN_9;  HAL_GPIO_Init(GPIOH, &gpio);     /* I2C1_SCL */
}

/* ~100 kHz timing computed from the real I2C kernel clock (the walker project's fixed value 0x30C0EDFF assumed ITS clock tree). */
static uint32_t i2c1_timing_100khz(void)
{
  uint32_t f = HAL_RCC_GetPCLK1Freq();
  uint32_t presc = (f / 8000000U);
  if (presc == 0U) presc = 1U;
  presc -= 1U;                                    /* f_presc ~ 8 MHz: 80 cycles per 100 kHz period */
  if (presc > 15U) presc = 15U;
  return (presc << 28) | (4UL << 20) | (2UL << 16) | (35UL << 8) | 43UL;   /* SCLDEL 4, SDADEL 2, SCLH 36 cycles, SCLL 44 cycles */
}

static void walker_thread_fct(void *arg)
{
  (void)arg;
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = i2c1_timing_100khz();
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK) { vTaskDelete(NULL); return; }       /* leave the camera running even if the servo bus fails */
  (void)HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE);
  (void)HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0);

  ServoCal_Init(&hi2c1, &huart1);
  Walker_Init(&hi2c1, &huart1);      /* USART1 RX interrupt + laptop-control mode ('run confirm') */
  for (;;)
  {
    Walker_Process();
    ServoCal_Process();
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

void app_walker_start(void)
{
  TaskHandle_t hdl = xTaskCreateStatic(walker_thread_fct, "walker", 1024, NULL, tskIDLE_PRIORITY + 2, walker_stack, &walker_task);
  configASSERT(hdl != NULL);
}

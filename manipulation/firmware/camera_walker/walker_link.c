#include "walker_link.h"

#include <stdio.h>
#include <string.h>

#include "servo_calibration.h"
#include "walker_frame.h"
#include "walker_limits.h"

#define PCA_LED0_ON_L     0x06U
#define PCA_FREQUENCY_HZ  50U
#define RING_SIZE         512U            /* power of two */
#define WATCHDOG_MS       100U            /* no valid frame for this long -> relax */
#define ARMED_TIMEOUT_MS  120000U         /* give up waiting for the first frame */
#define MAX_JUMP_US       200U            /* per-frame change guard (laptop slew is ~25 us/frame) */

/* Simulator joint order (what the policy outputs) -> index in the console's joint table.
 * Legs are certain. Arms (confirmed by the user): channel 2 = arm, channel 1 = elbow.
 * STILL A GUESS: sim hand_tip <- table "gripper" (channel 0, right 0 / left 10).
 * sim: 0 Lhip_torso 1 Lhip_upper 2 Llower_thigh 3 Lknee 4 Lfoot 5 Rhip_torso 6 Rhip_upper 7 Rlower_thigh
 *      8 Rknee 9 Rfoot 10 Lshoulder 11 Larm 12 Lelbow 13 Rshoulder 14 Rarm 15 Relbow 16 head 17 Lhand_tip 18 Rhand_tip */
static const uint8_t sim_to_table[WALKER_JOINTS] =
{
  14, 15, 16, 17, 18,    /* left leg   */
   4,  5,  6,  7,  8,    /* right leg  */
  13, 12, 11,            /* left: shoulder (ch3), arm (ch2), elbow (ch1) */
   3,  2,  1,            /* right: shoulder (ch3), arm (ch2), elbow (ch1) */
   9,                    /* head       */
  10,  0                 /* left / right hand tip */
};

static I2C_HandleTypeDef *w_i2c;
static UART_HandleTypeDef *w_uart;

static volatile uint8_t ring[RING_SIZE];
static volatile uint16_t ring_head, ring_tail;
static volatile uint32_t rx_dropped, rx_errors;

static WalkerMode mode = WALKER_CONSOLE;
static uint8_t dry_run;          /* 1: parse + validate frames but never write PWM to the PCA boards */
static WalkerParser parser;
static uint32_t bad_crc, frames_applied, jump_clamps;
static uint32_t armed_ms, last_valid_ms;
static uint16_t last_us[WALKER_JOINTS];
static uint8_t have_last[WALKER_JOINTS];

static void Print(const char *text)
{
  (void)HAL_UART_Transmit(w_uart, (const uint8_t *)text, (uint16_t)strlen(text), 100U);
}

/* ---- serial receive interrupt: bytes go to a ring buffer, nothing else happens here ---- */
void USART1_IRQHandler(void)
{
  USART_TypeDef *u = w_uart->Instance;
  if ((u->ISR & (USART_ISR_ORE | USART_ISR_FE | USART_ISR_NE)) != 0U)
  {
    u->ICR = USART_ICR_ORECF | USART_ICR_FECF | USART_ICR_NECF;
    rx_errors++;
  }
  while ((u->ISR & USART_ISR_RXNE_RXFNE) != 0U)
  {
    const uint8_t byte = (uint8_t)u->RDR;
    const uint16_t next = (uint16_t)((ring_head + 1U) & (RING_SIZE - 1U));
    if (next != ring_tail) { ring[ring_head] = byte; ring_head = next; }
    else rx_dropped++;
  }
}

bool Walker_PopByte(uint8_t *byte)
{
  if (ring_tail == ring_head) return false;
  *byte = ring[ring_tail];
  ring_tail = (uint16_t)((ring_tail + 1U) & (RING_SIZE - 1U));
  return true;
}

void Walker_Init(I2C_HandleTypeDef *i2c, UART_HandleTypeDef *uart)
{
  w_i2c = i2c;
  w_uart = uart;
  ring_head = ring_tail = 0U;
  WalkerParser_Init(&parser);
  SET_BIT(uart->Instance->CR1, USART_CR1_RXNEIE_RXFNEIE);
  HAL_NVIC_SetPriority(USART1_IRQn, 5U, 0U);
  HAL_NVIC_EnableIRQ(USART1_IRQn);
}

WalkerMode Walker_Mode(void) { return mode; }

void Walker_Relax(const char *reason)
{
  ServoCal_RelaxAll();
  memset(have_last, 0, sizeof(have_last));
  mode = WALKER_CONSOLE;
  WalkerParser_Init(&parser);
  char text[96];
  (void)snprintf(text, sizeof(text), "\r\nWALKER,RELAX,%s,frames=%lu,bad_crc=%lu\r\n",
                 reason, (unsigned long)frames_applied, (unsigned long)bad_crc);
  Print(text);
  Print("cal> ");
}

void Walker_Hold(const char *reason)
{
  mode = WALKER_CONSOLE;
  WalkerParser_Init(&parser);
  if (dry_run == 0U)
  {
    for (unsigned s = 0U; s < WALKER_JOINTS; s++)
    {
      const unsigned joint = sim_to_table[s];
      if (!ServoCal_JointUnused(joint) && (have_last[joint] != 0U)) ServoCal_NoteCommand(joint, last_us[joint]);
    }
  }
  char text[160];
  (void)snprintf(text, sizeof(text), "\r\nWALKER,HOLD,%s,frames=%lu,bad_crc=%lu\r\n"
                 "Servos keep their last pulses (still stiff). 'stand' = go to neutral, 'off all' = release.\r\n",
                 reason, (unsigned long)frames_applied, (unsigned long)bad_crc);
  Print(text);
  Print("cal> ");
}

bool Walker_Arm(bool dry)
{
#if !WALKER_LIMITS_VALID
  Print("REFUSED: walker mode needs calibrated limits. Export the calibration CSV, run tools/csv_to_limits.py,\r\n"
        "rebuild with WALKER_LIMITS_VALID 1 (walker_limits.h), then try again.\r\n");
  return false;
#else
  frames_applied = bad_crc = jump_clamps = 0U;
  memset(have_last, 0, sizeof(have_last));
  if (!dry)
  {
    /* LIVE needs a known starting point: every used joint must already be at the pulse the console commanded
     * ('stand'); the first laptop frame is then limited to +-MAX_JUMP_US from it. */
    for (unsigned s = 0U; s < WALKER_JOINTS; s++)
    {
      const unsigned joint = sim_to_table[s];
      uint16_t last;
      if (ServoCal_JointUnused(joint)) continue;
      if (!ServoCal_JointCommanded(joint, &last))
      {
        Print("REFUSED: LIVE mode needs a known starting pose. Type 'stand' first (servo supply ON, robot supported),\r\n"
              "then 'run confirm'. A relax, 'off all' or watchdog clears it, so repeat 'stand' after each one.\r\n");
        return false;
      }
      last_us[joint] = last;
      have_last[joint] = 1U;
    }
  }
  WalkerParser_Init(&parser);
  uint8_t drop;
  while (Walker_PopByte(&drop)) { }
  armed_ms = HAL_GetTick();
  dry_run = dry ? 1U : 0U;
  ImuStream_Set(1U);   /* the laptop needs the IMU rows */
  mode = WALKER_ARMED;
  if (dry_run)
  {
    ServoCal_RelaxAll();   /* all PCA outputs fully OFF; they stay OFF for the whole dry run */
    Print("WALKER,ARMED,dry: frames are checked and counted but NO PWM is written (safe with servo power on or off).\r\n"
          "Close this terminal and start the laptop script. 100 ms without a valid frame -> back to the console.\r\n");
    return true;
  }
  Print("WALKER,ARMED: close this terminal, start robot_host.py now. Outputs stay untouched until the first valid\r\n"
        "frame arrives (waits up to 120 s). No valid frame for 100 ms after that -> all servos released.\r\n"
        "WARNING: the first frame moves every joint at once to its commanded pulse. Support the robot.\r\n");
  return true;
#endif
}

static uint16_t PulseToCount(uint16_t pulse_us)
{
  return (uint16_t)(((uint32_t)pulse_us * 4096U * PCA_FREQUENCY_HZ + 500000U) / 1000000U);
}

static void Apply(const WalkerFrame *frame)
{
  uint16_t counts[2][16];
  uint8_t present[2][16];
  uint8_t max_channel[2] = {0U, 0U};
  memset(present, 0, sizeof(present));
  memset(counts, 0, sizeof(counts));

  for (unsigned s = 0U; s < WALKER_JOINTS; s++)
  {
    const unsigned joint = sim_to_table[s];
    if (ServoCal_JointUnused(joint)) continue;     /* left limp: never driven */
    uint8_t address, channel;
    uint16_t lo, hi;
    if (!ServoCal_GetLimits(joint, &address, &channel, &lo, &hi) || channel > 15U)
    {
      Walker_Hold("bad joint table");
      return;
    }
    uint16_t pulse = frame->pulse_us[s];
    if (pulse < lo) pulse = lo;
    if (pulse > hi) pulse = hi;
    if (have_last[joint])
    {
      const uint16_t previous = last_us[joint];
      if (pulse > previous + MAX_JUMP_US) { pulse = (uint16_t)(previous + MAX_JUMP_US); jump_clamps++; }
      else if (pulse + MAX_JUMP_US < previous) { pulse = (uint16_t)(previous - MAX_JUMP_US); jump_clamps++; }
    }
    last_us[joint] = pulse;
    have_last[joint] = 1U;
    const unsigned board = (address == 0x40U) ? 0U : 1U;
    counts[board][channel] = PulseToCount(pulse);
    present[board][channel] = 1U;
    if (channel > max_channel[board]) max_channel[board] = channel;
  }

  if (dry_run != 0U)    /* dry run: everything above was validated and clamped; nothing is written */
  {
    frames_applied++;
    return;
  }
  for (unsigned board = 0U; board < 2U; board++)
  {
    uint8_t buffer[64];
    const unsigned n = (unsigned)max_channel[board] + 1U;
    for (unsigned ch = 0U; ch < n; ch++)
    {
      if (present[board][ch])
      {
        buffer[4U * ch + 0U] = 0U;
        buffer[4U * ch + 1U] = 0U;
        buffer[4U * ch + 2U] = (uint8_t)(counts[board][ch] & 0xFFU);
        buffer[4U * ch + 3U] = (uint8_t)((counts[board][ch] >> 8) & 0x0FU);
      }
      else  /* a gap in the table: keep that channel fully OFF */
      {
        buffer[4U * ch + 0U] = 0U; buffer[4U * ch + 1U] = 0U;
        buffer[4U * ch + 2U] = 0U; buffer[4U * ch + 3U] = 0x10U;
      }
    }
    /* One auto-increment burst per board (MODE1 AI bit is set by the console's board init). */
    if (HAL_I2C_Mem_Write(w_i2c, (uint16_t)(((board == 0U) ? 0x40U : 0x41U) << 1), PCA_LED0_ON_L,
                          I2C_MEMADD_SIZE_8BIT, buffer, (uint16_t)(4U * n), 20U) != HAL_OK)
    {
      Walker_Hold("i2c write failed");
      return;
    }
  }
  frames_applied++;
}

void Walker_Process(void)
{
  static uint32_t next_heartbeat_ms;
  if (mode == WALKER_CONSOLE) return;

  /* Heartbeat once a second while armed/running: the laptop refuses to send frames (or stops) without it. */
  const uint32_t tick = HAL_GetTick();
  if ((int32_t)(tick - next_heartbeat_ms) >= 0)
  {
    next_heartbeat_ms = tick + 1000U;
    Print((mode == WALKER_ARMED) ? "WALKER,STATE,armed\r\n" : "WALKER,STATE,run\r\n");
  }

  WalkerFrame latest;
  uint8_t have = 0U;
  uint8_t byte;
  while (Walker_PopByte(&byte))
  {
    WalkerFrame frame;
    if (WalkerParser_Push(&parser, byte, &frame, &bad_crc))
    {
      if ((frame.flags & WALKER_FLAG_RELAX) != 0U) { Walker_Relax("host request"); return; }
      latest = frame;   /* keep only the newest frame so a slow loop pass never builds up latency */
      have = 1U;
    }
  }

  const uint32_t now = HAL_GetTick();
  if (have != 0U)
  {
    last_valid_ms = now;
    if (mode == WALKER_ARMED) { mode = WALKER_RUN; Print(dry_run ? "WALKER,RUN,dry\r\n" : "WALKER,RUN\r\n"); }
    Apply(&latest);
    if (mode != WALKER_RUN) return;
  }
  if ((mode == WALKER_ARMED) && ((now - armed_ms) > ARMED_TIMEOUT_MS))
  {
    mode = WALKER_CONSOLE;
    Print("WALKER,DISARM,timeout waiting for the first frame\r\ncal> ");
  }
  else if ((mode == WALKER_RUN) && ((now - last_valid_ms) > WATCHDOG_MS))
  {
    Walker_Hold("watchdog");        /* no frames for 100 ms: stay stiff where we are; release only on request */
  }
}

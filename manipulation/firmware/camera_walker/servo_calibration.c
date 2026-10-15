#include "servo_calibration.h"
#include "walker_limits.h"
#include "walker_link.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PCA_RIGHT_7BIT       0x40U
#define PCA_LEFT_7BIT        0x41U
#define PCA_MODE1            0x00U
#define PCA_MODE2            0x01U
#define PCA_LED0_ON_L        0x06U
#define PCA_ALL_LED_ON_L     0xFAU
#define PCA_PRESCALE         0xFEU
#define PCA_FREQUENCY_HZ     50U

#define DEFAULT_SAFE_MIN_US  1200U
#define DEFAULT_SAFE_MAX_US  1800U
#define DEFAULT_STEP_US      10U
#define LINE_LENGTH          128U
#define DIRECTION_LENGTH     52U

typedef struct
{
  uint8_t address;
  uint8_t channel;
  const char *joint;
  uint16_t safe_min_us;
  uint16_t safe_max_us;
  uint16_t neutral_us;
  uint16_t tested_min_us;
  uint16_t tested_max_us;
  uint16_t last_command_us;
  uint8_t has_command;
  char direction[DIRECTION_LENGTH];
} JointCalibration;

static I2C_HandleTypeDef *cal_i2c;
static UART_HandleTypeDef *cal_uart;
static char input_line[LINE_LENGTH];
static uint16_t input_length;
static uint16_t increment_us = DEFAULT_STEP_US;
static int selected_joint = -1;

static JointCalibration joints[19] =
{
  {0x40, 0, "right_gripper",     1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 1, "right_elbow",       1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 2, "right_arm",         1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 3, "right_shoulder",    1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 4, "right_hip_torso",   1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 5, "right_upper_thigh", 1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 6, "right_lower_thigh", 1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 7, "right_knee",        1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 8, "right_foot",        1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x40, 9, "head",              1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 0, "left_gripper",      1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 1, "left_elbow",        1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 2, "left_arm",          1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 3, "left_shoulder",     1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 4, "left_hip_torso",    1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 5, "left_upper_thigh",  1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 6, "left_lower_thigh",  1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 7, "left_knee",         1200, 1800, 0, 0, 0, 0, 0, ""},
  {0x41, 8, "left_foot",         1200, 1800, 0, 0, 0, 0, 0, ""}
};

static void Print(const char *text)
{
  (void)HAL_UART_Transmit(cal_uart, (const uint8_t *)text,
                          (uint16_t)strlen(text), 1000U);
}

static void Printf(const char *format, ...)
{
  char buffer[256];
  va_list args;
  va_start(args, format);
  (void)vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  Print(buffer);
}

static HAL_StatusTypeDef WriteRegister(uint8_t address, uint8_t reg,
                                       uint8_t *data, uint16_t length)
{
  return HAL_I2C_Mem_Write(cal_i2c, (uint16_t)(address << 1), reg,
                           I2C_MEMADD_SIZE_8BIT, data, length, 100U);
}

static HAL_StatusTypeDef DisableBoard(uint8_t address)
{
  uint8_t data[4] = {0U, 0U, 0U, 0x10U};
  return WriteRegister(address, PCA_ALL_LED_ON_L, data, sizeof(data));
}

static HAL_StatusTypeDef DisableChannel(uint8_t address, uint8_t channel)
{
  uint8_t data[4] = {0U, 0U, 0U, 0x10U};
  return WriteRegister(address, (uint8_t)(PCA_LED0_ON_L + 4U * channel),
                       data, sizeof(data));
}

static HAL_StatusTypeDef InitBoard(uint8_t address)
{
  uint8_t old_mode;
  uint8_t value;

  if (HAL_I2C_IsDeviceReady(cal_i2c, (uint16_t)(address << 1), 3U, 100U) != HAL_OK)
    return HAL_ERROR;
  if (HAL_I2C_Mem_Read(cal_i2c, (uint16_t)(address << 1), PCA_MODE1,
                       I2C_MEMADD_SIZE_8BIT, &old_mode, 1U, 100U) != HAL_OK)
    return HAL_ERROR;

  value = (uint8_t)((old_mode & 0x7FU) | 0x10U);
  if (WriteRegister(address, PCA_MODE1, &value, 1U) != HAL_OK) return HAL_ERROR;
  value = 121U;
  if (WriteRegister(address, PCA_PRESCALE, &value, 1U) != HAL_OK) return HAL_ERROR;
  value = (uint8_t)((old_mode & 0x6FU) | 0x21U);
  if (WriteRegister(address, PCA_MODE1, &value, 1U) != HAL_OK) return HAL_ERROR;
  HAL_Delay(5U);
  value |= 0x80U;
  if (WriteRegister(address, PCA_MODE1, &value, 1U) != HAL_OK) return HAL_ERROR;
  value = 0x04U;
  if (WriteRegister(address, PCA_MODE2, &value, 1U) != HAL_OK) return HAL_ERROR;
  return DisableBoard(address);
}

static HAL_StatusTypeDef CommandPulse(JointCalibration *joint, uint16_t pulse_us)
{
  uint16_t count;
  uint8_t data[4];

  if ((pulse_us < joint->safe_min_us) || (pulse_us > joint->safe_max_us))
  {
    Printf("REJECTED: %u us is outside configured safe limits %u..%u us.\r\n",
           pulse_us, joint->safe_min_us, joint->safe_max_us);
    return HAL_ERROR;
  }

  count = (uint16_t)(((uint32_t)pulse_us * 4096U * PCA_FREQUENCY_HZ + 500000U)
                     / 1000000U);
  data[0] = 0U;
  data[1] = 0U;
  data[2] = (uint8_t)(count & 0xFFU);
  data[3] = (uint8_t)((count >> 8) & 0x0FU);
  if (WriteRegister(joint->address,
                    (uint8_t)(PCA_LED0_ON_L + 4U * joint->channel),
                    data, sizeof(data)) != HAL_OK)
  {
    Print("ERROR: PCA9685 did not accept the command.\r\n");
    return HAL_ERROR;
  }

  joint->last_command_us = pulse_us;
  joint->has_command = 1U;
  /* Structured, timestamped setpoint record; this is not measured joint feedback. */
  Printf("CMD,%lu,0x%02X,%u,%u\r\n", (unsigned long)HAL_GetTick(),
         joint->address, joint->channel, pulse_us);
  Printf("COMMAND: board=0x%02X channel=%u joint=%s pulse=%u us (not a measured angle)\r\n",
         joint->address, joint->channel, joint->joint, pulse_us);
  return HAL_OK;
}

static void MoveSlowly(JointCalibration *joint, uint16_t target_us,
                       uint16_t step_us, uint16_t delay_ms)
{
  int32_t current;
  int32_t target = target_us;

  if (!joint->has_command)
  {
    Print("REJECTED: first establish a safe starting point with 'pulse N'.\r\n");
    return;
  }
  if ((target_us < joint->safe_min_us) || (target_us > joint->safe_max_us))
  {
    Printf("REJECTED: target %u us is outside configured safe limits %u..%u us.\r\n",
           target_us, joint->safe_min_us, joint->safe_max_us);
    return;
  }
  if ((step_us < 1U) || (step_us > 50U))
  {
    Print("REJECTED: ramp step must be 1..50 us.\r\n");
    return;
  }
  if ((delay_ms < 10U) || (delay_ms > 1000U))
  {
    Print("REJECTED: ramp delay must be 10..1000 ms.\r\n");
    return;
  }

  current = joint->last_command_us;
  Printf("RAMP START: %ld -> %u us, step=%u us, delay=%u ms. Support the robot.\r\n",
         (long)current, target_us, step_us, delay_ms);

  while (current != target)
  {
    if (current < target)
    {
      current += step_us;
      if (current > target) current = target;
    }
    else
    {
      current -= step_us;
      if (current < target) current = target;
    }

    if (CommandPulse(joint, (uint16_t)current) != HAL_OK)
    {
      Print("RAMP ABORTED because a PWM command failed.\r\n");
      return;
    }
    HAL_Delay(delay_ms);
  }
  Printf("RAMP COMPLETE: commanded target=%u us (not a measured angle).\r\n",
         target_us);
}

static int FindJoint(unsigned long address, unsigned long channel)
{
  for (unsigned int i = 0U; i < 19U; i++)
    if ((joints[i].address == address) && (joints[i].channel == channel)) return (int)i;
  return -1;
}

static void PrintSelected(void)
{
  if (selected_joint < 0)
  {
    Print("No joint selected. Use: select 40 3\r\n");
    return;
  }
  JointCalibration *j = &joints[selected_joint];
  Printf("SELECTED: board=0x%02X channel=%u joint=%s last_command=",
         j->address, j->channel, j->joint);
  if (j->has_command) Printf("%u us", j->last_command_us);
  else Print("NONE (PWM remains disabled)");
  Printf(" safe_limits=%u..%u us step=%u us\r\n",
         j->safe_min_us, j->safe_max_us, increment_us);
}

static void PrintHelp(void)
{
  Print("\r\nCommands:\r\n"
        "  select 1 3         select PCA #1 (0x40), channel 3; does NOT move it\r\n"
        "  select 2 7         select PCA #2 (0x41), channel 7; does NOT move it\r\n"
        "  pulse 1500         explicit pulse command in microseconds\r\n"
        "  move 1700 2 50     ramp to 1700 us: 2-us steps every 50 ms\r\n"
        "  +                   increase last command by current step\r\n"
        "  -                   decrease last command by current step\r\n"
        "  step 10             set increment in microseconds\r\n"
        "  limits 1200 1800    configure conservative command limits\r\n"
        "  neutral             record the last commanded pulse as neutral\r\n"
        "  testedmin           record last command as tested minimum\r\n"
        "  testedmax           record last command as tested maximum\r\n"
        "  direction TEXT      record movement when pulse increases\r\n"
        "  status              print selected joint and last command\r\n"
        "  off                 disable selected channel PWM\r\n"
        "  off all             disable all PWM on both boards\r\n"
        "  export csv          print all calibration records as CSV\r\n"
        "  list                list the 19 mapped joints\r\n"
        "  stand [legs|arms|grip] [STEP_US DELAY_MS]  used joints to neutral one at a time; ramp (default 30 us/25 ms; 'stand 25 30' is slower) where the start is known, else a jump\r\n"
        "  imu on | imu off    start / stop the 50 Hz IMU text rows (off at boot)\r\n"
        "  run dry             laptop link test: frames checked, NO servo output (safe, servo power on or off)\r\n"
        "  run confirm         LIVE: hand servo control to the laptop (needs calibrated limits; robot supported!)\r\n"
        "  help                show these commands\r\n"
        "WARNING: 'off' removes holding torque; support the robot first.\r\n");
}

static void ExportCsv(void)
{
  Print("CSV_BEGIN\r\nboard,channel,joint,safe_min_us,safe_max_us,neutral_us,tested_min_us,tested_max_us,direction\r\n");
  for (unsigned int i = 0U; i < 19U; i++)
  {
    JointCalibration *j = &joints[i];
    Printf("0x%02X,%u,%s,%u,%u,%u,%u,%u,\"%s\"\r\n",
           j->address, j->channel, j->joint, j->safe_min_us,
           j->safe_max_us, j->neutral_us, j->tested_min_us,
           j->tested_max_us, j->direction);
  }
  Print("CSV_END\r\nSave the text between CSV_BEGIN and CSV_END as servo_calibration.csv.\r\n");
}

/* Speed `stand` uses where it knows the starting pulse (the user's preferred ramp: 30 us every 25 ms = 1200 us/s ~ 145 deg/s). */
#define STAND_STEP_US   30U
#define STAND_DELAY_MS  25U
#define STAND_GAP_MS    400U   /* pause after each servo reaches neutral, before the next one starts */

/* Bring every USED joint to its neutral pulse, one at a time, so the supply never sees all servos start together. */
static void Stand(const char *group, uint16_t step_us, uint16_t delay_ms)
{
  static const uint8_t legs[]    = {18, 8, 17, 7, 16, 6, 15, 5, 14, 4};          /* feet first, then up */
  static const uint8_t arms[]    = {13, 3, 11, 1, 12, 2};                         /* shoulders, elbows, arms (incl. left ch2) */
  static const uint8_t grippers[] = {10, 0};
  const uint8_t *lists[3] = {legs, arms, grippers};
  const unsigned sizes[3] = {sizeof(legs), sizeof(arms), sizeof(grippers)};
  unsigned first = 0U, last = 3U;
  if ((group != NULL) && (strcmp(group, "legs") == 0)) last = 1U;
  else if ((group != NULL) && (strcmp(group, "arms") == 0)) { first = 1U; last = 2U; }
  else if ((group != NULL) && ((strcmp(group, "grip") == 0) || (strcmp(group, "grippers") == 0))) first = 2U;
  else if (group != NULL) { Print("Use: stand | stand legs | stand arms | stand grip\r\n"); return; }
  Print("STAND: servo supply must be ON; robot supported/hanging with the limbs near neutral.\r\n"
        "Each used joint jumps to its neutral pulse (400 ms apart). 'off all' releases everything.\r\n");
  for (unsigned g = first; g < last; g++)
  {
    for (unsigned int i = 0U; i < sizes[g]; i++)
    {
      JointCalibration *j = &joints[lists[g][i]];
      if (j->neutral_us == 0U) continue;                       /* unused joint: stays limp */
      if (j->has_command)                                      /* known start: ramp at the gentle speed */
      {
        MoveSlowly(j, j->neutral_us, step_us, delay_ms);
        if (j->last_command_us != j->neutral_us) { Print("STAND aborted.\r\n"); return; }
      }
      else                                                     /* unknown start: the first pulse is a jump at servo speed */
      {
        if (CommandPulse(j, j->neutral_us) != HAL_OK) { Print("STAND aborted.\r\n"); return; }
      }
      HAL_Delay(STAND_GAP_MS);
    }
  }
  Print("STAND done. Next: 'run confirm' hands over to the laptop (all groups must have been stood first).\r\n");
}

static void Execute(char *line)
{
  char *command;
  char *arg1;
  char *arg2;
  JointCalibration *j = (selected_joint >= 0) ? &joints[selected_joint] : NULL;

  command = strtok(line, " ");
  if (command == NULL) return;
  for (char *p = command; *p != '\0'; p++) *p = (char)tolower((unsigned char)*p);

  if (strcmp(command, "help") == 0) PrintHelp();
  else if (strcmp(command, "status") == 0) PrintSelected();
  else if (strcmp(command, "list") == 0)
  {
    for (unsigned int i = 0U; i < 19U; i++)
      Printf("0x%02X ch%u %s\r\n", joints[i].address, joints[i].channel, joints[i].joint);
  }
  else if (strcmp(command, "select") == 0)
  {
    arg1 = strtok(NULL, " "); arg2 = strtok(NULL, " ");
    if ((arg1 == NULL) || (arg2 == NULL)) Print("Usage: select 1 3\r\n");
    else
    {
      unsigned long address;
      if (strcmp(arg1, "1") == 0) address = PCA_RIGHT_7BIT;
      else if (strcmp(arg1, "2") == 0) address = PCA_LEFT_7BIT;
      else address = strtoul(arg1, NULL, 16);
      unsigned long channel = strtoul(arg2, NULL, 10);
      int found = FindJoint(address, channel);
      if (found < 0) Print("Invalid mapped board/channel. Use 'list'.\r\n");
      else
      {
        selected_joint = found;
        PrintSelected();
        Print("Selection caused NO movement. Type an explicit 'pulse N' first.\r\n");
      }
    }
  }
  else if (strcmp(command, "pulse") == 0)
  {
    arg1 = strtok(NULL, " ");
    if ((j == NULL) || (arg1 == NULL)) Print("Select a joint, then use: pulse 1500\r\n");
    else (void)CommandPulse(j, (uint16_t)strtoul(arg1, NULL, 10));
  }
  else if (strcmp(command, "move") == 0)
  {
    char *arg3;
    arg1 = strtok(NULL, " ");
    arg2 = strtok(NULL, " ");
    arg3 = strtok(NULL, " ");
    if ((j == NULL) || (arg1 == NULL) || (arg2 == NULL) || (arg3 == NULL))
      Print("Usage after selection and initial pulse: move TARGET_US STEP_US DELAY_MS\r\n");
    else
      MoveSlowly(j,
                 (uint16_t)strtoul(arg1, NULL, 10),
                 (uint16_t)strtoul(arg2, NULL, 10),
                 (uint16_t)strtoul(arg3, NULL, 10));
  }
  else if ((strcmp(command, "+") == 0) || (strcmp(command, "-") == 0))
  {
    if ((j == NULL) || !j->has_command)
      Print("REJECTED: first send an explicit 'pulse N' command.\r\n");
    else
    {
      int32_t next = j->last_command_us;
      next += (command[0] == '+') ? increment_us : -(int32_t)increment_us;
      if ((next < 0) || (next > 65535)) Print("REJECTED: invalid resulting pulse.\r\n");
      else (void)CommandPulse(j, (uint16_t)next);
    }
  }
  else if (strcmp(command, "step") == 0)
  {
    arg1 = strtok(NULL, " ");
    unsigned long value = arg1 ? strtoul(arg1, NULL, 10) : 0U;
    if ((value < 1U) || (value > 100U)) Print("Step must be 1..100 us.\r\n");
    else { increment_us = (uint16_t)value; Printf("Step=%u us.\r\n", increment_us); }
  }
  else if (strcmp(command, "limits") == 0)
  {
    arg1 = strtok(NULL, " "); arg2 = strtok(NULL, " ");
    unsigned long low = arg1 ? strtoul(arg1, NULL, 10) : 0U;
    unsigned long high = arg2 ? strtoul(arg2, NULL, 10) : 0U;
    if (j == NULL) Print("Select a joint first.\r\n");
    else if ((low < 500U) || (high > 2500U) || (low >= high))
      Print("Limits rejected. Required: 500 <= minimum < maximum <= 2500 us.\r\n");
    else { j->safe_min_us = low; j->safe_max_us = high; PrintSelected(); }
  }
  else if ((strcmp(command, "neutral") == 0) ||
           (strcmp(command, "testedmin") == 0) ||
           (strcmp(command, "testedmax") == 0))
  {
    if ((j == NULL) || !j->has_command) Print("Nothing to record: command a tested pulse first.\r\n");
    else
    {
      if (strcmp(command, "neutral") == 0) j->neutral_us = j->last_command_us;
      else if (strcmp(command, "testedmin") == 0) j->tested_min_us = j->last_command_us;
      else j->tested_max_us = j->last_command_us;
      Printf("Recorded %s=%u us for %s.\r\n", command, j->last_command_us, j->joint);
    }
  }
  else if (strcmp(command, "direction") == 0)
  {
    char *description = strtok(NULL, "");
    if ((j == NULL) || (description == NULL)) Print("Usage: direction knee bends\r\n");
    else
    {
      (void)strncpy(j->direction, description, sizeof(j->direction) - 1U);
      j->direction[sizeof(j->direction) - 1U] = '\0';
      Printf("Recorded increasing-pulse movement: %s\r\n", j->direction);
    }
  }
  else if (strcmp(command, "off") == 0)
  {
    arg1 = strtok(NULL, " ");
    Print("WARNING: disabling PWM releases holding torque; support the robot to prevent collapse.\r\n");
    if ((arg1 != NULL) && (strcmp(arg1, "all") == 0))
    {
      (void)DisableBoard(PCA_RIGHT_7BIT); (void)DisableBoard(PCA_LEFT_7BIT);
      for (unsigned int i = 0U; i < 19U; i++) joints[i].has_command = 0U;
      Print("COMMAND: all PCA outputs disabled.\r\n");
    }
    else if (j == NULL) Print("No selected joint. Use 'off all' or select a joint.\r\n");
    else
    {
      (void)DisableChannel(j->address, j->channel); j->has_command = 0U;
      Printf("COMMAND: PWM disabled for 0x%02X channel %u %s.\r\n",
             j->address, j->channel, j->joint);
    }
  }
  else if (strcmp(command, "export") == 0)
  {
    arg1 = strtok(NULL, " ");
    if ((arg1 != NULL) && (strcmp(arg1, "csv") == 0)) ExportCsv();
    else Print("Use: export csv\r\n");
  }
  else if (strcmp(command, "stand") == 0)
  {
    /* stand [legs|arms|grip] [STEP_US DELAY_MS]  - e.g. "stand legs 25 30" = 25 us every 30 ms (~830 us/s) */
    char *t1 = strtok(NULL, " ");
    char *t2 = strtok(NULL, " ");
    char *t3 = strtok(NULL, " ");
    const char *grp = NULL;
    uint16_t st = STAND_STEP_US, dl = STAND_DELAY_MS;
    if ((t1 != NULL) && isalpha((unsigned char)t1[0])) { grp = t1; t1 = t2; t2 = t3; }
    if (t1 != NULL) st = (uint16_t)strtoul(t1, NULL, 10);
    if (t2 != NULL) dl = (uint16_t)strtoul(t2, NULL, 10);
    Stand(grp, st, dl);
  }
  else if (strcmp(command, "imu") == 0)
  {
    arg1 = strtok(NULL, " ");
    if ((arg1 != NULL) && (strcmp(arg1, "on") == 0)) { ImuStream_Set(1U); Print("IMU stream ON (type 'imu off' to stop)\r\n"); }
    else if ((arg1 != NULL) && (strcmp(arg1, "off") == 0)) { ImuStream_Set(0U); Print("IMU stream OFF\r\n"); }
    else Print("Use: imu on | imu off\r\n");
  }
  else if (strcmp(command, "run") == 0)
  {
    arg1 = strtok(NULL, " ");
    if ((arg1 != NULL) && (strcmp(arg1, "dry") == 0)) (void)Walker_Arm(true);
    else if ((arg1 != NULL) && (strcmp(arg1, "confirm") == 0)) (void)Walker_Arm(false);
    else Print("Use: run dry  (link test, no servo output)  or  run confirm  (LIVE, robot supported)\r\n");
  }
  else Print("Unknown command. Type 'help'.\r\n");
}

void ServoCal_Init(I2C_HandleTypeDef *i2c, UART_HandleTypeDef *uart)
{
  cal_i2c = i2c;
  cal_uart = uart;
  input_length = 0U;
#if WALKER_LIMITS_VALID
  for (unsigned int i = 0U; i < 19U; i++)
  {
    joints[i].safe_min_us = walker_min_us[i];
    joints[i].safe_max_us = walker_max_us[i];
    joints[i].neutral_us = walker_neutral_us[i];
  }
#endif

  Print("\r\n19-DOF SERVO CALIBRATION CONSOLE\r\n");
  Print("No servo pulse is commanded at startup. Initializing both boards with all outputs OFF.\r\n");
  HAL_StatusTypeDef right = InitBoard(PCA_RIGHT_7BIT);
  HAL_StatusTypeDef left = InitBoard(PCA_LEFT_7BIT);
  Printf("PCA 0x40 right: %s | PCA 0x41 left: %s\r\n",
         right == HAL_OK ? "READY/OFF" : "ERROR",
         left == HAL_OK ? "READY/OFF" : "ERROR");
  Print("Support the robot securely before enabling servo power or sending a pulse.\r\n");
  Print("Commands are requested positions only; MG996R servos provide no measured angle feedback.\r\n");
  PrintHelp();
  Print("cal> ");
}

void ServoCal_Process(void)
{
  uint8_t character;
  if (Walker_Mode() != WALKER_CONSOLE) return;   /* walker mode owns the serial port */
  if (!Walker_PopByte(&character)) return;       /* bytes are buffered by the USART1 interrupt */

  if ((character == '\r') || (character == '\n'))
  {
    if (input_length != 0U)
    {
      input_line[input_length] = '\0';
      Print("\r\n");
      Execute(input_line);
      input_length = 0U;
      Print("cal> ");
    }
  }
  else if ((character == 0x08U) || (character == 0x7FU))
  {
    if (input_length != 0U) { input_length--; Print("\b \b"); }
  }
  else if ((character >= 32U) && (character <= 126U) &&
           (input_length < (LINE_LENGTH - 1U)))
  {
    input_line[input_length++] = (char)character;
    (void)HAL_UART_Transmit(cal_uart, &character, 1U, 100U);
  }
}

bool ServoCal_GetLimits(unsigned index, uint8_t *address, uint8_t *channel,
                        uint16_t *min_us, uint16_t *max_us)
{
  if (index >= 19U) return false;
  *address = joints[index].address;
  *channel = joints[index].channel;
  *min_us = joints[index].safe_min_us;
  *max_us = joints[index].safe_max_us;
  return true;
}

void ServoCal_NoteCommand(unsigned index, uint16_t pulse_us)
{
  if (index >= 19U) return;
  joints[index].last_command_us = pulse_us;
  joints[index].has_command = 1U;
}

bool ServoCal_JointCommanded(unsigned index, uint16_t *last_us)
{
  if ((index >= 19U) || (joints[index].has_command == 0U)) return false;
  *last_us = joints[index].last_command_us;
  return true;
}

bool ServoCal_JointUnused(unsigned index)
{
  return (index >= 19U) || (joints[index].neutral_us == 0U);
}

void ServoCal_RelaxAll(void)
{
  (void)DisableBoard(PCA_RIGHT_7BIT);
  (void)DisableBoard(PCA_LEFT_7BIT);
  for (unsigned int i = 0U; i < 19U; i++) joints[i].has_command = 0U;
}

# camera_walker — ST camera-capture (UVC) + laptop-controlled servo "walker" mode, one firmware

A COPY of `C:\robot\x-cube-n6-camera-capture-main` with the walker modules from `C:\robot\stm32_walker\FSBL` added.
Both originals are untouched. **Built and linked (arm-none-eabi-gcc 14.3); NOT yet flashed or run on hardware.**

What runs on the STM32N6570-DK:
* the unchanged ST camera pipeline (DCMIPP + ISP + USBX UVC) on the board's user USB port, and
* a FreeRTOS task `walker` (priority idle+2, `Src/app_walker.c`): I2C1 (PC1 SDA / PH9 SCL, ~100 kHz computed from the real clock) -> PCA9685 0x40 / 0x41,
  the servo calibration console + `run confirm` walker mode on USART1 (ST-LINK virtual COM, 115200), 44-byte CRC frames from the laptop,
  clamps to `Inc/walker_limits.h`, 100 ms watchdog -> relax.  Safety logic is the walker project's, unchanged.

Differences from `stm32_walker`: **no IMU stream** (no ICM-20948 code; `ImuStream_Set` is a stub), so `robot_host.py` walking (needs IMU rows) is NOT supported by this build —
it is for camera + arm demos. Camera sensor is on I2C2, servos on I2C1: no bus clash. USART1 is shared with the camera app's boot prints.

## Build / flash
* `build_firmware.bat` (uses CubeIDE's gcc + make) -> `build\Project.bin`
* `sign_and_flash.bat`: signs (`STM32_SigningTool_CLI ... --align`) and programs external flash at 0x70000000. Board in dev mode, then boot-from-flash + power-cycle (see README.md "Boot modes").
* Go back: program `Binary\STM32N6570-DK\x-cube-n6-camera-capture-front-dk.hex` (camera only) or the stm32_walker FSBL/Appli (servos + IMU only).

## Before the servos may move
1. `Inc\walker_limits.h` is the walker project's walking limits (arm ranges tiny). Set real right-arm limits (console `limits MIN MAX`, `export csv`, `tools\csv_to_limits.py`), copy the new header here, rebuild, flash.
2. Servo supply OFF while flashing/first boot. Open COM3 @115200: expect `READY/OFF` on both boards. Then supply ON, `stand`, `run confirm`, close the terminal, run the laptop program.

# camera_walker: ST camera-capture (UVC) + laptop-controlled servo mode, one firmware

ST's `x-cube-n6-camera-capture` app streams the board camera over USB (UVC). This folder adds the servo link from the locomotion/walker project as one FreeRTOS task, so a single image on the
STM32N6570-DK **streams video and accepts servo frames** (44-byte CRC frames on the ST-LINK serial port, 115200) that drive two PCA9685 boards. Only my files and small patches are here; ST's code is not redistributed.

**Status:** builds, signs, flashes; the laptop-to-board link test passes against it (5 of 5, 100 frames, 0 bad CRC, dry mode). **Not yet used to move the arm.** No IMU stream in this build (so `deploy/robot_host.py` walking needs the original walker firmware).

## What is here

| File | What |
|---|---|
| `app_walker.c/.h` | the `walker` task: I2C1 (PC1 SDA, PH9 SCL; the camera sensor is on I2C2, no clash) timing computed from the real clock, then `ServoCal_Init`, `Walker_Init`, and `Walker_Process` every ms |
| `walker_frame.c/.h`, `walker_link.c/.h`, `servo_calibration.c/.h`, `walker_limits.h` | the walker modules (frame parser + CRC-16, USART1 interrupt ring buffer, modes CONSOLE / ARMED / RUN, 100 ms watchdog, clamp to limits, calibration console, PCA9685 bursts) |
| `app.c.patch`, `Makefile.patch`, `cubeide_project.patch` | three small patches to ST's files (start the task; add the sources to the Makefile and to the CubeIDE `.project`) |
| `servo_calibration_sim.json` | pulse/angle table used by `armcopy/robot.py` |
| `build_firmware.bat`, `sign_and_flash.bat` | build with CubeIDE's toolchain; sign and program external flash at 0x70000000 |

## Reproduce

1. Get ST's package: <https://github.com/STMicroelectronics/x-cube-n6-camera-capture>, and work in a **copy** (`PKG`).
2. Copy `app_walker.c walker_frame.c walker_link.c servo_calibration.c` to `PKG/Src/` and the five headers (`app_walker.h walker_frame.h walker_link.h servo_calibration.h walker_limits.h`) to `PKG/Inc/`.
3. From `PKG`: `patch -p1 < app.c.patch`, `patch -p1 < Makefile.patch`, `patch -p1 < cubeide_project.patch` (verified to apply to the files as ST ships them).
4. `build_firmware.bat PKG`, then `sign_and_flash.bat PKG` (board in development mode, servo power OFF), then boot from flash and power-cycle.
5. Check: the board shows up as a camera, and the serial port at 115200 shows a `cal>` prompt. `run dry` then the dry link test (`deploy`'s serial link, `armcopy/robot.py`) confirms the frames.

## Before the servos may move

`walker_limits.h` carries the *walking* limits, which are tiny for the arm (about 0.3 rad on the arm joint, one-sided on the elbow), and the right gripper is marked unused. Set real limits in the console
(`select`, slow `move`, `testedmin` / `testedmax`, `limits`, `neutral`, `export csv`), regenerate `walker_limits.h` from the CSV, rebuild and reflash; console limits alone live in RAM and vanish at reset.
Servo supply OFF while flashing; supply ON only after the board boots; robot hanging; then `stand`, `run confirm`, close the terminal and run `robot/arm_jog.py`. After any relax or watchdog hold, repeat `stand` and `run confirm`.

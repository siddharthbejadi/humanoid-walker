@echo off
rem Signs the merged firmware and programs it into the board's external flash at 0x70000000 (ST's boot-from-flash flow for this app).
rem 1) Board in DEVELOPMENT mode (boot switches per the board-mode picture in README.md), ST-LINK cable connected, servo power OFF.
rem 2) Run this file.  3) Set the board to BOOT FROM FLASH and power-cycle it.
set CP=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin
cd /d C:\robot\camera_walker
"%CP%\STM32_SigningTool_CLI.exe" -bin build\Project.bin -nk -t ssbl -hv 2.3 --align -o build\Project_sign.bin || exit /b 1
"%CP%\STM32_Programmer_CLI.exe" -c port=SWD mode=HOTPLUG -el "%CP%\ExternalLoader\MX66UW1G45G_STM32N6570-DK.stldr" -hardRst -w build\Project_sign.bin 0x70000000 -v
pause

@echo off
rem usage: sign_and_flash.bat PACKAGE_DIR      (after build_firmware.bat)
rem Signs build\Project.bin and programs it into the board's external flash at 0x70000000 (ST's boot-from-flash flow for this app).
rem 1) Board in DEVELOPMENT mode (see the boot-mode pictures in ST's README), ST-LINK cable connected, servo power OFF.
rem 2) Run this.   3) Set the board to BOOT FROM FLASH and power-cycle it.
if "%~1"=="" (echo usage: sign_and_flash.bat PACKAGE_DIR & exit /b 1)
if "%CUBEIDE%"=="" set CUBEIDE=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE
for /d %%d in ("%CUBEIDE%\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.*") do set CP=%%d\tools\bin
if not defined CP (echo STM32CubeProgrammer not found under %CUBEIDE% & exit /b 1)
"%CP%\STM32_SigningTool_CLI.exe" -bin "%~1\build\Project.bin" -nk -t ssbl -hv 2.3 --align -o "%~1\build\Project_sign.bin" || exit /b 1
"%CP%\STM32_Programmer_CLI.exe" -c port=SWD mode=HOTPLUG -el "%CP%\ExternalLoader\MX66UW1G45G_STM32N6570-DK.stldr" -hardRst -w "%~1\build\Project_sign.bin" 0x70000000 -v

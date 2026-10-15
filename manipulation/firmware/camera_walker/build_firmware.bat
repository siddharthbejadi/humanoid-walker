@echo off
rem usage: build_firmware.bat PACKAGE_DIR
rem PACKAGE_DIR = a copy of ST's x-cube-n6-camera-capture with this folder's files applied (see README_CAMERA_WALKER.md).
rem CUBEIDE = your STM32CubeIDE install folder; it provides arm-none-eabi-gcc and make. Override it with: set CUBEIDE=D:\path\to\STM32CubeIDE
if "%~1"=="" (echo usage: build_firmware.bat PACKAGE_DIR & exit /b 1)
if "%CUBEIDE%"=="" set CUBEIDE=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE
for /d %%d in ("%CUBEIDE%\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*") do set GCC=%%d\tools\bin
for /d %%d in ("%CUBEIDE%\plugins\com.st.stm32cube.ide.mcu.externaltools.make.*") do set MK=%%d\tools\bin
if not defined GCC (echo arm-none-eabi-gcc not found under %CUBEIDE% & exit /b 1)
set PATH=%GCC%;%MK%;%PATH%
cd /d "%~1" && make -j8

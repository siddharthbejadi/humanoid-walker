@echo off
set GCC=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740\tools\bin
set MK=C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615\tools\bin
set PATH=%GCC%;%MK%;%PATH%
cd /d C:\robot\camera_walker
make -j8 > C:\robot\camera_walker\build.log 2>&1
echo exit=%errorlevel% >> C:\robot\camera_walker\build.log

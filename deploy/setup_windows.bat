@echo off
rem One-time setup of the robot laptop (Windows): Python environment + the policy source. Re-runnable.
rem Tested package list: torch 2.8.0 (CPU), numpy, gymnasium, omegaconf, tensordict 0.9.1, tensorboard, pyserial.
setlocal
cd /d %~dp0
set PYV=
py -3.11 --version >nul 2>&1 && set PYV=-3.11
if "%PYV%"=="" py -3 --version >nul 2>&1 && set PYV=-3
if "%PYV%"=="" (echo Install Python 3.11 from python.org first ^(tick "Add to PATH"^), then run this again. & exit /b 1)
if not exist robot_env py %PYV% -m venv robot_env || exit /b 1
robot_env\Scripts\python -m pip install --upgrade pip || exit /b 1
robot_env\Scripts\python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu || exit /b 1
robot_env\Scripts\python -m pip install numpy gymnasium omegaconf "tensordict==0.9.1" tensorboard pyserial || exit /b 1
if not exist source_v6 mkdir source_v6
tar -xzf source_v6.tgz -C source_v6 || exit /b 1
echo.
echo SETUP DONE. Plumbing test (no robot needed, ~30 s):
echo   robot_env\Scripts\python robot_host.py --repo source_v6\r2dreamer --checkpoint student_realistic_12k.pt --selftest
echo List serial ports (find the ST-LINK COM port):
echo   robot_env\Scripts\python -m serial.tools.list_ports

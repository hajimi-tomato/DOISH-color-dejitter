@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :failed
)
if not exist ".venv\doish-ready-v1" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
  echo ready>".venv\doish-ready-v1"
)
".venv\Scripts\python.exe" launcher.py %*
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Startup failed. Use Python 3.10-3.12 and read the error above.
pause
exit /b 1

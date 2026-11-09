@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt "pyinstaller>=6,<7"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" build_windows.py
if errorlevel 1 goto :failed
pause
exit /b 0
:failed
echo Build failed. See the error above.
pause
exit /b 1

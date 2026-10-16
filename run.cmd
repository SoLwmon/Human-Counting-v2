@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment is missing. Run setup.ps1 first.
    exit /b 1
)
set "YOLO_CONFIG_DIR=%~dp0work\ultralytics"
if not exist "%YOLO_CONFIG_DIR%\Ultralytics" mkdir "%YOLO_CONFIG_DIR%\Ultralytics"
".venv\Scripts\python.exe" booth_counter.py %*

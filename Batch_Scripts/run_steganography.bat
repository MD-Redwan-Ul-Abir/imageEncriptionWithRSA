@echo off
REM Steganography Script Runner
REM This batch file helps run the steganography script easily

echo ================================================
echo    STEGANOGRAPHY SCRIPT RUNNER
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Change to Scripts directory
cd /d "%~dp0..\Scripts"

REM Check if script exists
if not exist "precision_fixed_dwt_lsb_steganography.py" (
    echo ERROR: Steganography script not found
    echo Please ensure precision_fixed_dwt_lsb_steganography.py is in the Scripts folder
    pause
    exit /b 1
)

REM Run the script
echo Running steganography script...
echo.
python precision_fixed_dwt_lsb_steganography.py

echo.
echo ================================================
echo Script execution completed.
echo Check the Logs folder for detailed execution logs.
echo ================================================
pause
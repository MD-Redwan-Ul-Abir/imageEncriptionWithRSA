@echo off
REM Setup Environment for Steganography Project
REM This script installs all required dependencies

echo ================================================
echo    STEGANOGRAPHY ENVIRONMENT SETUP
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    echo Make sure to add Python to your PATH during installation
    pause
    exit /b 1
)

echo Python found. Proceeding with package installation...
echo.

REM Change to Scripts directory
cd /d "%~dp0..\Scripts"

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    echo Please ensure requirements.txt is in the Scripts folder
    pause
    exit /b 1
)

echo Installing required packages...
echo This may take a few minutes...
echo.

REM Install packages
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ================================================
echo Environment setup completed successfully!
echo.
echo You can now run the steganography script using:
echo   run_steganography.bat
echo.
echo Or manually with:
echo   python precision_fixed_dwt_lsb_steganography.py
echo ================================================
pause
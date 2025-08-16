@echo off
REM Quick Start Script for Steganography Package
REM This script guides the user through the complete setup and first run

echo ========================================================
echo     STEGANOGRAPHY PACKAGE - QUICK START HELPER
echo ========================================================
echo.
echo This script will help you set up and run the steganography
echo system for the first time.
echo.
echo ========================================================
echo STEP 1: ENVIRONMENT SETUP
echo ========================================================
echo.
echo First, we need to install the required Python packages.
echo This may take a few minutes...
echo.
pause

cd Batch_Scripts
call setup_environment.bat

if errorlevel 1 (
    echo.
    echo ERROR: Environment setup failed!
    echo Please check that Python is installed and try again.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo STEP 2: READY TO RUN
echo ========================================================
echo.
echo Environment setup completed successfully!
echo.
echo Now you can:
echo 1. Put your 512x512 image in the Input_Images folder
echo 2. Run the steganography script
echo.
echo Would you like to run the steganography script now?
echo (There's already a test image you can use)
echo.
set /p answer="Run now? (y/n): "

if /i "%answer%"=="y" (
    echo.
    echo Starting steganography script...
    echo.
    call run_steganography.bat
) else (
    echo.
    echo You can run the steganography script anytime by:
    echo 1. Double-clicking Batch_Scripts\run_steganography.bat
    echo 2. Or running this QUICK_START.bat again
    echo.
)

echo.
echo ========================================================
echo QUICK START COMPLETED
echo ========================================================
echo.
echo Check these folders for results:
echo - Output\Steganographic_Images\ (hidden message images)
echo - Output\Keys\ (encryption keys - keep safe!)
echo - Output\Logs\ (detailed execution logs)
echo.
echo For detailed instructions, see README.txt
echo.
pause
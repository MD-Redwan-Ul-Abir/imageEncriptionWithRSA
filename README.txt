========================================================
    STEGANOGRAPHY PACKAGE - QUICK START GUIDE
========================================================

Hi! This package contains a complete steganography system that can hide
secret messages in images using advanced encryption and logging.

========================================================
🚀 QUICK START (3 SIMPLE STEPS)
========================================================

STEP 1: INSTALL REQUIREMENTS
----------------------------
Double-click: Batch_Scripts\setup_environment.bat
(This will automatically install all needed Python packages)

STEP 2: PREPARE YOUR IMAGE
--------------------------
- Put a 512x512 pixel image in the "Input_Images" folder
- Supported formats: PNG, JPG, BMP
- There's already a test image (test_image.jpg) you can use

STEP 3: RUN THE STEGANOGRAPHY
-----------------------------
Double-click: Batch_Scripts\run_steganography.bat
- Choose option "1" for test mode
- Enter your image filename (e.g., test_image.jpg)
- Enter your secret message
- The program will hide your message and create logs!

========================================================
📁 WHAT YOU GET
========================================================

After running, check these folders:
- Output\Steganographic_Images\ → Image with your hidden message
- Output\Keys\ → Encryption keys (KEEP PRIVATE KEY SAFE!)
- Output\Logs\ → Detailed execution logs for analysis

========================================================
📋 SYSTEM REQUIREMENTS
========================================================

- Windows 10 or newer
- Python 3.7+ (if not installed, get it from python.org)
- Internet connection (for initial package installation)
- About 100MB free disk space

========================================================
🆘 TROUBLESHOOTING
========================================================

Problem: "Python not found"
Solution: Install Python from python.org, make sure to check "Add to PATH"

Problem: "Package installation failed"
Solution: Check internet connection, run as administrator

Problem: "Could not load image"
Solution: Make sure your image is 512x512 pixels and in Input_Images folder

Problem: Still having issues?
Solution: Check Documentation\EXECUTION_GUIDE.txt for detailed help

========================================================
📖 DETAILED DOCUMENTATION
========================================================

For complete instructions, see Documentation folder:
- EXECUTION_GUIDE.txt → Complete step-by-step instructions
- SETUP_AND_FILES.txt → Detailed setup and file organization
- PROJECT_STRUCTURE.txt → How everything is organized
- FINAL_SUMMARY.txt → Complete project overview

========================================================
🔒 SECURITY IMPORTANT
========================================================

⚠️  CRITICAL: The private_key.pem file in Output\Keys\ is needed
    to decrypt your hidden messages. Keep it safe and make backups!

✅ Your messages are encrypted with RSA-2048 before hiding
✅ Even if someone finds the hidden data, it's still encrypted
✅ Log files don't contain your actual message content

========================================================
🎯 EXAMPLE USAGE
========================================================

1. Run setup_environment.bat (first time only)
2. Put your image in Input_Images\
3. Run run_steganography.bat
4. Choose "1"
5. Type: test_image.jpg
6. Type: Hello, this is my secret message!
7. Check Output\ folders for results and logs

========================================================
📞 SUPPORT
========================================================

If you need help:
1. Check the detailed guides in Documentation\
2. Look at the log files in Output\Logs\ for error details
3. Make sure all requirements are installed properly

========================================================

Package Version: 1.0
Created: 2024-08-16
Compatible: Windows 10+, Python 3.7+

Enjoy hiding your secret messages! 🕵️‍♂️

========================================================
@echo off
setlocal
title AEGIS - One-Way Threat Detection System

:: Ensure current working directory is the project root
cd /d "%~dp0"

echo ======================================================================
echo   AEGIS: Label-Free Threat Detection in One-Way Network Traffic
echo   Problem ID: SIH26145 ^| Unsupervised SOC Radar ^& Forensic XAI
echo ======================================================================
echo.

:: 1. Validate virtual environment interpreter
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment interpreter not found at:
    echo         "%VENV_PYTHON%"
    echo.
    echo Please ensure the project virtual environment is created before launching.
    echo Setup commands:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\pip install -r requirements.txt
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: 2. Validate application entrypoint script
set "DEMO_SCRIPT=%~dp0run_demo.py"
if not exist "%DEMO_SCRIPT%" (
    echo [ERROR] Entrypoint script 'run_demo.py' was not found in:
    echo         "%~dp0"
    echo.
    echo Please verify that the AEGIS project repository files are intact.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: 3. Check for GEMINI_API_KEY (support user environment or local .env)
if not defined GEMINI_API_KEY (
    if exist "%~dp0.env" (
        for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
            if /i "%%A"=="GEMINI_API_KEY" (
                set "GEMINI_API_KEY=%%B"
            )
        )
    )
)

if defined GEMINI_API_KEY (
    echo [+] GEMINI_API_KEY detected in environment.
) else (
    echo [!] Notice: GEMINI_API_KEY environment variable is not configured.
    echo     Standard heuristic Forensic XAI will operate with full local attribution.
    echo     (Optional: set GEMINI_API_KEY in user environment or in a local .env file)
)
echo.

:: 4. Check whether port 8501 is already active
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Port 8501 is already in use by an active service.
    echo        The application will connect to the active dashboard on http://localhost:8501
    echo        without launching duplicate server instances.
) else (
    echo [+] Port 8501 is available. Dashboard will launch at http://localhost:8501.
)
echo.

:: 5. Launch the AEGIS demonstration application
echo [*] Starting AEGIS (Dashboard + Optical Tap Stream Replay)...
echo     Press Ctrl+C anytime in this window to stop.
echo ----------------------------------------------------------------------
echo.

"%VENV_PYTHON%" "%DEMO_SCRIPT%" %*
set "APP_EXIT_CODE=%ERRORLEVEL%"

echo.
echo ----------------------------------------------------------------------
if %APP_EXIT_CODE% NEQ 0 (
    echo [!] Application exited with code %APP_EXIT_CODE%.
    echo Press any key to close this window...
    pause >nul
) else (
    echo [+] AEGIS session ended cleanly.
)

exit /b %APP_EXIT_CODE%

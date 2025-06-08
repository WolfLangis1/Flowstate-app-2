@echo off
echo Activating virtual environment from %~dp0 ...
call "%~dp0flowstate_env\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment at "%~dp0flowstate_env\Scripts\activate.bat".
    echo Ensure setup was successful and the venv 'flowstate_env' exists.
    pause
    exit /b 1
)
echo Starting Flow State Music Player...
"%~dp0flowstate_env\Scripts\python.exe" "%~dp0flow_state_launcher.py" %*
echo Flow State Music Player exited.
REM Add a pause if you want the window to stay open after app closes when double-clicked
pause >nul 
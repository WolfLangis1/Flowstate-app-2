

#!/usr/bin/env python3
"""
Flow State Music Player - Setup Script
Automated setup for development and basic user environments.
For broader distribution, consider tools like PyInstaller, cx_Freeze, or Briefcase.
"""

import os
import sys
import subprocess
import platform
import shutil 
from pathlib import Path
import logging
from typing import Tuple # Ensure Tuple is imported for type hints

# Configure logger for setup script specifically
setup_logger = logging.getLogger("FlowStateSetup")
setup_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Setup] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
if not setup_logger.handlers: 
    setup_logger.addHandler(ch)
setup_logger.propagate = False 

class FlowStateSetup:
    def __init__(self):
        self.platform_system = platform.system()
        self.python_version_info = sys.version_info
        self.script_dir = Path(__file__).parent.resolve()
        self.app_name = "Flow State Music Player"
        self.venv_name = "flowstate_env" 
        self.venv_path = self.script_dir / self.venv_name
        self.requirements_file_path = self.script_dir / "requirements.txt"

    def check_python_version(self):
        setup_logger.info(f"Python Version: {sys.version.splitlines()[0]}")
        if self.python_version_info < (3, 8):
            setup_logger.error(f"{self.app_name} requires Python 3.8 or higher. Please upgrade.")
            sys.exit(1)
        setup_logger.info("Python version check: OK.")

    def check_system_dependencies(self):
        setup_logger.info("--- System Dependency Guidance ---")
        if self.platform_system == "Linux":
            setup_logger.info("On Linux, ensure: python3-tk, portaudio19-dev, ffmpeg, libsndfile1, libgl1-mesa-glx, libegl1-mesa")
            setup_logger.info("Example (Debian/Ubuntu): sudo apt-get update && sudo apt-get install -y python3-tk portaudio19-dev ffmpeg libsndfile1 libgl1-mesa-glx libegl1-mesa")
        elif self.platform_system == "Darwin":
            setup_logger.info("On macOS (via Homebrew): brew install python-tk portaudio ffmpeg libsndfile")
        elif self.platform_system == "Windows":
            setup_logger.info("On Windows: Python (includes Tkinter), FFmpeg (add to PATH), up-to-date OpenGL drivers. MSVC C++ Build Tools might be needed for some packages.")
        setup_logger.info("--- End System Dependency Guidance ---")

    def create_or_confirm_virtual_env(self) -> Path:
        setup_logger.info(f"Venv: '{self.venv_path.name}' at {self.script_dir}")
        if self.venv_path.is_dir() and (self.venv_path / "pyvenv.cfg").exists():
            setup_logger.info(f"Venv '{self.venv_path.name}' already exists.")
        else:
            setup_logger.info(f"Creating new venv '{self.venv_path.name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_path)])
                setup_logger.info("Venv created successfully.")
            except subprocess.CalledProcessError as e:
                setup_logger.error(f"Failed to create venv: {e}. Check Python/venv module."); sys.exit(1)
            except Exception as e_gen: 
                setup_logger.error(f"Unexpected error creating venv: {e_gen}", exc_info=True); sys.exit(1)
        return self.venv_path

    def get_venv_paths(self) -> Tuple[Path, Path]:
        if self.platform_system == "Windows":
            python_exe = self.venv_path / "Scripts" / "python.exe"
            pip_exe = self.venv_path / "Scripts" / "pip.exe"
        else: 
            python_exe = self.venv_path / "bin" / "python"
            pip_exe = self.venv_path / "bin" / "pip"
        if not python_exe.exists() or not pip_exe.exists():
            setup_logger.error(f"Python/pip not found in venv: {self.venv_path}. Venv corrupt or not created?"); sys.exit(1)
        return python_exe, pip_exe

    def install_python_dependencies(self) -> bool:
        python_exe_in_venv, pip_exe_in_venv = self.get_venv_paths()
        if not self.requirements_file_path.exists():
            setup_logger.error(f"'{self.requirements_file_path.name}' not found. Cannot install dependencies."); return False
        setup_logger.info(f"Installing dependencies from '{self.requirements_file_path.name}' using '{pip_exe_in_venv}'...")
        try:
            setup_logger.info("Upgrading pip, setuptools, wheel in venv...")
            subprocess.check_call([str(python_exe_in_venv), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            setup_logger.info(f"Installing packages from {self.requirements_file_path.name}...")
            subprocess.check_call([str(pip_exe_in_venv), "install", "-r", str(self.requirements_file_path)])
            setup_logger.info("Python dependencies installed successfully."); return True
        except subprocess.CalledProcessError as e:
            setup_logger.error(f"Failed to install dependencies: {e}. Ensure venv active or system deps (compilers, headers) met."); return False
        except Exception as e_gen: 
            setup_logger.error(f"Unexpected error installing dependencies: {e_gen}", exc_info=True); return False

    def create_run_scripts(self):
        setup_logger.info("Creating run scripts...")
        launcher_rel_path = "flow_state_launcher.py" 
        run_script_base_name = f"run_{self.app_name.lower().replace(' ', '_')}"

        if self.platform_system == "Windows":
            activate_script_rel = Path(self.venv_name) / "Scripts" / "activate.bat"
            python_in_venv_rel = Path(self.venv_name) / "Scripts" / "python.exe"
            script_content = f'''@echo off
echo Activating virtual environment from %~dp0 ...
call "%~dp0{activate_script_rel}"
if errorlevel 1 (
    echo Failed to activate venv at "%~dp0{activate_script_rel}". Ensure setup successful.
    pause
    exit /b 1
)
echo Starting {self.app_name}...
"%~dp0{python_in_venv_rel}" "%~dp0{launcher_rel_path}" %*
echo {self.app_name} exited.
pause
''' # Changed pause >nul to pause for debugging visibility
            script_path = self.script_dir / f"{run_script_base_name}.bat"
        else: 
            activate_script_in_venv_rel = Path(self.venv_name) / "bin" / "activate"
            python_in_venv_rel = Path(self.venv_name) / "bin" / "python"
            script_content = f'''#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${{BASH_SOURCE[0]}}" || readlink "${{BASH_SOURCE[0]}}" || echo "${{BASH_SOURCE[0]}}")")" &>/dev/null && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/{activate_script_in_venv_rel}"
PYTHON_EXEC="$SCRIPT_DIR/{python_in_venv_rel}"
LAUNCHER_SCRIPT="$SCRIPT_DIR/{launcher_rel_path}"

if [ ! -f "$VENV_ACTIVATE" ]; then echo "Error: Venv activate script not found: $VENV_ACTIVATE" >&2; exit 1; fi
if [ ! -x "$PYTHON_EXEC" ]; then echo "Error: Python in venv not found/executable: $PYTHON_EXEC" >&2; exit 1; fi
if [ ! -f "$LAUNCHER_SCRIPT" ]; then echo "Error: Launcher not found: $LAUNCHER_SCRIPT" >&2; exit 1; fi

echo "Activating virtual environment..."
source "$VENV_ACTIVATE"
echo "Starting {self.app_name}..."
"$PYTHON_EXEC" "$LAUNCHER_SCRIPT" "$@" 
EXIT_CODE=$?
if type deactivate &>/dev/null; then deactivate &>/dev/null; fi
echo "{self.app_name} exited with code $EXIT_CODE."
# read -p "Press Enter to close..." # Optional pause
exit $EXIT_CODE
'''
            script_path = self.script_dir / f"{run_script_base_name}.sh"
            
        try:
            with open(script_path, 'w', newline='\n' if self.platform_system != "Windows" else None) as f: f.write(script_content)
            if self.platform_system != "Windows": os.chmod(script_path, 0o755)
            setup_logger.info(f"Created run script: {script_path.name}")
        except IOError as e: setup_logger.error(f"Failed to create run script {script_path.name}: {e}")
        except Exception as e_gen: setup_logger.error(f"Unexpected error creating script {script_path.name}: {e_gen}", exc_info=True)

    def run_full_setup(self):
        setup_logger.info(f"--- {self.app_name} Setup Initiated ---")
        self.check_python_version()
        self.check_system_dependencies()
        try: self.create_or_confirm_virtual_env()
        except SystemExit: return 
        if self.requirements_file_path.exists():
            if input(f"\nInstall dependencies from '{self.requirements_file_path.name}' into '{self.venv_name}'? (y/n): ").strip().lower() == 'y':
                if not self.install_python_dependencies(): setup_logger.warning("Dependency install failed/skipped. App may not run.")
            else: setup_logger.info("Skipping dependency installation.")
        else:
            setup_logger.error(f"'{self.requirements_file_path.name}' not found. Cannot install dependencies.")
        if input("\nCreate run scripts? (y/n): ").strip().lower() == 'y': self.create_run_scripts()
        setup_logger.info("--- Setup Process Finished ---")
        run_script_name = f"run_{self.app_name.lower().replace(' ', '_')}.{'sh' if self.platform_system != 'Windows' else 'bat'}"
        setup_logger.info(f"\nTo run {self.app_name}: cd \"{self.script_dir}\" && ./{run_script_name} (or {run_script_name} on Windows)")
        setup_logger.info("\nManual run: Activate venv (e.g. source flowstate_env/bin/activate) then: python flow_state_launcher.py")
        setup_logger.info(f"\n🎉 Setup complete. Enjoy {self.app_name}!")

if __name__ == "__main__":
    setup_manager = FlowStateSetup()
    setup_manager.run_full_setup()


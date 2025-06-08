

# Flow State - Next Generation Music Player 🎵

Flow State is an advanced, feature-rich, AI-powered music player designed for an immersive and intelligent listening experience. It goes beyond traditional players by integrating advanced audio visualizations, real-time collaboration features, voice control, and smart music discovery, all within an extensible, themable Python application.

## ✨ Core Vision

To create a music experience that is:
- **Immersive:** Through GPU-accelerated audio visualizations (using ModernGL) and synchronized lyrics.
- **Intelligent:** With AI-powered recommendations, automatic music analysis (BPM, key, energy), and intuitive voice control.
- **Connected:** Featuring real-time collaborative listening sessions and mobile remote control capabilities.
- **Customizable:** Offering a user-friendly theming system and an extensible plugin architecture for new audio effects and visualizers.
- **Performant:** Utilizing modern Python techniques, background processing with `asyncio`, `threading`, and `multiprocessing`, and efficient libraries like `sounddevice` for audio I/O.

## 🚀 Getting Started

### Prerequisites:
*   **Python:** Version 3.8 or higher is required.
*   **FFmpeg:** Essential for audio/video export features. Ensure FFmpeg is installed on your system and accessible via your system's PATH environment variable.
*   **System Dependencies:** These vary by operating system. The setup script (`flow_state_setup.py`) provides more detailed guidance. Common needs include:
    *   **Linux:** `python3-tk`, `portaudio19-dev`, `libsndfile1`, and OpenGL libraries (e.g., `libgl1-mesa-glx`, `libegl1-mesa`).
        *   Example (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install -y python3-tk portaudio19-dev ffmpeg libsndfile1 libgl1-mesa-glx libegl1-mesa`
    *   **macOS:** `portaudio`, `ffmpeg`, `libsndfile` (can be installed via Homebrew). Python installed via Homebrew usually includes a compatible Tk.
    *   **Windows:** Python from python.org typically includes Tkinter. FFmpeg needs to be downloaded and its `bin` folder added to PATH. Microsoft C++ Build Tools (with "Desktop development with C++" workload) are likely needed for compiling certain Python packages if pre-compiled wheels are unavailable. Ensure you run setup from an **x64 Developer Command Prompt as Administrator** if you have 64-bit Python.

### Installation & Setup:
1.  **Save All Files:** Ensure all provided `.py` module files, `requirements.txt`, and this `README.md` are saved in a single project directory. Remove any `--- START OF FILE ... ---` or `--- END OF FILE ... ---` markers from the actual code files.
2.  **Run Setup Script:** This script automates the creation of a virtual environment and installation of dependencies. Open a terminal or command prompt (for Windows, use the **x64 Developer Command Prompt for VS, run as Administrator**) in the project's root directory and run:
    ```bash
    python flow_state_setup.py
    ```
    Follow the on-screen prompts. The script will:
    *   Check your Python version.
    *   Guide you on system dependencies.
    *   Create a Python virtual environment (e.g., in a folder named `flowstate_env`).
    *   Install required Python packages from `requirements.txt` into the virtual environment.
    *   Create convenient run scripts (e.g., `run_flow_state_music_player.bat` or `run_flow_state_music_player.sh`).

### Running Flow State:
*   **Recommended Method (Using generated run scripts):**
    Navigate to the project's root directory in your terminal (for Windows, ideally the same type of Developer Command Prompt you used for setup) and execute the script:
    *   On Windows: `run_flow_state_music_player.bat`
    *   On Linux/macOS: `./run_flow_state_music_player.sh`
*   **Manual Method:**
    1.  Activate the virtual environment:
        *   Windows (in a regular cmd or PowerShell, after setup in Dev Prompt): `.\flowstate_env\Scripts\activate`
        *   Linux/macOS: `source flowstate_env/bin/activate`
    2.  Run the main launcher script:
        ```bash
        python flow_state_launcher.py
        ```

### Development Mode:
To launch with development flags or run specific module tests:
```bash
# Using generated script
./run_flow_state_music_player.sh --dev 
# or to attempt running a specific module's __main__ block
./run_flow_state_music_player.sh --dev --run main 


#!/usr/bin/env python3
"""
Flow State Music Player - Application Launcher
Main entry point that integrates all modules
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import importlib
import subprocess 
import json 
from pathlib import Path 
import concurrent.futures
from typing import Dict, Any, Optional, Tuple, List, Callable
import platform 
import inspect # Added for HostAppInterface.request_library_action

# --- Constants ---
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1200
MIN_WINDOW_HEIGHT = 750
APP_DATA_BASE_DIR_NAME = ".flowstate" 
APP_NAME_DISPLAY = "Flow State Music Player" 

# --- Logging Configuration ---
APP_DATA_BASE_PATH = Path.home() / APP_DATA_BASE_DIR_NAME
LOG_DIR = APP_DATA_BASE_PATH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s:%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "flow_state_main_app.log", mode='a')
    ]
)
logger = logging.getLogger("FlowStateLauncher")

# --- Global Thread Pools ---
GENERAL_PURPOSE_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="AppGenericThread")
GENERAL_PURPOSE_PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1))


class HostAppInterface:
    def __init__(self, tk_root: tk.Tk, main_notebook_widget: ttk.Notebook):
        self.root: tk.Tk = tk_root
        self.notebook: ttk.Notebook = main_notebook_widget 
        self.status_bar_var: Optional[tk.StringVar] = None 
        self.app_name: str = APP_NAME_DISPLAY

        self.theme_manager: Optional[Any] = None
        self.audio_engine_ref: Optional[Any] = None
        self.music_library_db_ref: Optional[Any] = None
        self.effects_chain_ref: Optional[Any] = None
        self.plugin_manager_ref: Optional[Any] = None
        self.recommendation_engine_ref: Optional[Any] = None
        self.export_manager_ref: Optional[Any] = None

        self.main_player_ui_ref: Optional[Any] = None
        self.library_ui_ref: Optional[Any] = None
        self.effects_ui_ref: Optional[Any] = None
        self.visualization_ui_ref: Optional[Any] = None
        self.plugin_ui_ref: Optional[Any] = None
        self.recommendation_ui_ref: Optional[Any] = None
        self.storyboard_generator_ui_ref: Optional[Any] = None
        self.collab_client_ui_ref: Optional[Any] = None
        self.mobile_sync_server_ui_ref: Optional[Any] = None
        self.voice_control_ui_ref: Optional[Any] = None
        self.theme_export_main_ui_ref: Optional[Any] = None
        
        self.loaded_modules: Dict[str, Any] = {}
        self._event_callbacks: Dict[str, List[Callable]] = {}
        self.root_tk_after_id_map: Dict[str, str] = {}

    def get_audio_properties(self) -> Tuple[int, int]:
        if self.audio_engine_ref and hasattr(self.audio_engine_ref, 'sample_rate') and hasattr(self.audio_engine_ref, 'channels'):
            sr = getattr(self.audio_engine_ref, 'sample_rate', 44100)
            ch = getattr(self.audio_engine_ref, 'channels', 2)
            return int(sr), int(ch)
        logger.warning("HostApp: Audio properties requested but audio engine not ready or props not set.")
        return 44100, 2

    def get_render_properties(self) -> Tuple[int, int, int]:
        if self.visualization_ui_ref and \
           hasattr(self.visualization_ui_ref, 'engine_instance') and \
           self.visualization_ui_ref.engine_instance and \
           hasattr(self.visualization_ui_ref.engine_instance, 'config'):
            cfg = self.visualization_ui_ref.engine_instance.config
            return cfg.width, cfg.height, cfg.fps
        return 1280, 720, 60

    def get_current_track_metadata(self) -> Optional[Any]:
        if self.audio_engine_ref and hasattr(self.audio_engine_ref, 'current_metadata_obj'):
            return self.audio_engine_ref.current_metadata_obj
        return None

    def get_current_lyrics_data(self) -> Optional[List[Tuple[float, str]]]:
        if self.main_player_ui_ref and hasattr(self.main_player_ui_ref, 'lyrics_display') \
           and self.main_player_ui_ref.lyrics_display and \
           hasattr(self.main_player_ui_ref.lyrics_display, 'lyrics_data'):
            return self.main_player_ui_ref.lyrics_display.lyrics_data
        return None
    
    def get_current_playback_position(self) -> float:
        return self.audio_engine_ref.get_position() if self.audio_engine_ref else 0.0

    def request_playback_action(self, action: str, params: Optional[Dict] = None, 
                                callback: Optional[Callable[[bool, Optional[str]], None]] = None):
        if not self.audio_engine_ref:
            msg = "Playback action requested, but audio engine not available."
            logger.warning(msg)
            if self.status_bar_var: self.status_bar_var.set("Error: Audio engine not ready.")
            if callback: self.root.after(0, callback, False, msg)
            return

        params = params or {}
        logger.info(f"Host: Playback action '{action}' with params {params}")
        success = False; err_msg = None

        try:
            if action == "load_track_from_path" or action == "load_and_play_path":
                if self.main_player_ui_ref and hasattr(self.main_player_ui_ref, 'load_track_by_path_and_play'):
                    self.main_player_ui_ref.load_track_by_path_and_play(params['filepath'])
                    success = True 
                else: err_msg = f"Main player UI cannot handle '{action}'."
            elif action == "play_track_by_id":
                 if self.main_player_ui_ref and hasattr(self.main_player_ui_ref, 'play_track_by_id_from_library'):
                     self.main_player_ui_ref.play_track_by_id_from_library(params['track_id'])
                     success = True
                 else: err_msg = "Main player UI cannot handle 'play_track_by_id'."
            elif action == "add_to_queue_path":
                if hasattr(self.audio_engine_ref, 'add_to_playlist'):
                    self.audio_engine_ref.add_to_playlist(params['filepath']) 
                    success = True
                else: err_msg = "Audio engine cannot handle 'add_to_queue_path'."
            elif action == "load_playlist_paths":
                if hasattr(self.audio_engine_ref, 'load_playlist_paths'):
                    self.audio_engine_ref.load_playlist_paths(
                        params.get('paths', []),
                        play_first=params.get('play_first', True),
                        replace_queue=params.get('replace_queue', True)
                    )
                    success = True
                else: err_msg = "Audio engine missing 'load_playlist_paths'."
            elif action == "play": self.audio_engine_ref.play(**params.get('play_args', {})); success = True
            elif action == "pause": self.audio_engine_ref.pause(); success = True
            elif action == "resume": self.audio_engine_ref.resume(); success = True
            elif action == "stop": self.audio_engine_ref.stop(); success = True
            elif action == "next": self.audio_engine_ref.next_track(); success = True
            elif action == "previous": self.audio_engine_ref.previous_track(); success = True
            elif action == "set_volume": self.audio_engine_ref.set_volume(params['level']); success = True
            elif action == "seek": self.audio_engine_ref.set_position(params['position_seconds']); success = True
            elif action == "toggle_mute": self.audio_engine_ref.toggle_mute(); success = True
            elif action == "set_shuffle_mode": self.audio_engine_ref.set_shuffle_mode(params['state']); success = True
            elif action == "set_repeat_mode": self.audio_engine_ref.set_repeat_mode(params['mode']); success = True
            elif action == "clear_playlist_or_queue": 
                 if hasattr(self.audio_engine_ref, 'clear_playlist'): self.audio_engine_ref.clear_playlist(); success = True
                 else: err_msg = "Audio engine missing 'clear_playlist'."
            elif action == "play_track_at_playlist_index":
                if hasattr(self.audio_engine_ref, 'play_track_at_playlist_index'):
                    self.audio_engine_ref.play_track_at_playlist_index(params['index'])
                    success = True
                else: err_msg = "Audio engine missing 'play_track_at_playlist_index'."
            elif action == "force_sync_playback": 
                if hasattr(self.audio_engine_ref, 'force_sync_playback_to_state'):
                    self.audio_engine_ref.force_sync_playback_to_state(
                        params['library_track_id'], params['position_seconds'], params['is_playing_target']
                    )
                    success = True
                else: err_msg = "AudioEngine cannot handle 'force_sync_playback'."
            else: err_msg = f"Unknown playback action: {action}"
        except AttributeError as ae:
            err_msg = f"Component missing method for playback action '{action}': {ae}"
            logger.error(err_msg, exc_info=True)
        except KeyError as ke:
            err_msg = f"Missing parameter '{ke}' for playback action '{action}'."
            logger.error(err_msg, exc_info=True)
        except Exception as e:
            err_msg = f"Error executing playback action '{action}': {e}"
            logger.error(err_msg, exc_info=True)
        
        if success and self.status_bar_var:
             self.status_bar_var.set(f"Playback: {action.replace('_',' ').capitalize()} processed.")
        elif err_msg and self.status_bar_var:
            self.status_bar_var.set(f"Error: {err_msg}")

        if callback: self.root.after(0, callback, success, err_msg)

    def request_library_action(self, action: str, params: Optional[Dict] = None, 
                               callback: Optional[Callable[[Optional[Any]], None]] = None) -> Optional[Any]:
        if not self.music_library_db_ref:
            logger.warning("Library action requested, but MusicLibraryDB not available.")
            if callback: self.root.after(0, callback, None)
            return None
        
        params = params or {}
        logger.info(f"Host: Library action '{action}' with params {params}")

        def _db_worker():
            try:
                if hasattr(self.music_library_db_ref, action):
                    method_to_call = getattr(self.music_library_db_ref, action)
                    sig = inspect.signature(method_to_call)
                    method_params = {k: params[k] for k in sig.parameters if k in params and k != 'self'}
                    
                    # Check if all non-default args in signature are present in method_params or params
                    required_from_sig = {p.name for p in sig.parameters.values() if p.default == inspect.Parameter.empty and p.name != 'self'}
                    missing_req = required_from_sig - set(method_params.keys())
                    if missing_req:
                        logger.error(f"Missing required params {missing_req} for lib action '{action}'. Provided: {list(params.keys())}")
                        return None
                        
                    return method_to_call(**method_params)
                else: logger.warning(f"Unknown library action/method not found on DB ref: {action}"); return None
            except KeyError as ke: logger.error(f"Missing parameter '{ke}' for lib action '{action}'. Params: {params}", exc_info=True); return None
            except Exception as e: logger.error(f"Error executing lib action '{action}': {e}", exc_info=True); return None

        if callback:
            def _run_worker_and_callback():
                result = _db_worker()
                if self.root and self.root.winfo_exists(): # Ensure root window still exists
                    self.root.after(0, callback, result)
                else:
                    logger.warning(f"Root window destroyed before library action callback for '{action}'.")
            GENERAL_PURPOSE_THREAD_POOL.submit(_run_worker_and_callback)
            return None 
        else: 
            return _db_worker()

    def request_ui_focus_tab(self, tab_text_name: str):
        try:
            for i, tab_id in enumerate(self.notebook.tabs()):
                if self.notebook.tab(tab_id, "text") == tab_text_name:
                    self.notebook.select(i); logger.info(f"Host: Switched UI focus to tab '{tab_text_name}'"); return
            logger.warning(f"Host: Could not find tab '{tab_text_name}' to focus.")
        except tk.TclError as e: logger.error(f"Host: TclError focusing tab '{tab_text_name}': {e}")

    def update_status_bar(self, message: str, duration_ms: Optional[int] = 5000):
        if self.status_bar_var:
            status_key = "main_status"
            if status_key in self.root_tk_after_id_map:
                try: self.root.after_cancel(self.root_tk_after_id_map[status_key])
                except tk.TclError: pass 
                if status_key in self.root_tk_after_id_map: del self.root_tk_after_id_map[status_key]
            self.status_bar_var.set(message)
            if duration_ms and duration_ms > 0 :
                def _clear_status():
                    if self.status_bar_var and self.status_bar_var.get() == message: self.status_bar_var.set("Ready")
                    if status_key in self.root_tk_after_id_map: del self.root_tk_after_id_map[status_key]
                if self.root and self.root.winfo_exists():
                    self.root_tk_after_id_map[status_key] = self.root.after(duration_ms, _clear_status)
        else: logger.debug(f"Status bar var not set. Message: {message}")

    def subscribe_to_event(self, event_name: str, callback: Callable):
        if event_name not in self._event_callbacks: self._event_callbacks[event_name] = []
        if callback not in self._event_callbacks[event_name]:
            self._event_callbacks[event_name].append(callback)
            logger.debug(f"Callback {getattr(callback, '__name__', 'unknown')} subscribed to event '{event_name}'")

    def unsubscribe_from_event(self, event_name: str, callback: Callable):
        if event_name in self._event_callbacks and callback in self._event_callbacks[event_name]:
            self._event_callbacks[event_name].remove(callback)
            logger.debug(f"Callback {getattr(callback, '__name__', 'unknown')} unsubscribed from event '{event_name}'")

    def publish_event(self, event_name: str, *args, **kwargs):
        logger.debug(f"Host: Publishing event '{event_name}' with args: {args}, kwargs: {kwargs}")
        if event_name in self._event_callbacks:
            for callback in list(self._event_callbacks[event_name]):
                try: 
                    if self.root and self.root.winfo_exists(): self.root.after(0, callback, *args, **kwargs) 
                except Exception as e: logger.error(f"Error in callback for event '{event_name}': {e}", exc_info=True)

    def request_save_session_state(self):
        if self.main_player_ui_ref and hasattr(self.main_player_ui_ref, 'gather_session_state'):
            session_data = self.main_player_ui_ref.gather_session_state()
            if session_data:
                session_file = APP_DATA_BASE_PATH / "session_state.json"
                try:
                    with open(session_file, 'w', encoding='utf-8') as f: json.dump(session_data, f, indent=2)
                    logger.info(f"Session state saved to {session_file}")
                except IOError as e: logger.error(f"Failed to save session state: {e}")

    def request_load_session_state(self):
        session_file = APP_DATA_BASE_PATH / "session_state.json"
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f: session_data = json.load(f)
                if self.main_player_ui_ref and hasattr(self.main_player_ui_ref, 'restore_session_state'):
                    self.main_player_ui_ref.restore_session_state(session_data)
                self.publish_event("session_state_loaded", data=session_data)
            except Exception as e: logger.error(f"Failed to load/restore session state: {e}", exc_info=True)
        else: logger.info("No previous session state file found.")


class FlowStateLauncher:
    def __init__(self, dev_mode: bool = False, run_specific_module: Optional[str] = None):
        self.root_dir = Path(__file__).parent.resolve()
        logger.info(f"Application Root Directory: {self.root_dir}")
        self.app_name = APP_NAME_DISPLAY 
        self.dev_mode = dev_mode
        self.run_specific_module = run_specific_module

        self.modules_spec = {
            'theme_manager_service': ('flow_state_theme_export', "[Service] Theme Manager", None, 0),
            'audio_engine_service': ('flow_state_main', "[Service] Audio Engine", None, 1),
            'music_library_db_service': ('flow_state_music_library', "[Service] Music Library DB", None, 2),
            'effects_chain_service': ('flow_state_audio_effects', "[Service] Effects Chain", None, 3),
            'plugin_manager_service': ('flow_state_plugin_system', "[Service] Plugin Manager", None, 4),
            'recommendation_engine_service': ('flow_state_ai_recommendations', "[Service] Recommendation Engine", None, 5),
            'export_manager_service': ('flow_state_theme_export', "[Service] Export Manager", None, 6),
            'main_player_ui': ('flow_state_main', "Player", 'create_main_player_tab', 10),
            'library_ui': ('flow_state_music_library', "Library", 'create_library_tab', 11),
            'effects_ui': ('flow_state_audio_effects', "Effects", 'create_effects_tab', 12),
            'advanced_viz_ui': ('flow_state_advanced_viz', "Visualizations", 'create_visualization_tab', 13),
            'plugins_ui': ('flow_state_plugin_system', "Plugins", 'create_plugin_tab', 14),
            'recommendations_ui': ('flow_state_ai_recommendations', "Discover", 'create_recommendation_tab', 15),
            'storyboard_ui': ('flow_state_storyboard', "Storyboard", 'create_storyboard_tab', 16),
            'collaboration_ui': ('flow_state_collaboration', "Collaborate", 'create_collaboration_tab', 17),
            'mobile_sync_ui': ('flow_state_mobile_sync', "Remote", 'create_remote_control_tab', 18),
            'voice_control_ui': ('flow_state_voice_control', "Voice", 'create_voice_control_tab', 19),
            'theme_export_ui': ('flow_state_theme_export', "Manage", 'create_theme_export_main_tab', 20),
        }
            
        self.loaded_modules: Dict[str, Any] = {} 
        self.root_tk_instance: Optional[tk.Tk] = None 
        self.file_menu_ref: Optional[tk.Menu] = None

        self.check_python_environment()
        self.create_app_directories()

    def check_python_environment(self):
        logger.info(f"Python Version: {sys.version.splitlines()[0]}")
        if sys.version_info < (3, 8):
            logger.error(f"{self.app_name} requires Python 3.8 or higher. Please upgrade.")
            try:
                root_temp = tk.Tk(); root_temp.withdraw()
                messagebox.showerror("Python Version Error", f"{self.app_name} requires Python 3.8+.\nYou are using {platform.python_version()}.", parent=None)
                root_temp.destroy()
            except tk.TclError: pass
            sys.exit(1)
        logger.info("Python version check: OK.")

    def create_app_directories(self):
        try:
            APP_DATA_BASE_PATH.mkdir(parents=True, exist_ok=True)
            (APP_DATA_BASE_PATH / "data").mkdir(parents=True, exist_ok=True)
            (APP_DATA_BASE_PATH / "themes").mkdir(parents=True, exist_ok=True)
            (APP_DATA_BASE_PATH / "plugins").mkdir(parents=True, exist_ok=True)
            (APP_DATA_BASE_PATH / "plugin_configs").mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured application data directories exist at: {APP_DATA_BASE_PATH}")
        except OSError as e:
            logger.error(f"Could not create application data directories: {e}", exc_info=True)
            try:
                root_temp = tk.Tk(); root_temp.withdraw()
                messagebox.showerror("Directory Error", f"Failed to create app data folder at {APP_DATA_BASE_PATH}.\nPlease check permissions.", parent=None)
                root_temp.destroy()
            except tk.TclError: pass
            sys.exit(1)

    def _import_module_dynamically(self, module_key: str) -> Optional[Any]:
        if module_key in self.loaded_modules: return self.loaded_modules[module_key]
        if module_key not in self.modules_spec:
            logger.error(f"Module key '{module_key}' not found in specification.")
            return None

        import_name, friendly_name, _, _ = self.modules_spec[module_key]
        logger.info(f"Attempting to import module: {friendly_name} (from {import_name}) for key '{module_key}'")
        try:
            module = importlib.import_module(import_name)
            self.loaded_modules[module_key] = module
            logger.info(f"Successfully imported module: {friendly_name} (from {import_name})")
            return module
        except ImportError as e:
            logger.error(f"CRITICAL IMPORT ERROR for module '{friendly_name}' ({import_name}): {e}", exc_info=True)
            if self.root_tk_instance and self.root_tk_instance.winfo_exists(): 
                 messagebox.showerror("Fatal Module Load Error", f"Could not load essential module: {friendly_name}\nFile: {import_name}.py\n\nError: {e}\n\nThe application cannot continue.", parent=self.root_tk_instance)
            sys.exit(f"Failed to import {friendly_name}")
        except Exception as e_gen:
            logger.error(f"CRITICAL UNEXPECTED ERROR while importing module '{friendly_name}': {e_gen}", exc_info=True)
            if self.root_tk_instance and self.root_tk_instance.winfo_exists():
                messagebox.showerror("Fatal Module Load Error", f"Unexpected error loading: {friendly_name}\n\nError: {e_gen}", parent=self.root_tk_instance)
            sys.exit(f"Unexpected error importing {friendly_name}")
        return None 

    def _initialize_core_services(self, host_app: HostAppInterface):
        logger.info("Initializing core services...")
        critical_service_failed = False

        logger.info("Attempting to initialize ThemeManager service...")
        theme_module = self._import_module_dynamically('theme_manager_service')
        if theme_module and hasattr(theme_module, 'ThemeManager'):
            host_app.theme_manager = theme_module.ThemeManager(host_app.root)
            logger.info("ThemeManager service initialized.")
        else: logger.error("Failed to initialize ThemeManager service (module load or class missing)."); critical_service_failed = True
        
        logger.info("Attempting to initialize AudioEngine service...")
        audio_module = self._import_module_dynamically('audio_engine_service')
        if audio_module and hasattr(audio_module, 'AudioEngine'):
            host_app.audio_engine_ref = audio_module.AudioEngine(host_app_ref=host_app)
            logger.info("AudioEngine service initialized.")
        else: logger.error("Failed to initialize AudioEngine service (module load or class missing)."); critical_service_failed = True
        
        if critical_service_failed:
            logger.critical("One or more critical services (ThemeManager or AudioEngine) failed to initialize. Application cannot continue.")
            if host_app.root and host_app.root.winfo_exists(): messagebox.showerror("Critical Service Error", "Failed to initialize essential services (ThemeManager or AudioEngine).\nApplication cannot continue.", parent=host_app.root)
            sys.exit(1)

        logger.info("Attempting to initialize MusicLibraryDB service...")
        lib_db_module = self._import_module_dynamically('music_library_db_service')
        if lib_db_module and hasattr(lib_db_module, 'MusicLibraryDB'):
            db_file = APP_DATA_BASE_PATH / "data" / "flow_state_library.db"
            host_app.music_library_db_ref = lib_db_module.MusicLibraryDB(db_path=str(db_file))
            logger.info("MusicLibraryDB service initialized.")
        else: logger.error("Failed to initialize MusicLibraryDB service.")

        logger.info("Attempting to initialize AudioEffectsChain service...")
        effects_module = self._import_module_dynamically('effects_chain_service')
        if effects_module and hasattr(effects_module, 'AudioEffectsChain'):
            sr, ch = host_app.get_audio_properties()
            host_app.effects_chain_ref = effects_module.AudioEffectsChain(sample_rate=sr, channels=ch)
            if host_app.audio_engine_ref: host_app.audio_engine_ref.effects_chain_ref_from_host = host_app.effects_chain_ref
            logger.info("AudioEffectsChain service initialized.")
        else: logger.error("Failed to initialize AudioEffectsChain service.")

        logger.info("Attempting to initialize PluginManager service...")
        plugin_module = self._import_module_dynamically('plugin_manager_service')
        if plugin_module and hasattr(plugin_module, 'PluginManager'):
            host_app.plugin_manager_ref = plugin_module.PluginManager(host_app_ref=host_app)
            logger.info("PluginManager service initialized.")
        else: logger.error("Failed to initialize PluginManager service.")
        
        logger.info("Attempting to initialize RecommendationEngine service...")
        rec_module = self._import_module_dynamically('recommendation_engine_service')
        if rec_module and hasattr(rec_module, 'RecommendationEngine') and host_app.music_library_db_ref:
            host_app.recommendation_engine_ref = rec_module.RecommendationEngine(host_app_ref=host_app, default_user_id="default_user_main")
            logger.info("RecommendationEngine service initialized.")
        elif not host_app.music_library_db_ref: logger.warning("RecommendationEngine service skipped: MusicLibraryDB not available.")
        else: logger.error("Failed to initialize RecommendationEngine service.")
        
        logger.info("Attempting to initialize ExportManager service...")
        if theme_module and hasattr(theme_module, 'ExportManager'): 
            def _export_progress_cb(percentage, message): host_app.update_status_bar(f"Export: {message} ({percentage:.0f}%)" if percentage < 100 else f"Export: {message}", duration_ms=None if percentage < 100 else 5000)
            host_app.export_manager_ref = theme_module.ExportManager(progress_callback=_export_progress_cb, host_app_ref=host_app)
            logger.info("ExportManager service initialized.")
        else: logger.error("Failed to initialize ExportManager service (likely theme_module issue).")
        logger.info("Core services initialization attempt complete.")

    def _create_and_integrate_module_tabs(self, host_app: HostAppInterface):
        # --- RESTORED ORIGINAL LOGIC WITH DETAILED LOGGING ---
        logger.info("Creating and integrating UI module tabs...")
        ui_module_specs = sorted(
            [(key, spec) for key, spec in self.modules_spec.items() if spec[2] is not None],
            key=lambda item: item[1][3]
        )

        for module_key, (_import_name, friendly_tab_name, creator_func_name, _priority) in ui_module_specs:
            logger.info(f"--- Processing UI module: {friendly_tab_name} (Key: {module_key}) ---")
            module_obj = self.loaded_modules.get(module_key)
            if not module_obj: # If it's a UI-only module not preloaded as a service
                logger.info(f"Importing module for {friendly_tab_name} as it wasn't preloaded as a service (Key: {module_key}).")
                module_obj = self._import_module_dynamically(module_key)
            
            if module_obj and creator_func_name:
                logger.info(f"Found module object '{module_obj.__name__}' and creator function '{creator_func_name}' for {friendly_tab_name}.")
                try:
                    creator_function = getattr(module_obj, creator_func_name)
                    logger.info(f"About to call creator_function '{creator_func_name}' for {friendly_tab_name}...")
                    created_item = creator_function(host_app.notebook, host_app)
                    logger.info(f"Creator function for {friendly_tab_name} finished. Returned item type: {type(created_item)}")
                    
                    ui_ref_attr_name = f"{module_key}_ref" 
                    ui_instance_to_store = created_item
                    if isinstance(created_item, tuple) and len(created_item) > 0:
                        if isinstance(created_item[-1], (tk.Widget, ttk.Frame)): # Heuristic: last item is UI frame
                             ui_instance_to_store = created_item[-1]
                    
                    ui_ref_map = { # More explicit mapping
                        'main_player_ui': 'main_player_ui_ref', 'library_ui': 'library_ui_ref',
                        'effects_ui': 'effects_ui_ref', 'advanced_viz_ui': 'visualization_ui_ref',
                        'plugins_ui': 'plugin_ui_ref', 'recommendations_ui': 'recommendation_ui_ref',
                        'storyboard_ui': 'storyboard_generator_ui_ref', 'collaboration_ui': 'collab_client_ui_ref',
                        'mobile_sync_ui': 'mobile_sync_server_ui_ref', 'voice_control_ui': 'voice_control_ui_ref',
                        'theme_export_ui': 'theme_export_main_ui_ref',
                    }
                    attr_to_set = ui_ref_map.get(module_key)

                    if attr_to_set:
                        if hasattr(host_app, attr_to_set):
                            setattr(host_app, attr_to_set, ui_instance_to_store)
                            logger.info(f"UI for '{friendly_tab_name}' created and ref stored as 'host_app.{attr_to_set}'.")
                        else:
                            logger.error(f"HostAppInterface has no predefined attribute '{attr_to_set}' for UI ref of '{friendly_tab_name}'. UI might not be fully integrated.")
                    else: # Fallback to convention if not in map (less safe)
                        if hasattr(host_app, ui_ref_attr_name):
                            setattr(host_app, ui_ref_attr_name, ui_instance_to_store)
                            logger.info(f"UI for '{friendly_tab_name}' created and ref stored by convention as 'host_app.{ui_ref_attr_name}'.")
                        else:
                            logger.warning(f"HostAppInterface has no attribute (by map or convention) for UI ref of '{friendly_tab_name}'. Module key: {module_key}")

                except AttributeError as ae:
                    logger.error(f"Creator function '{creator_func_name}' not found or other AttributeError in module '{getattr(module_obj, '__name__', 'N/A')}' for '{friendly_tab_name}'. Tab not created. Error: {ae}", exc_info=True)
                except Exception as e_create:
                    logger.error(f"Error creating tab for '{friendly_tab_name}': {e_create}", exc_info=True) # Catch all here
            elif not module_obj:
                 logger.error(f"Module for '{friendly_tab_name}' (key: {module_key}) could not be loaded. Tab not created.")
            # else: module_obj exists but no creator_func_name (handled by modules_spec filtering)
        logger.info("UI module tab integration attempt complete.")

    def _setup_main_window_ui(self, root: tk.Tk, host_app: HostAppInterface):
        status_bar_frame = ttk.Frame(root, relief=tk.SUNKEN, borderwidth=1)
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        host_app.status_bar_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(status_bar_frame, textvariable=host_app.status_bar_var, anchor=tk.W)
        status_label.pack(fill=tk.X, padx=5, pady=2)
        host_app.notebook.pack(expand=True, fill='both', padx=5, pady=5)

    def _create_main_menu(self, root_tk: tk.Tk, host_app: HostAppInterface):
        menubar = tk.Menu(root_tk)
        root_tk.config(menu=menubar)

        self.file_menu_ref = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu_ref)
        self.file_menu_ref.add_separator() # Separator before Preferences
        self.file_menu_ref.add_command(label="Preferences...", state=tk.DISABLED)
        self.file_menu_ref.add_separator()
        self.file_menu_ref.add_command(label="Exit", command=lambda: self.on_app_close_launcher(host_app))

        view_menu = tk.Menu(menubar, tearoff=0) # Placeholder for View menu items
        menubar.add_cascade(label="View", menu=view_menu)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Plugin Manager...", state=tk.DISABLED) 
        tools_menu.add_command(label="Batch Export Audio...", state=tk.DISABLED)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Flow State...", command=self._show_about_dialog)

    def _populate_dynamic_menu_items(self, host_app: HostAppInterface):
        if self.file_menu_ref:
            if host_app.main_player_ui_ref and hasattr(host_app.main_player_ui_ref, 'open_file'):
                self.file_menu_ref.insert_command(0, label="Open File...", command=host_app.main_player_ui_ref.open_file)
            if host_app.main_player_ui_ref and hasattr(host_app.main_player_ui_ref, 'open_folder_threaded'):
                self.file_menu_ref.insert_command(1, label="Open Folder...", command=host_app.main_player_ui_ref.open_folder_threaded)
        
        try:
            actual_menubar_widget = host_app.root.winfo_toplevel().cget('menu') # Get menu path
            actual_menubar = host_app.root.winfo_toplevel().nametowidget(actual_menubar_widget) if isinstance(actual_menubar_widget, str) else None
            
            if isinstance(actual_menubar, tk.Menu):
                theme_module = self.loaded_modules.get('theme_export_ui') or self.loaded_modules.get('theme_manager_service')
                if theme_module and hasattr(theme_module, 'create_theme_menu_items') and host_app.theme_manager:
                     theme_module.create_theme_menu_items(actual_menubar, host_app)
                else: logger.warning("Could not create Themes menu dynamically.")

                # Tools menu items enabling - Assuming "Tools" is the 3rd menu (index 2) created in _create_main_menu
                if actual_menubar.index(tk.END) is not None and actual_menubar.index(tk.END) >=2: # Check enough menus exist
                    tools_menu_name_path = actual_menubar.entrycget(2, "menu") # "File", "View", "Tools"
                    if tools_menu_name_path:
                        tools_menu_widget = host_app.root.winfo_toplevel().nametowidget(tools_menu_name_path)
                        if isinstance(tools_menu_widget, tk.Menu):
                            if host_app.plugin_ui_ref: tools_menu_widget.entryconfigure("Plugin Manager...", state=tk.NORMAL, command=lambda: host_app.request_ui_focus_tab("Plugins"))
                            if host_app.theme_export_main_ui_ref and hasattr(host_app.theme_export_main_ui_ref, 'open_detailed_batch_audio_export_dialog'):
                                 tools_menu_widget.entryconfigure("Batch Export Audio...", state=tk.NORMAL, command=host_app.theme_export_main_ui_ref.open_detailed_batch_audio_export_dialog)
            else: logger.warning("Could not obtain actual menubar widget to populate dynamic items.")
        except (tk.TclError, IndexError, AttributeError) as e_menu_dyn:
            logger.warning(f"Could not dynamically configure some menu items: {e_menu_dyn}", exc_info=True)

    def _show_about_dialog(self):
        parent_window = self.root_tk_instance if self.root_tk_instance else None
        messagebox.showinfo("About Flow State Music Player",
                            f"{APP_NAME_DISPLAY}\n\nVersion: 0.1.0 (Alpha Development)\n\n"
                            "An Immersive Music Experience.\n\n"
                            "Placeholder Copyright.", parent=parent_window)
        
    def on_app_close_launcher(self, host_app: HostAppInterface):
        logger.info(f"{self.app_name} is closing via WM_DELETE_WINDOW or Exit menu...")
        try:
            if host_app: host_app.request_save_session_state()
            # Iterate over known UI ref attributes on host_app
            ui_ref_attr_names = [attr for attr in dir(host_app) if attr.endswith('_ui_ref') or attr.endswith('_ui_instance_ref')]
            for attr_name in ui_ref_attr_names:
                ui_ref = getattr(host_app, attr_name, None)
                if ui_ref and hasattr(ui_ref, 'on_app_exit') and callable(ui_ref.on_app_exit):
                    try: logger.debug(f"Calling on_app_exit for {ui_ref.__class__.__name__}"); ui_ref.on_app_exit()
                    except Exception as e: logger.error(f"Error: on_app_exit for {ui_ref.__class__.__name__}: {e}", exc_info=True)
            
            if host_app and host_app.audio_engine_ref and hasattr(host_app.audio_engine_ref, 'cleanup'): host_app.audio_engine_ref.cleanup()
            
            logger.info("Shutting down global thread/process pools...")
            # cancel_futures is Python 3.9+ for ThreadPoolExecutor
            GENERAL_PURPOSE_THREAD_POOL.shutdown(wait=False, cancel_futures=True if sys.version_info >= (3,9) else False)
            GENERAL_PURPOSE_PROCESS_POOL.shutdown(wait=True, cancel_futures=True if sys.version_info >= (3,9) else False) 
            logger.info("Global pools shutdown initiated.")
        except Exception as e: logger.error(f"Error during pre-destroy cleanup: {e}", exc_info=True)
        finally:
            if self.root_tk_instance and self.root_tk_instance.winfo_exists():
                 try: self.root_tk_instance.destroy()
                 except tk.TclError: pass 
            logger.info(f"{self.app_name} main close sequence finished from launcher.")
            sys.exit(0)

    def run_integrated_application(self):
        logger.info(f"Starting {self.app_name} in Integrated Mode...")
        if not self.root_tk_instance: self.root_tk_instance = tk.Tk()
        
        self.root_tk_instance.title(f"{self.app_name}")
        self.root_tk_instance.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
        self.root_tk_instance.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        icon_name = "flow_state_icon"
        icon_path_resolved = None
        if platform.system() == "Windows": icon_path_resolved = self.root_dir / f"{icon_name}.ico"
        else: icon_path_resolved = self.root_dir / f"{icon_name}.png"

        if icon_path_resolved and icon_path_resolved.is_file():
            try:
                if platform.system() == "Windows": self.root_tk_instance.iconbitmap(str(icon_path_resolved))
                else: 
                    img = tk.PhotoImage(file=str(icon_path_resolved))
                    self.root_tk_instance.iconphoto(True, img)
            except tk.TclError as e: logger.warning(f"Could not set window icon from {icon_path_resolved}: {e}")
        else: logger.debug(f"Icon file {icon_path_resolved} not found. Ensure it's in the project root.")

        host_app = HostAppInterface(self.root_tk_instance, ttk.Notebook(self.root_tk_instance))
        host_app.loaded_modules = self.loaded_modules 

        self._initialize_core_services(host_app)
        
        if host_app.theme_manager:
            current_theme_obj = host_app.theme_manager.get_current_theme() 
            if current_theme_obj:
                self.root_tk_instance.configure(background=current_theme_obj.primary_bg)
                host_app.theme_manager.apply_theme(current_theme_obj.name) 
            else: logger.warning("No current theme returned by ThemeManager after init. UI will use defaults.")
        else: logger.error("ThemeManager not initialized on HostAppInterface! UI theming will fail.")


        self._setup_main_window_ui(self.root_tk_instance, host_app) 
        self._create_main_menu(self.root_tk_instance, host_app)      
        self._create_and_integrate_module_tabs(host_app) # <--- THIS IS NOW THE ORIGINAL COMPLEX VERSION
        self._populate_dynamic_menu_items(host_app)           
        
        self.root_tk_instance.after(300, host_app.request_load_session_state)
        self.root_tk_instance.after(500, lambda: host_app.publish_event("app_initialized_and_ready"))
        
        self.root_tk_instance.protocol("WM_DELETE_WINDOW", lambda ha=host_app: self.on_app_close_launcher(ha))
        
        logger.info(f"{self.app_name} UI fully initialized. Starting Tkinter main loop.")
        try:
            self.root_tk_instance.mainloop()
        except KeyboardInterrupt: 
            logger.info("KeyboardInterrupt received by launcher, initiating shutdown.")
            self.on_app_close_launcher(host_app)
        
        logger.info("Tkinter mainloop in launcher has finished.")
        active_threads = [t.name for t in threading.enumerate() if t.name not in ["MainThread", "_shutdown"] and not t.daemon] # Check for non-daemon threads
        if active_threads: logger.warning(f"Non-daemon threads still active after mainloop: {active_threads}")
        
        # Check if sys.exit has already been called by on_app_close_launcher
        # This is tricky. A simple flag might be better.
        # For now, assume on_app_close_launcher always calls sys.exit.
        # If it didn't, the process might hang if non-daemon threads are running.

    def run_development_mode(self):
        logger.info(f"--- {self.app_name} Development Mode ---")
        if self.run_specific_module:
            module_py_filename = f"flow_state_{self.run_specific_module}.py"
            module_path = self.root_dir / module_py_filename
            if module_path.exists():
                logger.info(f"Attempting to run specific module's test code: {module_py_filename}")
                try:
                    process = subprocess.Popen([sys.executable, str(module_path)])
                    process.wait() 
                    logger.info(f"Module {module_py_filename} test finished with code {process.returncode}")
                except Exception as e_run: logger.error(f"Failed to run module {module_py_filename}: {e_run}")
            else:
                logger.error(f"Specified module file not found: {module_py_filename}")
                self.run_integrated_application() 
        else:
            self.run_integrated_application()

    def run(self):
        if self.dev_mode:
            self.run_development_mode()
        else:
            self.run_integrated_application()

if __name__ == "__main__":
    is_dev_mode = "--dev" in sys.argv or "-d" in sys.argv
    specific_module_to_run = None
    if "--run" in sys.argv:
        try:
            idx = sys.argv.index("--run")
            if idx + 1 < len(sys.argv) and not sys.argv[idx+1].startswith("--"):
                specific_module_to_run = sys.argv[idx+1]
        except ValueError: pass

    if is_dev_mode: logger.info("Development mode activated.")
    if specific_module_to_run: logger.info(f"Dev mode: Target module to run standalone: '{specific_module_to_run}'.")
    
    APP_DATA_BASE_PATH.mkdir(parents=True, exist_ok=True)

    launcher = FlowStateLauncher(dev_mode=is_dev_mode, run_specific_module=specific_module_to_run)
    try:
        launcher.run()
    except SystemExit as se: 
        logger.info(f"Application exited via SystemExit with code: {se.code}")
    except tk.TclError as e_tk_fatal: 
        logger.critical(f"A fatal Tkinter TclError occurred: {e_tk_fatal}", exc_info=True)
        try: 
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("Fatal UI Error", f"A critical UI error occurred: {e_tk_fatal}\nApplication will close.", parent=None)
            root_err.destroy()
        except: pass
    except Exception as e_fatal:
        logger.critical(f"An unhandled fatal error occurred at launcher's top level: {e_fatal}", exc_info=True)
        try:
            root_err = tk.Tk(); root_err.withdraw()
            messagebox.showerror("Fatal Error", f"Unexpected critical error: {e_fatal}\nPlease check logs. Application will close.", parent=None)
            root_err.destroy()
        except: pass
    finally:
        logger.info("Launcher __main__ sequence has fully completed or program is terminating.")


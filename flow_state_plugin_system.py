
"""
Flow State: Plugin System Module
Extensible plugin architecture for audio effects, visualizers, and features
"""

import os
import sys
import json
import importlib.util
import inspect
import traceback
from enum import Enum 
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type, Union, Tuple # Added Tuple
from abc import ABC, abstractmethod 
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np 
from scipy import signal 
import concurrent.futures
import logging
from pathlib import Path 

logger = logging.getLogger("FlowStatePluginSys")

PLUGIN_LOAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="PluginLoad")

class PluginType(Enum):
    AUDIO_EFFECT = "Audio Effect" 
    VISUALIZER = "Visualizer"
    ANALYZER = "Audio Analyzer"
    EXPORT = "Export Format"
    UI_EXTENSION = "UI Extension"
    METADATA_PROVIDER = "Metadata Provider"
    LYRICS_PROVIDER = "Lyrics Provider"

@dataclass
class PluginInfo:
    name: str 
    version: str
    author: str
    description: str
    plugin_type: PluginType 
    # Optional, for PluginManager UI if needed, actual runtime enabled is on instance
    # globally_enabled_by_user: bool = True 
    
    # Filled by PluginManager during loading
    _module_path: Optional[Path] = field(default=None, repr=False, compare=False)
    _class_name: Optional[str] = field(default=None, repr=False, compare=False)


    @property
    def type_str(self) -> str:
        return self.plugin_type.value


class PluginBase(ABC):
    def __init__(self):
        self.info: PluginInfo = PluginInfo( 
            name="DefaultPluginBase", version="0.0.0", author="System",
            description="This is a base plugin class.", plugin_type=PluginType.UI_EXTENSION
        )
        self.enabled: bool = True 
        self.bypass: bool = False 
        self.config: Dict[str, Any] = {} 
        self.host_app: Optional[Any] = None

    def initialize(self, host_app_interface: Any):
        self.host_app = host_app_interface
        logger.info(f"Plugin '{self.info.name}' instance initialized (Type: {self.info.type_str}).")
        # Example: Load instance config from a central plugin config store if available
        # if self.host_app and self.host_app.plugin_manager_ref:
        #     instance_id_for_config = # Some unique ID for this instance
        #     saved_config = self.host_app.plugin_manager_ref.get_plugin_instance_config(self.info.name, instance_id_for_config)
        #     if saved_config: self.load_config(saved_config)

    @abstractmethod
    def process(self, data: Any) -> Any:
        if not self.enabled: return data
        return data

    def get_ui(self, parent_widget: tk.Widget) -> Optional[ttk.Frame]:
        return None 

    def cleanup(self):
        logger.info(f"Plugin '{self.info.name}' instance cleaned up.")
        pass

    def save_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def load_config(self, saved_config_dict: Dict[str, Any]):
        if isinstance(saved_config_dict, dict):
            self.config.update(saved_config_dict)
            if hasattr(self, '_on_config_loaded'):
                 self._on_config_loaded() 
            logger.info(f"Plugin '{self.info.name}' instance loaded config: {list(saved_config_dict.keys())}")
        else:
            logger.warning(f"Invalid config format for {self.info.name} instance: {type(saved_config_dict)}")

    def _on_config_loaded(self):
        """Protected method, optionally implemented by subclasses to react after config is loaded."""
        pass


class AudioEffectPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        # self.info must be overridden by concrete subclass
        self.sample_rate: int = 44100
        self.channels: int = 2
        self.parameters: Dict[str, Any] = {} # Audio processing parameters

    def set_stream_properties(self, sample_rate: int, channels: int):
        needs_reinit = False
        if self.sample_rate != sample_rate: self.sample_rate = sample_rate; needs_reinit = True
        if self.channels != channels: self.channels = channels; needs_reinit = True
        if needs_reinit:
            if hasattr(self, '_on_parameter_change') and callable(self._on_parameter_change):
                 self._on_parameter_change("_stream_props_changed", None)
            self.reset() 

    def reset(self):
        logger.debug(f"AudioEffectPlugin '{self.info.name}' reset for SR={self.sample_rate}, CH={self.channels}")
        pass 

    @abstractmethod
    def process_audio_block(self, audio_block: np.ndarray) -> np.ndarray:
        # Subclass implements the core audio processing.
        # This base version provides enabled/bypass and channel adaptation.
        if self.bypass or not self.enabled: return audio_block
        input_ch = audio_block.shape[1] if audio_block.ndim == 2 else 1
        if input_ch == self.channels: return np.copy(audio_block)
        elif input_ch == 1 and self.channels == 2: return np.tile(audio_block[:, np.newaxis], (1,2)).astype(audio_block.dtype)
        elif input_ch == 2 and self.channels == 1: return np.mean(audio_block, axis=1, keepdims=True).astype(audio_block.dtype)
        logger.warning(f"{self.info.name}: Unhandled channel conversion {input_ch}->{self.channels}. Passing through."); return audio_block

    def process(self, data: Any) -> Any:
        if not self.enabled: return data # Top-level enabled check from PluginBase
        # AudioEffectPlugin specific process delegates to process_audio_block
        if isinstance(data, np.ndarray):
            return self.process_audio_block(data) # Subclass override of process_audio_block will be called
        logger.warning(f"{self.info.name} (AudioEffectPlugin) received non-numpy data: {type(data)}")
        return data

    def save_config(self) -> Dict[str, Any]: # Override to save parameters
        return self.parameters.copy()

    def load_config(self, saved_params_dict: Dict[str, Any]): # Override to load parameters
        if isinstance(saved_params_dict, dict):
            for key, value in saved_params_dict.items():
                if hasattr(self, 'set_parameter') and callable(getattr(self, 'set_parameter')):
                    getattr(self, 'set_parameter')(key, value)
                elif key in self.parameters: self.parameters[key] = value
                else: self.parameters[key] = value; logger.debug(f"Loaded new/unknown param '{key}' for {self.info.name}")
            
            if hasattr(self, '_on_parameter_change') and callable(self._on_parameter_change):
                 self._on_parameter_change("_config_loaded", self.parameters)
            logger.info(f"AudioEffectPlugin '{self.info.name}' loaded params from config: {list(saved_params_dict.keys())}")
        else:
            logger.warning(f"Invalid parameter config format for {self.info.name}: {type(saved_params_dict)}")
    
    def set_parameter(self, name: str, value: Any): # Optional helper for subclasses
        if name not in self.parameters or self.parameters[name] != value or type(self.parameters[name]) != type(value):
            self.parameters[name] = value
            if hasattr(self, '_on_parameter_change') and callable(self._on_parameter_change):
                self._on_parameter_change(name, value)


class VisualizerPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        # self.info must be overridden by concrete subclass
        self.width: int = 1280; self.height: int = 720; self.fps: int = 30
        self.ctx: Optional[Any] = None 

    def set_render_properties(self, width: int, height: int, fps: int, moderngl_ctx: Optional[Any] = None):
        self.width, self.height, self.fps, self.ctx = width, height, fps, moderngl_ctx
        self.reset_visual()

    def reset_visual(self):
        logger.debug(f"VisualizerPlugin '{self.info.name}' reset for {self.width}x{self.height} @ {self.fps}fps")
        pass

    @abstractmethod
    def render_frame(self, audio_data_mono: np.ndarray, audio_data_fft: np.ndarray, waveform_data: np.ndarray, time_info: Dict) -> Optional[Any]:
        pass

    def process(self, data: Any) -> Any:
        if not self.enabled: return None # If visualizer is disabled, return nothing
        if isinstance(data, dict) and all(k in data for k in ['audio_mono', 'audio_fft', 'waveform', 'time_info']):
            return self.render_frame(data['audio_mono'], data['audio_fft'], data['waveform'], data['time_info'])
        logger.warning(f"{self.info.name} (VisualizerPlugin) received unexpected data: {type(data)}")
        return None


class SimpleFilterPlugin(AudioEffectPlugin): 
    def __init__(self):
        super().__init__() 
        self.info = PluginInfo( 
            name="Simple Filter", version="1.0.2", author="FlowState Core Plugins",
            description="A basic Butterworth Lowpass/Highpass filter plugin.",
            plugin_type=PluginType.AUDIO_EFFECT
        )
        self.parameters = {'filter_type': 'lowpass', 'cutoff_hz': 1000.0, 'order': 2}
        self.b_coeffs: np.ndarray = np.array([1.0]); self.a_coeffs: np.ndarray = np.array([1.0])
        self.filter_zi_per_channel: List[np.ndarray] = [] 
        # Initial design called by set_stream_properties via initialize() from PluginManager/EffectsChain

    def initialize(self, host_app_interface: Any): 
        super().initialize(host_app_interface) 
        if self.host_app and hasattr(self.host_app, 'get_audio_properties'):
            sr, ch = self.host_app.get_audio_properties()
            self.set_stream_properties(sr, ch) 
        else: self.set_stream_properties(44100, 2) 

    def _design_filter(self): 
        if self.sample_rate <= 0: self.b_coeffs,self.a_coeffs=np.array([1.]),np.array([1.]); logger.warning(f"{self.info.name}: Invalid SR."); return
        ft, co, od = self.parameters.get('filter_type','lowpass'), float(self.parameters.get('cutoff_hz',1000)), int(self.parameters.get('order',2))
        nyq = self.sample_rate/2.; co = np.clip(co, 1., nyq-1.) # Clip cutoff
        od = max(1, min(od, 10)) # Clip order (e.g. 1 to 10)
        try: self.b_coeffs,self.a_coeffs=signal.butter(od,co,btype=ft,fs=self.sample_rate)
        except ValueError as e: logger.error(f"{self.info.name} filter design error: {e}"); self.b_coeffs,self.a_coeffs=np.array([1.]),np.array([1.])
        
        self.filter_zi_per_channel = []
        if self.channels > 0 and self.b_coeffs.size > 0 and self.a_coeffs.size > 0 and not (self.b_coeffs[0]==1. and len(self.b_coeffs)==1): # not passthrough
            try:
                zi_s = signal.lfilter_zi(self.b_coeffs,self.a_coeffs)
                self.filter_zi_per_channel = [np.copy(zi_s) for _ in range(self.channels)]
            except ValueError as e_zi: logger.error(f"{self.info.name} ZI error: {e_zi}"); self.filter_zi_per_channel = []

    def _on_parameter_change(self, name: str, value: Any): # Called by self.set_parameter (from base)
        if name in ['filter_type', 'cutoff_hz', 'order', '_stream_props_changed', '_config_loaded']:
            self._design_filter()

    def reset(self): super().reset(); self._design_filter() 

    def process_audio_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_audio_block(audio_block) 
        if block_to_process is audio_block and (self.bypass or not self.enabled): return audio_block
        if self.b_coeffs.size <= 1 or not self.filter_zi_per_channel or len(self.filter_zi_per_channel) != block_to_process.shape[1]:
            if len(self.filter_zi_per_channel) != block_to_process.shape[1] and self.channels == block_to_process.shape[1]: self._design_filter() 
            if len(self.filter_zi_per_channel) != block_to_process.shape[1]: return block_to_process 
            if self.b_coeffs.size <= 1: return block_to_process

        output_block = np.empty_like(block_to_process)
        for ch_idx in range(block_to_process.shape[1]):
            output_block[:, ch_idx], self.filter_zi_per_channel[ch_idx] = signal.lfilter(
                self.b_coeffs, self.a_coeffs, block_to_process[:, ch_idx], zi=self.filter_zi_per_channel[ch_idx] )
        return output_block.squeeze() if output_block.shape[1] == 1 and audio_block.ndim == 1 else output_block

    def get_ui(self, parent_widget: tk.Widget) -> Optional[ttk.Frame]:
        frame = ttk.Frame(parent_widget); ttk.Label(frame, text=f"{self.info.name} Controls", font=('Arial',11,'bold')).pack(pady=(5,10),anchor='center')
        
        type_var = tk.StringVar(value=self.parameters.get('filter_type','lowpass'))
        def _set_type(): self.set_parameter('filter_type', type_var.get())
        tf = ttk.Frame(frame); tf.pack(fill=tk.X,pady=3); ttk.Label(tf,text="Type:").pack(side=tk.LEFT,padx=5)
        ttk.Radiobutton(tf,text="Lowpass",variable=type_var,value="lowpass",command=_set_type).pack(side=tk.LEFT)
        ttk.Radiobutton(tf,text="Highpass",variable=type_var,value="highpass",command=_set_type).pack(side=tk.LEFT)

        cf = ttk.Frame(frame); cf.pack(fill=tk.X, pady=3); ttk.Label(cf, text="Cutoff (Hz):").pack(side=tk.LEFT, padx=5)
        co_val_str = tk.StringVar(value=f"{self.parameters.get('cutoff_hz',1000.0):.0f}")
        co_slider_var = tk.DoubleVar(value=self.parameters.get('cutoff_hz',1000.0)) # For slider binding
        def _upd_co_sl(v_str): co_val = co_slider_var.get(); co_val_str.set(f"{co_val:.0f}"); self.set_parameter('cutoff_hz', co_val)
        def _upd_co_en(e=None): 
            try: nv=float(co_val_str.get()); co_slider_var.set(nv); self.set_parameter('cutoff_hz',nv)
            except ValueError: co_val_str.set(f"{self.parameters.get('cutoff_hz',1000.0):.0f}")
        co_sl = ttk.Scale(cf,from_=20,to=20000,orient=tk.HORIZONTAL,variable=co_slider_var,command=_upd_co_sl,length=180); co_sl.pack(side=tk.LEFT,padx=5,expand=True,fill=tk.X)
        co_en = ttk.Entry(cf,textvariable=co_val_str,width=7); co_en.pack(side=tk.LEFT,padx=5); co_en.bind("<Return>",_upd_co_en); co_en.bind("<FocusOut>",_upd_co_en)

        of = ttk.Frame(frame); of.pack(fill=tk.X,pady=3); ttk.Label(of,text="Order:").pack(side=tk.LEFT,padx=5)
        order_var = tk.IntVar(value=self.parameters.get('order',2))
        def _upd_ord(): self.set_parameter('order',order_var.get())
        ttk.Spinbox(of,from_=1,to=8,textvariable=order_var,command=_upd_ord,width=5).pack(side=tk.LEFT,padx=5)
        return frame


class PluginManager:
    def __init__(self, host_app_ref: Any):
        self.host_app = host_app_ref
        self.default_plugin_dir = Path.home() / ".flowstate" / "plugins"
        self.default_plugin_dir.mkdir(parents=True, exist_ok=True)
        self.available_plugin_types: Dict[str, Tuple[PluginInfo, Type[PluginBase]]] = {} # PluginInfo.name -> (PluginInfo, PluginClass)
        self.plugin_type_configs: Dict[str, Dict] = {} # PluginInfo.name -> {is_globally_enabled: bool, ...}
        self.plugin_configs_dir = Path.home() / ".flowstate" / "plugin_configs" # For instance configs
        self.plugin_configs_dir.mkdir(parents=True, exist_ok=True)
        self._load_plugin_type_configs() # Load global enable/disable states for plugin types
        self._load_all_plugins_async()

    def _load_all_plugins_async(self, plugin_dirs: Optional[List[Path]] = None):
        dirs = plugin_dirs if plugin_dirs else [self.default_plugin_dir]
        logger.info(f"PluginManager: Scanning for plugins in: {dirs}")
        self.available_plugin_types.clear() # Rescan
        
        futures_map = {} # filepath -> future
        for p_dir in dirs:
            if not p_dir.is_dir(): continue
            for filepath in p_dir.glob("*.py"):
                if filepath.name == "__init__.py": continue
                future = PLUGIN_LOAD_POOL.submit(self._load_plugin_from_file, filepath)
                futures_map[filepath] = future
        
        # Process results as they complete, or after all are submitted
        def _process_loaded_plugins():
            newly_loaded_count = 0
            for filepath, future in futures_map.items():
                try:
                    result = future.result(timeout=5) # Timeout per plugin load
                    if result: # Result is (PluginInfo, PluginClass)
                        p_info, p_class = result
                        # Store with PluginInfo.name as key for uniqueness
                        if p_info.name not in self.available_plugin_types:
                            self.available_plugin_types[p_info.name] = (p_info, p_class)
                            newly_loaded_count +=1
                        else: logger.warning(f"Plugin name conflict: '{p_info.name}' from {filepath} already loaded.")
                except concurrent.futures.TimeoutError: logger.error(f"Timeout loading plugin from {filepath}.")
                except Exception as e_load_res: logger.error(f"Error processing plugin load result for {filepath}: {e_load_res}", exc_info=True)
            
            if newly_loaded_count > 0 or not futures_map: # Publish if new plugins or if it was an empty scan (to clear UI)
                 if self.host_app: self.host_app.publish_event("plugin_list_updated")
            logger.info(f"Plugin scan complete. Total available plugin types: {len(self.available_plugin_types)}")

        # Run _process_loaded_plugins in a separate thread so _load_all_plugins_async returns quickly
        threading.Thread(target=_process_loaded_plugins, daemon=True).start()


    def _load_plugin_from_file(self, filepath: Path) -> Optional[Tuple[PluginInfo, Type[PluginBase]]]:
        module_name = f"flowstate.plugin.{filepath.stem}" # Unique module name
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(filepath))
            if not spec or not spec.loader: logger.error(f"Could not create spec for {filepath}"); return None
            
            module = importlib.util.module_from_spec(spec)
            # sys.modules[module_name] = module # Avoid polluting sys.modules globally if not necessary, exec_module handles it
            spec.loader.exec_module(module)

            for _item_name, item_class in inspect.getmembers(module, inspect.isclass):
                if issubclass(item_class, PluginBase) and item_class not in [PluginBase, AudioEffectPlugin, VisualizerPlugin]:
                    try:
                        # Instantiate temporarily to get PluginInfo (plugins MUST define self.info in __init__)
                        temp_instance = item_class()
                        plugin_info = temp_instance.info
                        if not isinstance(plugin_info, PluginInfo): continue # Not a valid plugin class
                        
                        plugin_info._module_path = filepath # Store path for reference
                        plugin_info._class_name = item_class.__name__ # Store class name
                        
                        logger.debug(f"Found plugin class '{item_class.__name__}' with info '{plugin_info.name}' in {filepath.name}")
                        return plugin_info, item_class # Return first valid plugin class found
                    except Exception as e_inst: logger.error(f"Error instantiating plugin candidate {item_class.__name__} from {filepath.name} for info: {e_inst}"); continue
        except Exception as e: logger.error(f"Error loading module from {filepath.name}: {e}", exc_info=True)
        return None # No valid plugin found in this file

    def get_available_audio_effect_plugin_infos_and_classes(self) -> List[Tuple[PluginInfo, Type[AudioEffectPlugin]]]:
        fx_plugins = []
        for p_name_key, (p_info, p_class) in self.available_plugin_types.items():
            if p_info.plugin_type == PluginType.AUDIO_EFFECT and issubclass(p_class, AudioEffectPlugin):
                # Only add if it's globally enabled for its type (from plugin_type_configs)
                # type_config = self.plugin_type_configs.get(p_info.name, {'globally_enabled': True})
                # if type_config.get('globally_enabled', True):
                fx_plugins.append((p_info, p_class)) # type: ignore [arg-type]
        return fx_plugins

    def create_plugin_instance_by_id(self, plugin_info_name: str) -> Optional[PluginBase]:
        if plugin_info_name in self.available_plugin_types:
            p_info, p_class = self.available_plugin_types[plugin_info_name]
            try:
                instance = p_class()
                # initialize() is called by the system that uses the instance (e.g. EffectsChainUI, VizEngine)
                return instance
            except Exception as e: logger.error(f"Failed to instantiate plugin '{plugin_info_name}': {e}", exc_info=True)
        else: logger.warning(f"Plugin type '{plugin_info_name}' not found in available plugins.")
        return None

    def _get_plugin_type_config_filepath(self) -> Path:
        return self.plugin_configs_dir / "plugin_type_settings.json"

    def _load_plugin_type_configs(self):
        filepath = self._get_plugin_type_config_filepath()
        if filepath.exists():
            try:
                with open(filepath, 'r') as f: self.plugin_type_configs = json.load(f)
                logger.info(f"Loaded plugin type configurations from {filepath}")
            except Exception as e: logger.error(f"Error loading plugin type configs: {e}")
        else: self.plugin_type_configs = {} # Default to empty if no file

    def save_plugin_type_configs(self):
        filepath = self._get_plugin_type_config_filepath()
        try:
            with open(filepath, 'w') as f: json.dump(self.plugin_type_configs, f, indent=2)
            logger.info(f"Saved plugin type configurations to {filepath}")
        except Exception as e: logger.error(f"Error saving plugin type configs: {e}")

    # --- Instance Config Management (Optional: If PM is central store) ---
    # def get_plugin_instance_config_path(...)
    # def save_plugin_instance_config(plugin: PluginBase, instance_id: str, config: Dict): ...
    # def load_plugin_instance_config(plugin_info_name: str, instance_id: str) -> Optional[Dict]: ...


class PluginManagerUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, plugin_manager: PluginManager, host_app_ref: Any):
        super().__init__(parent)
        self.plugin_manager = plugin_manager
        self.host_app = host_app_ref
        self.selected_plugin_info_name: Optional[str] = None
        self.current_plugin_config_frame: Optional[ttk.Frame] = None
        self._create_ui()
        self.refresh_plugin_list()
        if self.host_app:
            self.host_app.subscribe_to_event("plugin_list_updated", self.on_plugin_list_updated_event)
            if self.host_app.theme_manager: 
                self.host_app.theme_manager.register_callback(self.on_theme_changed)
                if self.host_app.theme_manager.get_current_theme(): self.on_theme_changed(self.host_app.theme_manager.get_current_theme())

    def on_theme_changed(self, theme: Any): # ... (theming logic as before) ...
        pass
    def on_plugin_list_updated_event(self, *args, **kwargs): self.refresh_plugin_list()
    def _create_ui(self): # ... (Full implementation as before, with Refresh List button) ...
        pass
    def _ask_refresh_plugins(self): self.plugin_manager._load_all_plugins_async()
    def refresh_plugin_list(self): # ... (Full implementation as before) ...
        pass
    def _on_plugin_selected_from_list(self, event: Optional[tk.Event]): # ... (Full implementation as before) ...
        pass
    def _display_plugin_details(self, plugin_info: PluginInfo): # ... (Full implementation as before) ...
        pass
    def _display_plugin_type_config_ui(self, plugin_info: PluginInfo, plugin_class: Type[PluginBase]): # ... (Full implementation as before) ...
        pass
    def _clear_details_and_config_ui(self): # ... (Full implementation as before) ...
        pass
    def cleanup_ui(self): # ... (Full implementation as before) ...
        pass


class PluginCreator(tk.Toplevel):
    def __init__(self, parent_tk_root: tk.Tk, host_app_ref: Any):
        super().__init__(parent_tk_root)
        # ... (Full implementation as before) ...
        pass
    def _create_ui(self): pass
    def generate_plugin(self): pass


def create_plugin_tab(notebook: ttk.Notebook, host_app_ref: Any) -> PluginManagerUI: # Returns UI, Manager is on host_app
    plugin_frame = ttk.Frame(notebook)
    notebook.add(plugin_frame, text="Plugins")
    
    # PluginManager service is already on host_app_ref.plugin_manager_ref
    if not host_app_ref.plugin_manager_ref:
        logger.error("PluginManager service not found on host_app! Cannot create Plugins Tab UI properly.")
        # Create a dummy manager or show error in tab
        class DummyPM: _load_all_plugins_async=lambda s:None; available_plugin_types={}; default_plugin_dir=Path(".")
        pm_instance = DummyPM() # type: ignore
        ttk.Label(plugin_frame, text="Error: Plugin Manager service failed to initialize.").pack(padx=20,pady=20)
    else:
        pm_instance = host_app_ref.plugin_manager_ref
        
    plugin_ui = PluginManagerUI(plugin_frame, pm_instance, host_app_ref=host_app_ref) # Pass correct PM
    plugin_ui.pack(fill=tk.BOTH, expand=True)
    
    create_btn_frame = ttk.Frame(plugin_frame); create_btn_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
    ttk.Button(create_btn_frame, text="Create New Plugin Template...",
               command=lambda: PluginCreator(host_app_ref.root, host_app_ref) if host_app_ref.plugin_manager_ref else None
              ).pack(anchor='center')
    
    logger.info("Plugins Tab UI created.")
    return plugin_ui # Return UI instance for consistency, launcher already has manager ref


if __name__ == "__main__":
    # ... (Standalone test block as before, ensuring MockHostApp has plugin_manager_ref) ...
    pass


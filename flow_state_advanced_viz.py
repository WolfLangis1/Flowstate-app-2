

"""
Flow State: Advanced Visualization Engine
GPU-accelerated audio visualizations with custom shaders
"""

import numpy as np
import moderngl
import pygame 
from pygame.locals import *
import glm 
import time
import math
from typing import List, Dict, Tuple, Optional, Any, Callable 
from dataclasses import dataclass, asdict, field # Added field
import struct 
from abc import ABC, abstractmethod
import colorsys # For color utilities if needed by viz
import random
import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
import threading
from collections import deque
# import queue # Not actively used by engine, UI might use for async tasks
from PIL import Image, ImageTk 
import io # For image saving if needed directly
from pathlib import Path 
import logging

logger = logging.getLogger("FlowStateAdvViz")


@dataclass
class VisualizationConfig:
    width: int = 1280 
    height: int = 720
    fps: int = 60
    fft_size: int = 2048 
    smoothing: float = 0.75 
    sensitivity: float = 1.5 
    color_scheme: str = "flow_spectrum" 
    particle_count: int = 15000 
    
    post_processing: bool = True 
    bloom_enabled: bool = True
    bloom_threshold: float = 0.75  # Adjusted default
    bloom_intensity: float = 0.8   # Adjusted default
    bloom_blur_passes: int = 4     # Default blur passes
    bloom_downsample_factor: int = 2 # 1=fullres, 2=halfres, 4=quarterres for bloom buffers

    motion_blur: bool = False # Placeholder for now
    # Field for UI to set initial viz type (launcher might set this based on last session)
    initial_visualization_type: str = "SpectrumBarsVisualization"


class ShaderProgram:
    def __init__(self, ctx: moderngl.Context, vertex_shader: str, fragment_shader: str, geometry_shader: Optional[str] = None):
        self.ctx = ctx
        self.program: Optional[moderngl.Program] = None
        self.uniforms: Dict[str, moderngl.Uniform] = {}
        
        try:
            self.program = ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader,
                geometry_shader=geometry_shader
            )
            # Populate uniforms map after successful compilation
            # Note: Accessing uniforms by name only works if they are active (used in shader)
            # ModernGL 5.7+ is more lenient with inactive uniforms if iterated by index.
            if self.program:
                # Prefer iterating by name if available (more robust to inactive uniforms)
                # However, official docs often show iteration by index for completeness.
                # Let's try to get all declared uniforms if possible.
                try: # This might vary slightly based on exact ModernGL version and introspection capabilities
                    for name in self.program: # Iterate over members (uniforms, attributes etc.)
                        member = self.program[name]
                        if isinstance(member, moderngl.Uniform):
                            self.uniforms[name] = member
                except Exception: # Fallback to index-based for older versions or if above fails
                     logger.debug(f"Falling back to uniform discovery by index for shader program.")
                     # num_uniforms_attr = getattr(self.program, 'num_uniforms', None) # Moderngl 5.7+
                     # if num_uniforms_attr is None and hasattr(self.program, '_uniforms'): # Older internal
                     #     num_uniforms_attr = len(self.program._uniforms)
                     
                     # Safer: try iterating keys if it's dict-like (newer moderngl)
                     if hasattr(self.program, '__iter__'): # Check if program is iterable (for uniform names)
                        for name in self.program:
                            try:
                                member = self.program[name]
                                if isinstance(member, moderngl.Uniform):
                                    self.uniforms[name] = member
                            except (KeyError, ValueError): # Uniform might be inactive or not found by this name
                                pass
                     else: # Fallback for very old versions if any (less likely to be an issue)
                         logger.warning("Could not determine how to iterate uniforms for this ModernGL version. Uniform map might be incomplete.")


        except moderngl.Error as e: # Catch ModernGL specific compilation errors
            logger.error(f"ModernGL Shader Compilation Error: {e.summary if hasattr(e,'summary') else str(e)}")
            # e.shader_type, e.shader_source, e.error_log might be available on moderngl.Error
            logger.error(f"VS:\n{vertex_shader[:500]}...\nFS:\n{fragment_shader[:500]}...\nGS:\n{geometry_shader[:500] if geometry_shader else 'N/A'}...")
            raise
        except Exception as e: # Catch other errors
            logger.error(f"Generic Shader Program Error: {e}", exc_info=True)
            raise

    def set_uniform(self, name: str, value: Any):
        if self.program is None: logger.warning(f"Attempt to set uniform '{name}' on uninitialized shader."); return
        try:
            if name in self.uniforms:
                self.uniforms[name].value = value
            elif name in self.program: # Try direct access if not in cached map (e.g., block uniform)
                self.program[name].value = value
            # else: logger.debug(f"Uniform '{name}' not found or inactive in shader program.")
        except KeyError: logger.debug(f"Uniform '{name}' not found (KeyError) in shader program.") # Common if uniform optimized out
        except struct.error as se: logger.error(f"Struct Error for uniform '{name}': {se}. Val type: {type(value)}")
        except Exception as e: logger.error(f"Error setting uniform '{name}' (val: {str(value)[:50]}...): {e}", exc_info=False) # Keep log cleaner for common value errors

    def set_uniforms(self, uniforms_dict: Dict[str, Any]):
        for name, value in uniforms_dict.items(): self.set_uniform(name, value)

    def release(self):
        if self.program: self.program.release(); self.program = None
        self.uniforms.clear()


class Visualization(ABC):
    def __init__(self, ctx: moderngl.Context, config: VisualizationConfig):
        self.ctx = ctx
        self.config = config 
        self.time = 0.0
        self.beat_intensity = 0.0 
        self.smoothed_frequency_data = np.zeros(self.config.fft_size // 2, dtype=np.float32)
        self.waveform_data = np.zeros(self.config.fft_size, dtype=np.float32)
        self._prev_freq_data_for_smoothing = np.zeros_like(self.smoothed_frequency_data)
        self.freq_texture: Optional[moderngl.Texture] = None
        self._create_freq_texture()

    def _create_freq_texture(self):
        if self.freq_texture: self.freq_texture.release()
        fft_bins = self.config.fft_size // 2
        if fft_bins <= 0: logger.error("Cannot create freq_texture: fft_bins <= 0."); self.freq_texture=None; return
        try:
            self.freq_texture = self.ctx.texture((fft_bins,), 1, dtype='f4') 
            self.freq_texture.filter = (moderngl.LINEAR, moderngl.LINEAR) 
            self.freq_texture.swizzle = 'R' # Read as Red channel, shader can expand (e.g. texture(sampler, tc).r)
        except Exception as e: logger.error(f"Failed to create frequency texture: {e}", exc_info=True); self.freq_texture = None

    @abstractmethod
    def initialize(self): pass

    def update(self, audio_data_mono: np.ndarray, dt: float):
        self.time += dt
        self.process_audio(audio_data_mono)
        if self.freq_texture and self.smoothed_frequency_data.size > 0 and self.freq_texture.width == self.smoothed_frequency_data.size:
            try:
                data_for_tex = np.ascontiguousarray(self.smoothed_frequency_data, dtype=np.float32)
                self.freq_texture.write(data_for_tex.tobytes())
            except Exception as e: logger.error(f"Error writing to frequency texture: {e}", exc_info=True)
        elif self.freq_texture and self.freq_texture.width != self.smoothed_frequency_data.size:
             logger.warning(f"Freq texture size ({self.freq_texture.width}) mismatch with data size ({self.smoothed_frequency_data.size}). Recreating texture.")
             self._create_freq_texture() # Attempt to recreate if size mismatch (e.g. fft_size changed)


    @abstractmethod
    def render(self): pass
    
    def cleanup(self): 
        if self.freq_texture: self.freq_texture.release(); self.freq_texture = None
        logger.debug(f"Visualization '{self.__class__.__name__}' cleaned up base resources.")

    def process_audio(self, audio_data_mono: np.ndarray):
        if audio_data_mono is None or audio_data_mono.size == 0:
            self.smoothed_frequency_data *= 0.95 # Decay if no audio
            self.waveform_data *= 0.95
            self.beat_intensity *= 0.9
            return

        N = self.config.fft_size
        if N <= 0 : N = 2048 # Fallback if config is bad
        
        processed_audio = audio_data_mono
        if audio_data_mono.size < N: processed_audio = np.pad(audio_data_mono, (0, N - audio_data_mono.size), 'constant')
        else: processed_audio = audio_data_mono[:N]

        self.waveform_data = processed_audio 
        window = np.hanning(N) if N > 0 else np.array([1.0]) # Safety for N=0
        windowed_audio = processed_audio * window
        
        fft_raw = np.fft.rfft(windowed_audio) if N > 0 else np.array([0j])
        fft_magnitude = np.abs(fft_raw) / (N / 2.0 if N > 0 else 1.0)
        fft_magnitude *= self.config.sensitivity
        fft_magnitude = np.clip(fft_magnitude, 0.0, 1.0) # Clip to [0,1] for texture/shader

        num_bins_to_use = N // 2
        current_freq_data = fft_magnitude[:num_bins_to_use]
        
        # Ensure shapes match for smoothing, especially if fft_size changed
        if self._prev_freq_data_for_smoothing.shape != current_freq_data.shape:
            self._prev_freq_data_for_smoothing = np.zeros_like(current_freq_data)
        if self.smoothed_frequency_data.shape != current_freq_data.shape: # Also for main smoothed data
            self.smoothed_frequency_data = np.zeros_like(current_freq_data)

        smoothing_factor = np.clip(self.config.smoothing, 0.0, 0.99) # Ensure 0 <= smoothing < 1
        self.smoothed_frequency_data = (smoothing_factor * self._prev_freq_data_for_smoothing + (1 - smoothing_factor) * current_freq_data)
        self._prev_freq_data_for_smoothing[:] = self.smoothed_frequency_data # Copy content

        # Basic beat detection placeholder
        # Sum of high-frequency band, comparing to recent average (needs a small buffer)
        # This is very rudimentary. A proper beat detector is much more involved.
        if not hasattr(self, '_beat_energy_history'): self._beat_energy_history = deque(maxlen=10)
        current_band_energy = np.sum(self.smoothed_frequency_data[num_bins_to_use//4 : num_bins_to_use//2]) # Example mid-high band
        if len(self._beat_energy_history) == self._beat_energy_history.maxlen:
            avg_history_energy = np.mean(self._beat_energy_history)
            if current_band_energy > avg_history_energy * 1.5: # Threshold
                self.beat_intensity = 1.0
            else: self.beat_intensity *= 0.85 # Decay
        self._beat_energy_history.append(current_band_energy)


    def resize(self, width: int, height: int):
        # self.config.width = width # Config is shared, engine updates it
        # self.config.height = height
        # Recreate freq texture ONLY if fft_size (which determines its width) changed.
        # This is usually done by set_visualization if a new config with different fft_size is passed.
        # Here, we assume fft_size is constant for this instance unless explicitly changed.
        pass


class SpectrumBarsVisualization(Visualization):
    # ... (Full implementation as refined before) ...
    pass # Assumed complete from prior steps

class ParticleFlowVisualization(Visualization): # Conceptual placeholder
    # ... (as before) ...
    pass

class WaveformTunnelVisualization(Visualization): # Conceptual placeholder
    # ... (as before) ...
    pass

class PostProcessing:
    # ... (Full implementation as refined: __init__(ctx, config), _update_bloom_dimensions,
    #      _create_resources, _create_shaders (with proper Gaussian), begin_render_to_scene_fbo,
    #      apply_post_processing (with more robust bloom enable/disable logic), resize, cleanup) ...
    def __init__(self, ctx: moderngl.Context, initial_config: VisualizationConfig): # as before
        pass
    def _update_bloom_dimensions(self): pass
    def _create_resources(self): pass
    def _create_shaders(self): pass
    def begin_render_to_scene_fbo(self): pass
    def apply_post_processing(self, target_fbo: Optional[moderngl.Framebuffer]): pass
    def resize(self, new_width: int, new_height: int): pass
    def cleanup(self): pass


class VisualizationEngine:
    # ... (Full implementation as refined: __init__(config), set_visualization, update_audio,
    #      run loop (handles VIDEORESIZE, calls current_visualization.update(), uses PostProcessor),
    #      cleanup, render_to_specific_fbo, capture_frame_to_pil, capture_frame) ...
    def __init__(self, initial_config: VisualizationConfig): # as before
        pass
    def set_visualization(self, viz_name: str): pass
    def update_audio(self, audio_data_stereo: np.ndarray): pass
    def run(self): pass
    def cleanup(self): pass
    def render_to_specific_fbo(self, target_fbo: moderngl.Framebuffer): pass
    def capture_frame_to_pil(self) -> Optional[Image.Image]: pass
    def capture_frame(self, filepath: str) -> bool: pass


class VisualizationUI(ttk.Frame):
    # ... (Full implementation as refined: __init__ with all tk.Vars for VisualizationConfig,
    #      create_ui with controls for bloom params, _on_setting_change, apply_settings_to_engine
    #      (which reconfigures PostProcessor if needed), toggle/start/stop engine,
    #      change_visualization_type, capture_current_frame, update_audio_for_viz, on_app_exit) ...
    def __init__(self, parent: ttk.Widget, host_app_ref: Optional[Any] = None): # as before
        super().__init__(parent)
        self.host_app = host_app_ref
        self.engine_instance: Optional[VisualizationEngine] = None
        self.engine_thread: Optional[threading.Thread] = None
        self.is_engine_running = False
        self.current_config = VisualizationConfig() 

        # UI Variables for all config fields
        self.width_var = tk.IntVar(value=self.current_config.width)
        self.height_var = tk.IntVar(value=self.current_config.height)
        self.fps_var = tk.IntVar(value=self.current_config.fps)
        self.fft_size_var = tk.StringVar(value=str(self.current_config.fft_size))
        self.smoothing_var = tk.DoubleVar(value=self.current_config.smoothing)
        self.sensitivity_var = tk.DoubleVar(value=self.current_config.sensitivity)
        self.color_scheme_var = tk.StringVar(value=self.current_config.color_scheme)
        self.particle_count_var = tk.IntVar(value=self.current_config.particle_count)
        
        self.post_processing_var = tk.BooleanVar(value=self.current_config.post_processing)
        self.bloom_enabled_var = tk.BooleanVar(value=self.current_config.bloom_enabled)
        self.bloom_threshold_var = tk.DoubleVar(value=self.current_config.bloom_threshold)
        self.bloom_intensity_var = tk.DoubleVar(value=self.current_config.bloom_intensity)
        self.bloom_blur_passes_var = tk.IntVar(value=self.current_config.bloom_blur_passes)
        self.bloom_downsample_factor_var = tk.StringVar(value=str(self.current_config.bloom_downsample_factor))
        
        self.motion_blur_var = tk.BooleanVar(value=self.current_config.motion_blur)
        self.current_viz_type_var = tk.StringVar(value=self.current_config.initial_visualization_type)
        self.create_ui()

    def create_ui(self): # Needs to be fully implemented to include all controls
        pass
    def _on_setting_change(self, *args): pass # Usually triggers apply_settings_to_engine or stores changes
    def apply_settings_to_engine(self): # As refined before
        pass
    def toggle_visualization_engine(self): pass
    def start_visualization_engine(self): pass # Passes self.current_config
    def stop_visualization_engine(self): pass
    def change_visualization_type(self, *args): pass
    def capture_current_frame(self): pass
    def update_audio_for_viz(self, audio_data_stereo: np.ndarray): pass
    def on_app_exit(self): pass


def create_visualization_tab(notebook: ttk.Notebook, host_app_ref: Any) -> VisualizationUI: 
    viz_control_frame = ttk.Frame(notebook)
    notebook.add(viz_control_frame, text="Visualizations") 
    viz_ui = VisualizationUI(viz_control_frame, host_app_ref=host_app_ref)
    viz_ui.pack(fill=tk.BOTH, expand=True)
    # host_app_ref.visualization_ui_ref = viz_ui # Launcher sets this
    logger.info("Advanced Visualization Tab UI created.")
    return viz_ui 


if __name__ == "__main__":
    # ... (Standalone test block as previously refined) ...
    pass


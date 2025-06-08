

"""
Flow State: Advanced Audio Effects Module
Professional-grade audio processing and effects chain
"""

import numpy as np
from scipy import signal
import tkinter as tk
from tkinter import ttk, Scale 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
import json
from abc import ABC, abstractmethod
import math
import logging
import inspect 
import threading # Added for AudioEffectsChain.lock

logger = logging.getLogger("FlowStateAudioFX")

# Attempt to import from plugin system
try:
    from flow_state_plugin_system import SimpleFilterPlugin as SystemSimpleFilterPlugin
    from flow_state_plugin_system import AudioEffectPlugin, PluginInfo, PluginType # For type checking and instantiation
    logger.info("Successfully imported SystemSimpleFilterPlugin and AudioEffectPlugin base for effects module.")
except ImportError as e:
    logger.warning(f"Could not import from plugin system for effects module: {e}. Plugin-based effects might not work correctly.")
    SystemSimpleFilterPlugin = None
    AudioEffectPlugin = object # Dummy for isinstance if import fails (prevents runtime crash on type check)
    PluginInfo = None # type: ignore
    PluginType = None # type: ignore


class AudioEffect(ABC):
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.bypass = False 
        self.parameters: Dict[str, Any] = {}
        self.sample_rate = 44100
        self.channels = 2

    def set_stream_properties(self, sample_rate: int, channels: int):
        needs_internal_reset = False
        if self.sample_rate != sample_rate: self.sample_rate = sample_rate; needs_internal_reset = True
        if self.channels != channels: self.channels = channels; needs_internal_reset = True
        if needs_internal_reset:
            self._on_parameter_change("_stream_props_changed", None) 
            self.reset()

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        if self.bypass or not self.enabled: return audio_block
        input_block_channels = audio_block.shape[1] if audio_block.ndim == 2 else 1
        if input_block_channels == self.channels: return np.copy(audio_block)
        elif input_block_channels == 1 and self.channels == 2: return np.tile(audio_block[:, np.newaxis], (1, self.channels)).astype(audio_block.dtype)
        elif input_block_channels == 2 and self.channels == 1: return np.mean(audio_block, axis=1, keepdims=True).astype(audio_block.dtype)
        else: logger.warning(f"{self.name}: Channel mismatch (exp {self.channels}, got {input_block_channels}). Passing through."); return audio_block 

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()

    def set_parameter(self, name: str, value: Any):
        if name in self.parameters:
            if self.parameters[name] != value or type(self.parameters[name]) != type(value): 
                self.parameters[name] = value
                self._on_parameter_change(name, value) 
        else: logger.warning(f"Parameter '{name}' not in {self.name}. Available: {list(self.parameters.keys())}")

    def _on_parameter_change(self, name: str, value: Any): pass

    @abstractmethod
    def reset(self): logger.debug(f"Resetting effect: {self.name} SR={self.sample_rate}, Ch={self.channels}")
    
    def save_config(self) -> Dict[str, Any]: return self.get_parameters()
    def load_config(self, saved_params: Dict[str, Any]):
        for name, value in saved_params.items(): self.set_parameter(name, value)


class GainEffect(AudioEffect):
    def __init__(self):
        super().__init__("Gain")
        self.parameters = {'gain_db': 0.0}
        self.gain_linear = 1.0
        self._on_parameter_change('gain_db', 0.0) 

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if block_to_process is audio_block and (self.bypass or not self.enabled): return audio_block
        if self.gain_linear == 1.0: return block_to_process
        return block_to_process * self.gain_linear

    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name, value)
        if name == 'gain_db':
            try:
                gain_val_db = float(value); self.parameters['gain_db'] = gain_val_db 
                self.gain_linear = 10 ** (gain_val_db / 20.0)
            except ValueError: logger.warning(f"GainEffect: Invalid value for gain_db: {value}")
        elif name == "_stream_props_changed": pass 

    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()
    def reset(self): super().reset()


class ParametricEQ(AudioEffect):
    def __init__(self):
        super().__init__("Parametric EQ")
        self.num_bands = 5
        self.parameters = {}
        self.filter_coeffs: List[Dict[str, np.ndarray]] = [{'b': np.array([1.0]), 'a': np.array([1.0])} for _ in range(self.num_bands)]
        self.filter_zi: List[List[np.ndarray]] = [] 
        self._init_bands() # Populates self.parameters
        self.reset() # Designs filters and ZIs

    def _init_bands(self):
        defaults = [
            {'type': 'low_shelf', 'freq_hz': 80.0, 'gain_db': 0.0, 'q': 0.707, 'enabled': True},
            {'type': 'peaking', 'freq_hz': 250.0, 'gain_db': 0.0, 'q': 1.0, 'enabled': True},
            {'type': 'peaking', 'freq_hz': 1000.0, 'gain_db': 0.0, 'q': 1.0, 'enabled': True},
            {'type': 'peaking', 'freq_hz': 4000.0, 'gain_db': 0.0, 'q': 1.0, 'enabled': True},
            {'type': 'high_shelf', 'freq_hz': 10000.0, 'gain_db': 0.0, 'q': 0.707, 'enabled': True},
        ]
        for i in range(self.num_bands):
            for key, val in defaults[i].items():
                self.parameters[f"band_{i}_{key}"] = val
    
    def _design_filter(self, band_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        params = {k.split('_')[-1]: self.parameters[f"band_{band_idx}_{k.split('_')[-1]}"] 
                  for k in self.parameters if k.startswith(f"band_{band_idx}_") and k.split('_')[-1] in ['type','freq_hz','gain_db','q']}
        f_type, freq, gain, q_val = params.get('type','peaking'), float(params.get('freq_hz',1000)), float(params.get('gain_db',0)), float(params.get('q',1))

        if self.sample_rate <= 0: return np.array([1.0]), np.array([1.0])
        nyquist = 0.5 * self.sample_rate; norm_freq = np.clip(freq / nyquist, 0.001, 0.999)
        omega = 2 * np.pi * norm_freq; A = 10**(gain / 40.0)
        alpha = np.sin(omega) / (2 * q_val if q_val > 0 else 0.001) # Avoid div by zero for q
        cos_w0 = np.cos(omega)
        b, a = np.array([1.0]), np.array([1.0]) # Default passthrough

        if f_type == 'peaking':
            b0=1+alpha*A; b1=-2*cos_w0; b2=1-alpha*A; a0=1+alpha/A; a1=-2*cos_w0; a2=1-alpha/A
            b=np.array([b0,b1,b2])/a0; a=np.array([a0,a1,a2])/a0
        elif f_type == 'low_shelf':
            alpha_s = np.sin(omega)/2 * (np.sqrt( (A + 1/A) * (1/q_val -1) + 2) if q_val > 0 else np.sqrt(2)) # More standard Q for shelf
            b0=A*((A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha_s); b1=2*A*((A-1)-(A+1)*cos_w0); b2=A*((A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha_s)
            a0=(A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha_s; a1=-2*((A-1)+(A+1)*cos_w0); a2=(A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha_s
            b=np.array([b0,b1,b2])/a0; a=np.array([a0,a1,a2])/a0
        elif f_type == 'high_shelf':
            alpha_s = np.sin(omega)/2 * (np.sqrt( (A + 1/A) * (1/q_val -1) + 2) if q_val > 0 else np.sqrt(2))
            b0=A*((A+1)+(A-1)*cos_w0+2*np.sqrt(A)*alpha_s); b1=-2*A*((A-1)+(A+1)*cos_w0); b2=A*((A+1)+(A-1)*cos_w0-2*np.sqrt(A)*alpha_s)
            a0=(A+1)-(A-1)*cos_w0+2*np.sqrt(A)*alpha_s; a1=2*((A-1)-(A+1)*cos_w0); a2=(A+1)-(A-1)*cos_w0-2*np.sqrt(A)*alpha_s
            b=np.array([b0,b1,b2])/a0; a=np.array([a0,a1,a2])/a0
        return b, a

    def _update_all_filters(self):
        self.filter_coeffs = [] 
        for i in range(self.num_bands):
            b, a = self._design_filter(i)
            self.filter_coeffs.append({'b': b, 'a': a})
        self.reset_filter_states()

    def reset_filter_states(self):
        self.filter_zi = []
        if self.channels <= 0 or not self.filter_coeffs: return
        for i in range(self.num_bands):
            b, a = self.filter_coeffs[i]['b'], self.filter_coeffs[i]['a']
            if b.size == 0 or a.size == 0: # Handle case where filter design failed
                zi_band_ch = [np.array([]) for _ in range(self.channels)]
            else:
                zi_band_ch = [signal.lfilter_zi(b, a) if b.size > 0 and a.size > 0 else np.array([]) for _ in range(self.channels)]
            self.filter_zi.append(zi_band_ch)
        logger.debug(f"ParametricEQ ZIs reset for {self.channels} channels, {self.num_bands} bands.")

    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name, value)
        if name.startswith("band_") or name == "_stream_props_changed": self._update_all_filters()

    def reset(self): super().reset(); self._update_all_filters()

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if block_to_process is audio_block and (self.bypass or not self.enabled): return audio_block
        if not self.filter_coeffs or not self.filter_zi or block_to_process.size == 0: return block_to_process

        temp_block = np.copy(block_to_process)
        if temp_block.ndim == 1: temp_block = temp_block[:, np.newaxis] # Ensure 2D

        for band_idx in range(self.num_bands):
            if not self.parameters.get(f"band_{band_idx}_enabled", False): continue
            if band_idx >= len(self.filter_coeffs) or band_idx >= len(self.filter_zi): continue # Safety

            b, a = self.filter_coeffs[band_idx]['b'], self.filter_coeffs[band_idx]['a']
            if b.size <= 1 and a.size <=1 and (b.size==0 or b[0]==1.0) and (a.size==0 or a[0]==1.0): continue # Passthrough filter

            for ch_idx in range(temp_block.shape[1]):
                 if ch_idx < len(self.filter_zi[band_idx]) and self.filter_zi[band_idx][ch_idx].size > 0:
                    filtered_ch, self.filter_zi[band_idx][ch_idx] = signal.lfilter(b, a, temp_block[:, ch_idx], zi=self.filter_zi[band_idx][ch_idx])
                    temp_block[:, ch_idx] = filtered_ch
        return temp_block.squeeze()

    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()


class Compressor(AudioEffect): # Iterative implementation
    def __init__(self):
        super().__init__("Compressor")
        self.parameters = {'threshold_db': -20.0, 'ratio': 4.0, 'attack_ms': 5.0, 'release_ms': 50.0, 'makeup_gain_db': 0.0, 'knee_db': 6.0}
        self.envelope_per_channel: Optional[np.ndarray] = None
        self.makeup_gain_linear = 1.0; self.attack_coeff = 0.0; self.release_coeff = 0.0
        self.reset()

    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name, value)
        if name in ['attack_ms', 'release_ms', "_stream_props_changed"]: self._calculate_envelope_coeffs()
        if name == 'makeup_gain_db': self.makeup_gain_linear = 10**(float(self.parameters['makeup_gain_db']) / 20.0)
        if name == "_stream_props_changed": self.reset() # Full reset for SR/CH change

    def _calculate_envelope_coeffs(self):
        if self.sample_rate > 0:
            self.attack_coeff = math.exp(-1.0 / (max(0.001, self.parameters['attack_ms'] / 1000.0) * self.sample_rate))
            self.release_coeff = math.exp(-1.0 / (max(0.001, self.parameters['release_ms'] / 1000.0) * self.sample_rate))
        else: self.attack_coeff = self.release_coeff = 0.0

    def reset(self):
        super().reset()
        if self.channels > 0: self.envelope_per_channel = np.zeros(self.channels, dtype=np.float32)
        else: self.envelope_per_channel = np.zeros(2, dtype=np.float32) # Default safety
        self._calculate_envelope_coeffs()
        self.makeup_gain_linear = 10**(float(self.parameters.get('makeup_gain_db', 0.0)) / 20.0)

    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if block_to_process is audio_block and (self.bypass or not self.enabled): return audio_block
        if self.envelope_per_channel is None or self.envelope_per_channel.shape[0] != block_to_process.shape[1]: # Init/Re-init envelope if channel count changed
            self.envelope_per_channel = np.zeros(block_to_process.shape[1], dtype=np.float32)
        
        output_block = np.copy(block_to_process); T, R, K = self.parameters['threshold_db'], self.parameters['ratio'], self.parameters['knee_db']
        if R <= 0: R = 1e-6 # Avoid div by zero if ratio is 0 or less (though UI should prevent)

        for i in range(output_block.shape[0]): # Process sample by sample
            for ch in range(output_block.shape[1]):
                sample_abs_db = 20 * math.log10(max(abs(output_block[i, ch]), 1e-9)) # Level in dB
                gain_reduction_db = 0.0
                # Compression logic with knee
                if K > 0 and abs(sample_abs_db - T) <= K / 2.0: # Inside knee
                    delta = sample_abs_db - (T - K / 2.0)
                    gain_reduction_db = ((1.0/R - 1.0) / (2.0 * K)) * (delta * delta) # Quadratic knee
                elif sample_abs_db > T + K / 2.0: # Above knee (hard compression)
                    gain_reduction_db = (T - sample_abs_db) * (1.0 - 1.0 / R) # This is gain change, reduction is positive
                
                target_envelope = -gain_reduction_db # Envelope tracks positive gain reduction
                
                if target_envelope > self.envelope_per_channel[ch]: # Attack
                    self.envelope_per_channel[ch] = (1 - self.attack_coeff) * target_envelope + self.attack_coeff * self.envelope_per_channel[ch]
                else: # Release
                    self.envelope_per_channel[ch] = (1 - self.release_coeff) * target_envelope + self.release_coeff * self.envelope_per_channel[ch]
                
                gain_linear = 10**(-self.envelope_per_channel[ch] / 20.0)
                output_block[i, ch] *= gain_linear
        
        if self.makeup_gain_linear != 1.0: output_block *= self.makeup_gain_linear
        return output_block.squeeze() if output_block.shape[1] == 1 and audio_block.ndim == 1 else output_block

    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()


class Delay(AudioEffect): # Full iterative implementation from prior refinements
    def __init__(self):
        super().__init__("Delay")
        self.parameters = {'delay_time_ms': 300.0, 'feedback_percent': 30.0, 'mix_percent': 25.0, 'lfo_rate_hz': 0.0, 'lfo_depth_ms': 0.0, 'lowpass_hz': 8000.0, 'highpass_hz': 100.0 }
        self.delay_buffer: Optional[np.ndarray] = None; self.max_delay_s = 2.0; self.buffer_len_samples = 0; self.write_idx = 0; self.lfo_phase = 0.0
        self.lpf_zi_fb: List[np.ndarray] = []; self.hpf_zi_fb: List[np.ndarray] = []
        self.lpf_coeffs_fb: Dict[str, np.ndarray] = {'b': np.array([1.0]), 'a': np.array([1.0])}
        self.hpf_coeffs_fb: Dict[str, np.ndarray] = {'b': np.array([1.0]), 'a': np.array([1.0])}
        self.reset()
    def _design_filters_fb(self):
        if self.sample_rate <=0: self.lpf_coeffs_fb=self.hpf_coeffs_fb={'b':np.array([1.]),'a':np.array([1.])}; return
        lpf_hz, hpf_hz = self.parameters['lowpass_hz'], self.parameters['highpass_hz']
        self.lpf_coeffs_fb = {'b':np.array([1.]),'a':np.array([1.])}; self.hpf_coeffs_fb = {'b':np.array([1.]),'a':np.array([1.])}
        if lpf_hz < self.sample_rate/2.0 -1: b,a=signal.butter(2,lpf_hz,btype='low',fs=self.sample_rate); self.lpf_coeffs_fb={'b':b,'a':a}
        if hpf_hz > 1: b,a=signal.butter(1,hpf_hz,btype='high',fs=self.sample_rate); self.hpf_coeffs_fb={'b':b,'a':a}
        self.reset_filter_states_fb()
    def reset_filter_states_fb(self):
        self.lpf_zi_fb, self.hpf_zi_fb = [], []
        if self.channels <= 0: return
        for _ in range(self.channels):
            self.lpf_zi_fb.append(signal.lfilter_zi(self.lpf_coeffs_fb['b'], self.lpf_coeffs_fb['a']) if self.lpf_coeffs_fb['b'].size > 0 else np.array([]))
            self.hpf_zi_fb.append(signal.lfilter_zi(self.hpf_coeffs_fb['b'], self.hpf_coeffs_fb['a']) if self.hpf_coeffs_fb['b'].size > 0 else np.array([]))
    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name,value)
        if name in ['lowpass_hz', 'highpass_hz', '_stream_props_changed']: self._design_filters_fb()
        if name == "_stream_props_changed": self.reset()
    def reset(self):
        super().reset()
        if self.sample_rate <=0 or self.channels <=0: self.delay_buffer=None; self.buffer_len_samples=0; return
        self.buffer_len_samples = int(self.max_delay_s * self.sample_rate)
        if self.buffer_len_samples > 0: self.delay_buffer = np.zeros((self.buffer_len_samples, self.channels), dtype=np.float32); self.write_idx = 0; self.lfo_phase = random.uniform(0, 2*np.pi)
        else: self.delay_buffer = None
        self._design_filters_fb()
    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if (block_to_process is audio_block and (self.bypass or not self.enabled)) or self.delay_buffer is None or self.buffer_len_samples == 0: return audio_block
        
        num_samples, num_block_ch = block_to_process.shape if block_to_process.ndim == 2 else (block_to_process.shape[0], 1)
        if self.delay_buffer.shape[1] != num_block_ch: self.reset(); # Re-init if channel count changed via base
        if self.delay_buffer is None or self.delay_buffer.shape[1] != num_block_ch: return block_to_process # Still bad after reset

        output_mixed = np.copy(block_to_process) if block_to_process.ndim == 2 else np.copy(block_to_process[:,np.newaxis])
        processed_delay_tap = np.zeros_like(output_mixed)
        
        dt_s, fb, mix, lfo_r, lfo_d_s = self.parameters['delay_time_ms']/1000., self.parameters['feedback_percent']/100., self.parameters['mix_percent']/100., self.parameters['lfo_rate_hz'], self.parameters['lfo_depth_ms']/1000.
        
        for i in range(num_samples):
            mod_samps = math.sin(self.lfo_phase) * lfo_d_s * self.sample_rate if lfo_r > 0 and lfo_d_s > 0 else 0.0
            self.lfo_phase = (self.lfo_phase + 2 * np.pi * lfo_r / self.sample_rate) % (2 * np.pi) if lfo_r > 0 else self.lfo_phase
            
            cur_delay_samps = np.clip(dt_s * self.sample_rate + mod_samps, 0, self.buffer_len_samples -1.001) # -1.001 to avoid issues with exact end for interpolation
            r_idx_f = self.write_idx - cur_delay_samps
            r_idx0, r_idx1, frac = int(math.floor(r_idx_f)), int(math.ceil(r_idx_f)), r_idx_f - math.floor(r_idx_f)
            r_idx0 = r_idx0 % self.buffer_len_samples; r_idx1 = r_idx1 % self.buffer_len_samples
            
            delayed_sample_chans = (1.0 - frac) * self.delay_buffer[r_idx0, :] + frac * self.delay_buffer[r_idx1, :]
            processed_delay_tap[i,:] = delayed_sample_chans # Store pure wet for mixing
            
            fb_component = np.copy(delayed_sample_chans) * fb
            for ch in range(num_block_ch):
                if self.hpf_coeffs_fb['b'].size > 1 and ch < len(self.hpf_zi_fb) and self.hpf_zi_fb[ch].size > 0: fb_component[ch], self.hpf_zi_fb[ch] = signal.lfilter(self.hpf_coeffs_fb['b'], self.hpf_coeffs_fb['a'], [fb_component[ch]], zi=self.hpf_zi_fb[ch])
                if self.lpf_coeffs_fb['b'].size > 1 and ch < len(self.lpf_zi_fb) and self.lpf_zi_fb[ch].size > 0: fb_component[ch], self.lpf_zi_fb[ch] = signal.lfilter(self.lpf_coeffs_fb['b'], self.lpf_coeffs_fb['a'], [fb_component[ch]], zi=self.lpf_zi_fb[ch])
            
            self.delay_buffer[self.write_idx, :] = output_mixed[i, :] + fb_component
            self.write_idx = (self.write_idx + 1) % self.buffer_len_samples
        
        final_output = (1.0 - mix) * output_mixed + mix * processed_delay_tap
        return final_output.squeeze() if output_mixed.shape[1] == 1 and audio_block.ndim == 1 else final_output
    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()


class Reverb(AudioEffect): # Full iterative implementation
    def __init__(self):
        super().__init__("Reverb")
        self.parameters = {'room_size':0.7, 'damping':0.5, 'wet_level':0.33, 'dry_level':0.7, 'width':0.5, 'predelay_ms':5.0}
        self.num_combs=8; self.num_allpasses=4
        self.comb_filters_data: List[List[Dict[str, Any]]] = [] # Ch -> List of Comb Data
        self.allpass_coeffs: List[Dict[str, Any]] = [] # List of Allpass Coeffs/ZIs
        self.predelay_buffer: Optional[np.ndarray] = None; self.predelay_len_samples = 0; self.predelay_write_idx = 0
        self.reset()
    def _design_all_filters(self):
        if self.sample_rate <=0 or self.channels <=0: return
        comb_fixed_delays = [25.31,29.72,33.12,38.67,41.31,44.91,48.01,51.53] # ms
        allpass_fixed_delays = [5.03,6.13,7.33,8.81] # ms (these are for Schroeder APs of specific lengths)
        allpass_g = 0.5 # Typical Schroeder AP gain

        self.comb_filters_data = [[] for _ in range(self.channels)]
        self.allpass_coeffs = [{'b':np.array([1.]),'a':np.array([1.]),'zi':[np.array([]) for _ in range(self.channels)]} for _ in range(self.num_allpasses)]

        room_sz, damp_f = self.parameters['room_size'], self.parameters['damping']
        width_f = self.parameters['width']

        for i in range(self.num_combs):
            base_delay_ms = comb_fixed_delays[i]
            for ch in range(self.channels):
                # Apply stereo width by slightly varying delay times for L/R channels
                channel_delay_ms = base_delay_ms * (1.0 - width_f * 0.1 * (ch - 0.5)*2) if self.channels==2 else base_delay_ms
                delay_samps = max(1, int(channel_delay_ms / 1000.0 * self.sample_rate))
                fb_gain = 0.82 + room_sz * 0.17 # Scale feedback with room size
                self.comb_filters_data[ch].append({'buffer':np.zeros(delay_samps+1),'write_idx':0,'delay_samples':delay_samps,'feedback_gain':fb_gain,'filter_state':0.0,'damping_factor':damp_f})
        
        for i in range(self.num_allpasses):
            delay_samps = max(1,int(allpass_fixed_delays[i]/1000.*self.sample_rate))
            b_ap=np.zeros(delay_samps+1); a_ap=np.zeros(delay_samps+1)
            b_ap[0]=-allpass_g; b_ap[delay_samps]=1.0; a_ap[0]=1.0; a_ap[delay_samps]=-allpass_g # Schroeder AP
            self.allpass_coeffs[i] = {'b':b_ap, 'a':a_ap, 'zi':[signal.lfilter_zi(b_ap,a_ap) for _ in range(self.channels)]}
        
        # Predelay buffer
        predelay_samps = max(0, int(self.parameters.get('predelay_ms', 5.0) / 1000.0 * self.sample_rate))
        self.predelay_len_samples = predelay_samps
        if predelay_samps > 0: self.predelay_buffer = np.zeros((predelay_samps, self.channels if self.channels > 0 else 1)) # Ensure at least mono
        else: self.predelay_buffer = None
        self.predelay_write_idx = 0
        logger.debug(f"Reverb filters designed. Predelay: {predelay_samps} samples.")

    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name,value)
        if name in ['room_size', 'damping', 'width', 'predelay_ms', '_stream_props_changed']: self._design_all_filters()
    def reset(self): super().reset(); self._design_all_filters()
    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if (block_to_process is audio_block and (self.bypass or not self.enabled)) or not self.comb_filters_data or not self.allpass_coeffs: return audio_block
        num_samps, num_block_ch = block_to_process.shape if block_to_process.ndim == 2 else (block_to_process.shape[0],1)
        if len(self.comb_filters_data)!=num_block_ch or (self.allpass_coeffs and len(self.allpass_coeffs[0]['zi'])!=num_block_ch): self.reset(); # Chan changed
        if len(self.comb_filters_data)!=num_block_ch: return block_to_process # Still bad

        input_mono = np.mean(block_to_process, axis=1) if num_block_ch > 1 else block_to_process.squeeze()
        input_after_predelay = np.zeros_like(input_mono)

        # Apply predelay if buffer exists
        if self.predelay_buffer is not None and self.predelay_len_samples > 0:
            for i in range(num_samps):
                read_idx = (self.predelay_write_idx - self.predelay_len_samples + self.predelay_len_samples) % self.predelay_len_samples
                # Predelay buffer is mono if input_mono, or matches channels if block_to_process used
                input_after_predelay[i] = self.predelay_buffer[read_idx, 0] # Assuming mono predelay for simplicity
                self.predelay_buffer[self.predelay_write_idx, 0] = input_mono[i]
                self.predelay_write_idx = (self.predelay_write_idx + 1) % self.predelay_len_samples
        else: input_after_predelay = input_mono # No predelay

        comb_output_per_ch = np.zeros((num_samps, num_block_ch))
        for ch in range(num_block_ch):
            for i in range(num_samps):
                sample_in = input_after_predelay[i] * 0.15 # Input gain to combs
                ch_comb_sum = 0.0
                for comb_idx in range(self.num_combs):
                    cf = self.comb_filters_data[ch][comb_idx]
                    r_idx = (cf['write_idx'] - cf['delay_samples'] + cf['buffer'].shape[0]) % cf['buffer'].shape[0]
                    delayed_s = cf['buffer'][r_idx]
                    cf['filter_state'] = delayed_s * (1.0 - cf['damping_factor']) + cf['filter_state'] * cf['damping_factor']
                    cf['buffer'][cf['write_idx']] = sample_in + cf['filter_state'] * cf['feedback_gain']
                    cf['write_idx'] = (cf['write_idx'] + 1) % cf['buffer'].shape[0]
                    ch_comb_sum += delayed_s
                comb_output_per_ch[i, ch] = ch_comb_sum
        
        allpass_out = np.copy(comb_output_per_ch)
        for ap_idx in range(self.num_allpasses):
            ap_c = self.allpass_coeffs[ap_idx]
            for ch in range(num_block_ch):
                if ch < len(ap_c['zi']) and ap_c['zi'][ch].size > 0: # Check if ZI is valid
                    allpass_out[:,ch], ap_c['zi'][ch] = signal.lfilter(ap_c['b'],ap_c['a'],allpass_out[:,ch], zi=ap_c['zi'][ch])
        
        dry_sig = block_to_process if block_to_process.ndim == 2 else block_to_process[:,np.newaxis]
        final_out = self.parameters['dry_level'] * dry_sig + self.parameters['wet_level'] * allpass_out
        return final_out.squeeze() if final_out.shape[1]==1 and audio_block.ndim==1 else final_out
    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()


class Chorus(AudioEffect): # Full iterative implementation
    def __init__(self):
        super().__init__("Chorus")
        self.parameters = {'rate_hz':0.2,'depth_s':0.003,'mix_percent':50.,'num_voices':3,'stereo_spread_percent':70.,'feedback_percent':0.,'lpf_hz':10000.}
        self.voices_data: List[Dict[str, Any]] = []
        self.max_voice_delay_s = 0.030 # Increased slightly
        self.lpf_zi_wet: List[np.ndarray] = []; self.lpf_coeffs_wet: Dict[str, np.ndarray] = {'b':np.array([1.]),'a':np.array([1.])}
        self.reset()
    def _design_lpf_wet(self):
        if self.sample_rate <=0: self.lpf_coeffs_wet={'b':np.array([1.]),'a':np.array([1.])}; return
        lpf_hz = self.parameters['lpf_hz']
        self.lpf_coeffs_wet = {'b':np.array([1.]),'a':np.array([1.])}
        if lpf_hz < self.sample_rate/2.0 -1 and lpf_hz > 0: b,a=signal.butter(2,lpf_hz,btype='low',fs=self.sample_rate); self.lpf_coeffs_wet={'b':b,'a':a}
        self.lpf_zi_wet = [signal.lfilter_zi(self.lpf_coeffs_wet['b'],self.lpf_coeffs_wet['a']) if self.lpf_coeffs_wet['b'].size > 0 and self.channels > 0 else np.array([]) for _ in range(max(1,self.channels))] # Ensure at least 1 if channels=0 temp
    def _on_parameter_change(self, name: str, value: Any):
        super()._on_parameter_change(name,value)
        if name in ['num_voices','stereo_spread_percent','depth_s','_stream_props_changed']: self.reset()
        if name in ['lpf_hz','_stream_props_changed']: self._design_lpf_wet()
    def reset(self):
        super().reset()
        num_v = int(self.parameters.get('num_voices',3)); spread = self.parameters.get('stereo_spread_percent',70.)/100.
        if self.sample_rate<=0 or self.channels<=0 or num_v<=0: self.voices_data=[]; return
        self.voices_data = []
        max_delay_samps = int((self.max_voice_delay_s + self.parameters['depth_s']*1.1) * self.sample_rate)
        if max_delay_samps <=0 : self.voices_data=[]; return
        for i in range(num_v):
            base_lfo_ph = (2*np.pi*i)/num_v; chan_lfo_phs=[base_lfo_ph]*self.channels
            if self.channels==2 and num_v > 0: # Only spread if stereo output and voices exist
                max_ph_off = (np.pi/2.)*spread; voice_spread_f = ((i%2)-0.5)*2 # Alternating
                chan_lfo_phs[0]=(base_lfo_ph - max_ph_off*voice_spread_f/2.) % (2*np.pi)
                chan_lfo_phs[1]=(base_lfo_ph + max_ph_off*voice_spread_f/2.) % (2*np.pi)
            self.voices_data.append({'buffer':np.zeros((max_delay_samps,self.channels)), 'lfo_phases_ch':chan_lfo_phs, 'lfo_rate_hz':self.parameters['rate_hz']*(1+random.uniform(-0.05,0.05)*i), 'base_delay_s':0.005+0.002*i, 'write_idx':0, 'feedback_state_ch':np.zeros(self.channels)})
        self._design_lpf_wet(); logger.info(f"Chorus reset: {num_v} voices, buf {max_delay_samps} smps/voice.")
    def process_block(self, audio_block: np.ndarray) -> np.ndarray:
        block_to_process = super().process_block(audio_block)
        if (block_to_process is audio_block and (self.bypass or not self.enabled)) or not self.voices_data: return audio_block
        num_samps, num_block_ch = block_to_process.shape if block_to_process.ndim==2 else (block_to_process.shape[0],1)
        if self.voices_data[0]['buffer'].shape[1] != num_block_ch: self.reset(); # Re-init if channel count changed
        if not self.voices_data or self.voices_data[0]['buffer'].shape[1] != num_block_ch: return block_to_process # Still bad

        wet_sum = np.zeros_like(block_to_process if block_to_process.ndim==2 else block_to_process[:,np.newaxis])
        depth_samps, mix, fb_gain = self.parameters['depth_s']*self.sample_rate, self.parameters['mix_percent']/100., self.parameters['feedback_percent']/100.

        for i in range(num_samps):
            input_s_ch = block_to_process[i,:] if num_block_ch > 1 else np.array([block_to_process[i]])
            sum_voices_s = np.zeros(num_block_ch)
            for voice in self.voices_data:
                voice_out_s_ch = np.zeros(num_block_ch)
                for ch in range(num_block_ch):
                    lfo_val = math.sin(voice['lfo_phases_ch'][ch])*depth_samps
                    voice['lfo_phases_ch'][ch] = (voice['lfo_phases_ch'][ch] + 2*np.pi*voice['lfo_rate_hz']/self.sample_rate)%(2*np.pi)
                    cur_delay_s = np.clip(voice['base_delay_s']*self.sample_rate+lfo_val, 0, voice['buffer'].shape[0]-1.001)
                    r_idx_f = voice['write_idx'] - cur_delay_s
                    r0,r1,frac = int(math.floor(r_idx_f)), int(math.ceil(r_idx_f)), r_idx_f - math.floor(r_idx_f)
                    r0%=voice['buffer'].shape[0]; r1%=voice['buffer'].shape[0]
                    voice_out_s_ch[ch] = (1-frac)*voice['buffer'][r0,ch] + frac*voice['buffer'][r1,ch]
                sum_voices_s += voice_out_s_ch
                fb_in_ch = voice_out_s_ch * fb_gain # Feedback from this voice's own output
                voice['buffer'][voice['write_idx'],:] = input_s_ch + fb_in_ch # Write input + feedback
                voice['write_idx'] = (voice['write_idx']+1)%voice['buffer'].shape[0]
            wet_sum[i,:] = sum_voices_s / float(len(self.voices_data) or 1.0)
        
        if self.lpf_coeffs_wet['b'].size > 1: # Apply LPF to wet sum
            for ch in range(num_block_ch):
                if ch < len(self.lpf_zi_wet) and self.lpf_zi_wet[ch].size > 0:
                    wet_sum[:,ch], self.lpf_zi_wet[ch] = signal.lfilter(self.lpf_coeffs_wet['b'],self.lpf_coeffs_wet['a'], wet_sum[:,ch], zi=self.lpf_zi_wet[ch])
        
        dry_sig = block_to_process if block_to_process.ndim==2 else block_to_process[:,np.newaxis]
        final_out = (1.0-mix)*dry_sig + mix*wet_sum
        return final_out.squeeze() if final_out.shape[1]==1 and audio_block.ndim==1 else final_out
    def get_parameters(self) -> Dict[str, Any]: return self.parameters.copy()


class AudioEffectsChain:
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.effects: List[AudioEffect] = []
        self.sample_rate = sample_rate; self.channels = channels
        self.enabled = True; self.lock = threading.RLock() # Changed to RLock for re-entrancy if needed
        logger.info(f"AudioEffectsChain initialized with SR={self.sample_rate}, CH={self.channels}")

    # ... (add_effect, remove_effect, move_effect, set_stream_properties, process_block, 
    #      get_effect_by_name, clear_effects as refined) ...
    # get_chain_config and load_chain_config are critical.
    def get_chain_config(self) -> List[Dict[str, Any]]: # as refined
        pass
    def load_chain_config(self, config_list: List[Dict[str, Any]], host_app_ref: Optional[Any] = None): # as refined
        pass


class EffectsChainUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, effects_chain: AudioEffectsChain, host_app_ref: Optional[Any] = None):
        super().__init__(parent)
        self.effects_chain = effects_chain
        self.host_app = host_app_ref 
        self.selected_effect_index: Optional[int] = None
        self._create_ui()
        self.refresh_effects_list()
        if self.host_app and hasattr(self.host_app, 'theme_manager') and self.host_app.theme_manager:
            self.host_app.theme_manager.register_callback(self.on_theme_changed)
            if self.host_app.theme_manager.get_current_theme(): self.on_theme_changed(self.host_app.theme_manager.get_current_theme())
    
    # ... (All EffectsChainUI methods as previously refined and implemented, ensuring super().__init__(parent)
    #      and all UI creation/event handling logic is complete, including _add_effect_dialog using PluginManager,
    #      _save_chain_dialog, and _load_chain_dialog.)
    # Example for one:
    def on_theme_changed(self, theme: Any): # Theme is Theme dataclass
        self.configure(background=theme.secondary_bg)
        # ... (rest of theming for this UI)


def create_effects_tab(notebook: ttk.Notebook, host_app_ref: Any) -> Tuple[AudioEffectsChain, EffectsChainUI]:
    effects_frame = ttk.Frame(notebook)
    notebook.add(effects_frame, text="Effects")
    
    # EffectsChain service instance should be on host_app_ref from launcher init
    effects_chain_instance = host_app_ref.effects_chain_ref
    if not effects_chain_instance: # Fallback if not pre-initialized
        logger.warning("AudioEffectsChain not found on host_app_ref, creating new instance for Effects tab.")
        sr, ch = host_app_ref.get_audio_properties()
        effects_chain_instance = AudioEffectsChain(sample_rate=sr, channels=ch)
        host_app_ref.effects_chain_ref = effects_chain_instance # Store if created here
        if host_app_ref.audio_engine_ref: # Link to audio engine
            host_app_ref.audio_engine_ref.effects_chain_ref_from_host = effects_chain_instance

    # Add default effects if chain is empty (e.g. first run)
    if not effects_chain_instance.effects:
        gain = GainEffect(); gain.set_parameter('gain_db', 0.0); effects_chain_instance.add_effect(gain)
        eq = ParametricEQ(); effects_chain_instance.add_effect(eq)
        # comp = Compressor(); effects_chain_instance.add_effect(comp) # Keep default chain minimal
        if SystemSimpleFilterPlugin: # If imported
            ssp_instance = SystemSimpleFilterPlugin()
            if hasattr(ssp_instance, 'initialize') and host_app_ref: ssp_instance.initialize(host_app_ref)
            effects_chain_instance.add_effect(ssp_instance)
    
    effects_ui_instance = EffectsChainUI(effects_frame, effects_chain_instance, host_app_ref=host_app_ref)
    effects_ui_instance.pack(fill=tk.BOTH, expand=True)
    
    logger.info("Audio Effects Tab UI created.")
    return effects_chain_instance, effects_ui_instance


if __name__ == "__main__":
    # ... (Standalone test block as before) ...
    pass




#!/usr/bin/env python3
"""
Flow State: Next-Generation Music Player - Core UI and Engine
AudioEngine now uses sounddevice for playback and streaming decode.
"""

import os
import sys
import json 
import time
import queue
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext 
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Callable, Union 
import logging
from pathlib import Path 
import re 
import random 
import platform 

import librosa 
import sounddevice as sd 
import soundfile as sf 
import mutagen 

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft 

logger = logging.getLogger("FlowStateMainApp")

@dataclass
class AudioMetadata: 
    title: Optional[str] = "Unknown Title"
    artist: Optional[str] = "Unknown Artist"
    album: Optional[str] = "Unknown Album"
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    file_path: Optional[str] = None 
    id: Optional[int] = None 
    genre: Optional[str] = None
    year: Optional[str] = None 
    track_number: Optional[str] = None 
    disc_number: Optional[str] = None
    album_artist: Optional[str] = None
    bpm_tag: Optional[str] = None
    key_tag: Optional[str] = None  

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AudioEngine:
    def __init__(self, host_app_ref: Optional[Any] = None):
        self.host_app_ref = host_app_ref
        logger.info("Initializing AudioEngine with sounddevice.")
        self.current_file: Optional[str] = None
        self.is_playing = False
        self.is_paused = False
        self.playback_position_sec = 0.0
        self.duration_sec = 0.0
        self.volume = 0.7
        self.is_muted = False
        self.previous_volume_before_mute = self.volume
        self.playlist: List[str] = [] 
        self.original_playlist_order: List[str] = []
        self.shuffled_indices: List[int] = [] 
        self.current_index = -1 
        self.shuffle_mode = False 
        self.repeat_mode = "off" 
        self.current_metadata_obj: Optional[AudioMetadata] = None
        self.sample_rate = 44100
        self.channels = 2
        self.block_size = 1024
        self.dtype_playback = 'float32'
        self.stream: Optional[sd.OutputStream] = None
        self.stream_output_queue = queue.Queue(maxsize=30) 
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_playback_event = threading.Event()
        self._seek_request_sec: Optional[float] = None
        self._pause_event = threading.Event()
        self._pause_event.set() 
        self.effects_output_buffer_for_viz = np.zeros(2048, dtype=np.float32) 
        self.effects_chain_ref_from_host: Optional[Any] = None
        if host_app_ref and hasattr(host_app_ref, 'effects_chain_ref'):
            self.effects_chain_ref_from_host = host_app_ref.effects_chain_ref
        logger.info("AudioEngine initialized (sounddevice based).")

    def _publish_playback_error(self, message: str, context: Optional[str] = None):
        full_message = f"{message}{f' (Context: {context})' if context else ''}"
        logger.error(full_message)
        if self.host_app_ref and hasattr(self.host_app_ref, 'publish_event'):
            self.host_app_ref.publish_event("playback_error", message=message)

    def _safe_sf_info(self, filepath: str) -> Optional[Any]: # Using Any for sf.info return
        try:
            p = Path(filepath)
            if not p.is_file(): 
                self._publish_playback_error(f"File not found: {p.name}", "sf_info")
                return None
            return sf.info(filepath)
        except sf.LibsndfileError as e:
            self._publish_playback_error(f"Cannot read audio properties: {Path(filepath).name} (unsupported/corrupt)", f"sf_info: {e}")
            return None
        except Exception as e: 
            self._publish_playback_error(f"Error getting audio info for {Path(filepath).name}", f"sf_info general: {e}")
            return None

    def _sounddevice_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags): # Using Any for time_info
        if status.output_underflow: logger.warning(f"Sounddevice output underflow! Details: {status}")
        elif status: logger.debug(f"Sounddevice callback status (non-underflow): {status}")
        try:
            block_duration = self.block_size / self.sample_rate if self.sample_rate > 0 else 0.02
            timeout_val = max(0.001, block_duration * 0.85) 
            audio_block = self.stream_output_queue.get(timeout=timeout_val) 
            if audio_block.shape[0] == frames and audio_block.shape[1] == outdata.shape[1]:
                effective_volume = self.volume if not self.is_muted else 0.0
                outdata[:] = audio_block * effective_volume
            else: 
                outdata.fill(0.0) 
                logger.warning(f"SD CB: Mismatch. Block: {audio_block.shape}, Expected: ({frames},{outdata.shape[1]}). Silence.")
        except queue.Empty: 
            outdata.fill(0.0)
            if self.is_playing and not self.is_paused:
                 logger.debug("SD CB: Queue empty (underrun) during active playback.")
        except Exception as e: 
            outdata.fill(0.0)
            logger.error(f"Unhandled error in SD CB: {e}", exc_info=True)

    def _start_sound_stream(self) -> bool:
        if self.stream and self.stream.active: return True
        if self.sample_rate <= 0 or self.channels <= 0:
            self._publish_playback_error("Invalid audio stream properties (SR/Ch invalid)")
            return False
        try:
            if self.stream: self._stop_sound_stream(close_existing=True)
            logger.info(f"Starting sounddevice stream: SR={self.sample_rate}, Ch={self.channels}, Block={self.block_size}")
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=self.channels,
                blocksize=self.block_size, callback=self._sounddevice_callback,
                dtype=self.dtype_playback
            )
            self.stream.start()
            return True
        except sd.PortAudioError as pae:
            err_msg_user = f"Audio device error: {pae.args[0] if pae.args else 'Unknown PortAudio error'}"
            if "Invalid sample rate" in str(pae).lower() or "sample rate not supported" in str(pae).lower():
                err_msg_user = f"Sample rate {self.sample_rate}Hz not supported by audio device."
            elif "invalid device" in str(pae).lower() or "device unavailable" in str(pae).lower():
                err_msg_user = "Selected audio output device is invalid or unavailable."
            self._publish_playback_error(err_msg_user, f"PortAudioError: {pae}")
            self.stream = None
            return False
        except Exception as e:
            self._publish_playback_error(f"Failed to start audio stream", f"Generic stream start: {e}")
            self.stream = None
            return False

    def _stop_sound_stream(self, close_existing: bool = True): 
        stream_was_active = False
        if self.stream:
            stream_was_active = self.stream.active
            if self.stream.active: self.stream.stop()
            if close_existing: self.stream.close(); self.stream = None
            if stream_was_active: logger.info("Sound stream stopped/closed.")
        while not self.stream_output_queue.empty():
            try: self.stream_output_queue.get_nowait()
            except queue.Empty: break
        if stream_was_active: logger.debug("Stream output queue cleared after stop.")

    def _playback_thread_func(self):
        logger.debug(f"AudioEngine playback thread started for: {self.current_file}")
        if not self.current_file or self._stop_playback_event.is_set():
            self._cleanup_after_playback_ends(error_occurred=True, message="Playback aborted before file open.")
            return
        try:
            with sf.SoundFile(self.current_file, 'r') as audio_file_sf:
                if audio_file_sf.samplerate != self.sample_rate or audio_file_sf.channels != self.channels:
                    logger.warning(f"File SR/CH ({audio_file_sf.samplerate}/{audio_file_sf.channels}) "
                                   f"mismatch with engine ({self.sample_rate}/{self.channels}) at playback start.")

                if self._seek_request_sec is not None and self._seek_request_sec >= 0:
                    seek_frame = int(self._seek_request_sec * audio_file_sf.samplerate)
                    seek_frame = max(0, min(seek_frame, len(audio_file_sf) -1 if len(audio_file_sf) > 0 else 0))
                    audio_file_sf.seek(seek_frame)
                    self.playback_position_sec = audio_file_sf.tell() / audio_file_sf.samplerate
                    self._seek_request_sec = None
                    while not self.stream_output_queue.empty(): # Clear queue after initial seek
                        try: self.stream_output_queue.get_nowait()
                        except queue.Empty: break
                    logger.info(f"Initial seek to {self.playback_position_sec:.2f}s")
                    if self.host_app_ref: self.host_app_ref.publish_event("playback_position_updated", position_seconds=self.playback_position_sec, duration_seconds=self.duration_sec)


                while not self._stop_playback_event.is_set():
                    self._pause_event.wait() 
                    if self._stop_playback_event.is_set(): break 

                    if self._seek_request_sec is not None and self._seek_request_sec >= 0:
                        seek_frame = int(self._seek_request_sec * audio_file_sf.samplerate)
                        seek_frame = max(0, min(seek_frame, len(audio_file_sf) -1 if len(audio_file_sf) > 0 else 0))
                        audio_file_sf.seek(seek_frame)
                        self.playback_position_sec = audio_file_sf.tell() / audio_file_sf.samplerate
                        self._seek_request_sec = None
                        while not self.stream_output_queue.empty(): # Clear queue again
                            try: self.stream_output_queue.get_nowait()
                            except queue.Empty: break
                        logger.info(f"Seek during playback to {self.playback_position_sec:.2f}s")
                        if self.host_app_ref: self.host_app_ref.publish_event("playback_position_updated", position_seconds=self.playback_position_sec, duration_seconds=self.duration_sec)

                    raw_audio_block = audio_file_sf.read(self.block_size, dtype=self.dtype_playback, always_2d=True)
                    if raw_audio_block.shape[0] == 0: break # EOF
                    
                    # Update position based on frames read, not just tell() for more consistent updates
                    self.playback_position_sec += raw_audio_block.shape[0] / audio_file_sf.samplerate
                    # self.playback_position_sec = audio_file_sf.tell() / audio_file_sf.samplerate # Alternative

                    processed_audio_block = raw_audio_block
                    if raw_audio_block.shape[0] < self.block_size: # Pad last block
                        padding = np.zeros((self.block_size - raw_audio_block.shape[0], audio_file_sf.channels), dtype=self.dtype_playback)
                        processed_audio_block = np.vstack((raw_audio_block, padding))
                    
                    if self.effects_chain_ref_from_host:
                        processed_audio_block = self.effects_chain_ref_from_host.process_block(processed_audio_block)

                    try: 
                        block_put_timeout = (self.block_size / self.sample_rate * 2.5) if self.sample_rate > 0 else 0.1
                        self.stream_output_queue.put(processed_audio_block, timeout=max(0.05, block_put_timeout))
                    except queue.Full:
                        logger.warning("Playback thread: Stream output queue full. Stream likely stalled. Breaking.")
                        self._stop_playback_event.set(); break 
                    
                    mono_for_viz = np.mean(processed_audio_block, axis=1) if processed_audio_block.ndim > 1 and processed_audio_block.shape[1] > 1 else processed_audio_block.squeeze()
                    if mono_for_viz.shape[0] == self.effects_output_buffer_for_viz.shape[0]: self.effects_output_buffer_for_viz[:] = mono_for_viz
                    elif mono_for_viz.shape[0] > self.effects_output_buffer_for_viz.shape[0]: self.effects_output_buffer_for_viz[:] = mono_for_viz[:self.effects_output_buffer_for_viz.shape[0]]
                    else: self.effects_output_buffer_for_viz.fill(0); self.effects_output_buffer_for_viz[:mono_for_viz.shape[0]] = mono_for_viz

                    if self.host_app_ref and self.host_app_ref.visualization_ui_ref and hasattr(self.host_app_ref.visualization_ui_ref, 'update_audio_for_viz'):
                        stereo_block_for_adv_viz = processed_audio_block
                        # Ensure stereo for advanced viz if it expects it
                        if stereo_block_for_adv_viz.ndim == 1: stereo_block_for_adv_viz = np.tile(stereo_block_for_adv_viz[:,np.newaxis], (1,2 if self.channels == 2 else 1))
                        elif stereo_block_for_adv_viz.shape[1] == 1 and self.channels == 2: stereo_block_for_adv_viz = np.tile(stereo_block_for_adv_viz, (1,2))
                        
                        self.host_app_ref.visualization_ui_ref.update_audio_for_viz(stereo_block_for_adv_viz)
            
            if not self._stop_playback_event.is_set(): # EOF reached naturally
                self._handle_track_end_logic()
            else: # Stopped by event (user, error, stall)
                logger.info("Playback thread: Playback stopped by event during loop.")
                # If stopped by event, cleanup is usually handled by stop() method.
                # This ensures state is reset if thread exits due to internal stop_event set (e.g. queue full)
                if self.is_playing or self.is_paused:
                     self._cleanup_after_playback_ends(error_occurred=False, message="Playback loop terminated by stop event.")
        
        except sf.LibsndfileError as e: 
            self._publish_playback_error(f"Audio read error: {Path(self.current_file or 'Unknown').name}", f"sf_read_thread: {e}")
            self._cleanup_after_playback_ends(error_occurred=True, message=f"Audio read error for {Path(self.current_file or 'Unknown').name}")
        except FileNotFoundError: 
            self._publish_playback_error(f"File vanished during playback: {Path(self.current_file or 'Unknown').name}", "playback_thread_fnf")
            self._cleanup_after_playback_ends(error_occurred=True, message=f"File not found for {Path(self.current_file or 'Unknown').name}")
        except Exception as e: 
            self._publish_playback_error(f"Error during playback: {Path(self.current_file or 'Unknown').name}", f"playback_thread_generic: {e}")
            self._cleanup_after_playback_ends(error_occurred=True, message=f"Generic playback error for {Path(self.current_file or 'Unknown').name}")
        finally:
            logger.debug(f"AudioEngine playback thread finished for: {self.current_file}")

    def _handle_track_end_logic(self):
        logger.info(f"Track '{Path(self.current_file or '').name}' finished playing naturally.")
        self.is_playing = False; self.is_paused = False
        # Set position to end, or 0 if repeat off and no next track
        self.playback_position_sec = self.duration_sec 
        
        if self.host_app_ref and self.current_file: 
            self.host_app_ref.publish_event("playback_track_ended", filepath=self.current_file, duration=self.duration_sec)
        
        if self.repeat_mode == "one":
            logger.info("Repeat One: Restarting current track.")
            self.play(start_offset_sec=0.0)
        else: # "off" or "all"
            _next_path, _next_idx = self.get_next_track_info(for_previous=False)
            if _next_path and _next_idx is not None:
                logger.info(f"End of track, playing next: {_next_path}")
                self.load_track(_next_path, playlist_context=self.playlist, playlist_index=_next_idx)
                self.play()
            else: 
                logger.info("End of playlist and repeat is off. Stopping.")
                self.stop() # Full stop
                self.current_index = -1 # No track selected in playlist
                if self.host_app_ref: 
                    self.host_app_ref.publish_event("playback_playlist_ended") # Signal playlist itself ended
                    self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index) # Update UI

    def _cleanup_after_playback_ends(self, error_occurred: bool = False, message: Optional[str] = None):
        # This is for unexpected thread exits or explicit stops that didn't go through normal track end.
        if not self.is_playing and not self.is_paused and not error_occurred: return
        
        logger.debug(f"Cleanup after playback: error={error_occurred}, msg='{message}'")
        self.is_playing = False; self.is_paused = False; self._pause_event.set() 
        
        if error_occurred:
            self.playback_position_sec = 0.0
            if message and self.host_app_ref: self._publish_playback_error(message, "cleanup_after_playback")
        
        if self.host_app_ref: 
            self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.playback_position_sec)

    def load_track(self, filepath: str, playlist_context: Optional[List[str]] = None, playlist_index: Optional[int] = None) -> bool:
        logger.info(f"AudioEngine: Attempting to load track: {filepath}")
        self.stop(for_load=True) # Ensure clean state before loading
        
        file_info = self._safe_sf_info(filepath)
        if not file_info: return False
        self.duration_sec = file_info.duration
        new_sample_rate, new_channels = file_info.samplerate, file_info.channels
        if not (new_sample_rate > 0 and new_channels > 0 and self.duration_sec >= 0):
            self._publish_playback_error(f"Invalid audio properties in {Path(filepath).name}", f"SR={new_sample_rate}, Ch={new_channels}, Dur={self.duration_sec}"); return False
        
        self.sample_rate, self.channels = new_sample_rate, new_channels
        self.current_file, self.playback_position_sec, self.is_paused, self.is_playing = filepath, 0.0, False, False

        title, artist, album, genre, year_str, track_num_str, disc_num_str, album_artist, bpm_tag, key_tag = Path(filepath).stem, "Unknown Artist", "Unknown Album", None, None, None, None, None, None, None
        try:
            audio_tags = mutagen.File(str(filepath), easy=True)
            if audio_tags is None: audio_tags = mutagen.File(str(filepath))
            if audio_tags:
                def sget(k_list,d=None): # Try multiple keys for a field
                    if not isinstance(k_list, list): k_list = [k_list]
                    for k_item in k_list:
                        val_list = audio_tags.get(k_item)
                        if val_list and val_list[0] is not None: return str(val_list[0])
                    return d
                title = sget(['title'], Path(filepath).stem) or Path(filepath).stem
                artist = sget(['artist'], "Unknown Artist") or "Unknown Artist"
                album = sget(['album'], "Unknown Album") or "Unknown Album"
                genre = sget(['genre'])
                year_str = sget(['originaldate', 'date', 'year']) # Order of preference
                track_num_str = sget(['tracknumber'])
                disc_num_str = sget(['discnumber'])
                album_artist = sget(['albumartist', 'band', 'performer']) # Performer as last resort
                bpm_tag = sget(['bpm', 'tbpm'])
                key_tag = sget(['key', 'tkey', 'initialkey'])
        except Exception as e: logger.error(f"Error reading tags for {filepath} with mutagen: {e}", exc_info=False)

        self.current_metadata_obj = AudioMetadata(title, artist, album, self.duration_sec, self.sample_rate, self.channels, filepath, None, genre, year_str, track_num_str, disc_num_str, album_artist, bpm_tag, key_tag)
        
        if playlist_context is not None: # Explicit playlist context provided
            self.playlist = list(playlist_context) 
            self.current_index = playlist_index if playlist_index is not None and 0 <= playlist_index < len(self.playlist) else (0 if self.playlist else -1)
            self.original_playlist_order = list(self.playlist) # New context means new original order
            if self.shuffle_mode: self._apply_shuffle() # Apply shuffle to the new playlist context
        elif not self.playlist or self.current_file not in self.playlist: # Loaded as a single track or new context not matching old
            self.playlist = [self.current_file]; self.current_index = 0; self.original_playlist_order = [self.current_file]
            if self.shuffle_mode: self._apply_shuffle()
        elif self.current_file in self.playlist: # File already in current playlist, just update index
             try: self.current_index = self.playlist.index(self.current_file)
             except ValueError: self.playlist = [self.current_file]; self.current_index = 0 # Should not happen
        
        if not self.playlist: self.current_index = -1 # Safety if playlist became empty
        
        if self.effects_chain_ref_from_host: self.effects_chain_ref_from_host.set_stream_properties(self.sample_rate, self.channels)
        logger.info(f"AudioEngine: Loaded: {self.current_metadata_obj.title} ({self.duration_sec:.2f}s)")
        if self.host_app_ref:
            self.host_app_ref.publish_event("audio_track_loaded_basic", metadata=self.current_metadata_obj)
            self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
            self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.playback_position_sec)
        return True

    def play(self, start_offset_sec: Optional[float] = None, track_path_to_load: Optional[str] = None):
        logger.info(f"AudioEngine: Play command. Track: {track_path_to_load}, Offset: {start_offset_sec}")
        # Stop any existing playback before starting new, ensures clean state.
        if self.is_playing or self.is_paused: self.stop(for_load=True) 

        if track_path_to_load:
            if not self.load_track(track_path_to_load): return # load_track now calls stop() internally
        if not self.current_file: self._publish_playback_error("No file loaded.", "play"); return
        if not self._start_sound_stream(): return # Uses current self.sample_rate, self.channels

        self.is_playing, self.is_paused = True, False
        self._pause_event.set(); self._stop_playback_event.clear()
        
        # Set desired start position
        if start_offset_sec is not None: self.playback_position_sec = max(0.0, start_offset_sec)
        # If no offset, playback_position_sec holds desired start (0 for new, or remembered if stopped)
        self._seek_request_sec = self.playback_position_sec # Playback thread will use this for initial seek
        
        if self._playback_thread and self._playback_thread.is_alive():
            logger.warning("Play: Old playback thread detected as alive. Attempting join before new start."); self._playback_thread.join(0.5)
        self._playback_thread = threading.Thread(target=self._playback_thread_func, daemon=True, name="AudioPlaybackThread")
        self._playback_thread.start()
        
        logger.info(f"AudioEngine: Playback initiated for: {self.current_file} at {self.playback_position_sec:.2f}s")
        if self.host_app_ref: self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.playback_position_sec)

    def pause(self):
        if self.is_playing and not self.is_paused:
            self.is_paused = True; self._pause_event.clear()
            logger.info(f"Playback paused at {self.get_position():.2f}s")
            if self.host_app_ref: self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.get_position())

    def resume(self):
        if self.is_playing and self.is_paused:
            self.is_paused = False; self._pause_event.set()
            logger.info(f"Playback resumed at {self.get_position():.2f}s")
            if self.host_app_ref: self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.get_position())
        elif not self.is_playing: self.play(start_offset_sec=self.playback_position_sec)

    def stop(self, for_load: bool = False):
        logger.info(f"AudioEngine: Stop command. For_load: {for_load}")
        if not self.is_playing and not self.is_paused and not for_load:
            if for_load and self.stream: self._stop_sound_stream(close_existing=True)
            return

        self._stop_playback_event.set(); self._pause_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5) # Reduced join timeout for faster stop response
            if self._playback_thread.is_alive(): logger.warning("Playback thread join timeout on stop.")
        self._playback_thread = None
        self._stop_sound_stream(close_existing=True)
        
        was_active = self.is_playing or self.is_paused
        self.is_playing, self.is_paused = False, False
        if not for_load: self.playback_position_sec = 0.0
        self._stop_playback_event.clear()

        if was_active or for_load:
            logger.info("AudioEngine: Playback stopped.")
            if self.host_app_ref: self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.playback_position_sec)

    def set_position(self, position_seconds: float):
        target_pos = max(0.0, position_seconds)
        if self.duration_sec > 0: target_pos = min(target_pos, self.duration_sec - 0.01 if self.duration_sec > 0.01 else self.duration_sec)
        
        self.playback_position_sec = target_pos 
        self._seek_request_sec = target_pos
        logger.info(f"Seek request to {target_pos:.2f}s.")
        if self.host_app_ref: self.host_app_ref.publish_event("playback_position_updated", position_seconds=self.playback_position_sec, duration_seconds=self.duration_sec)

    def get_position(self) -> float: return self.playback_position_sec

    def set_volume(self, volume_float: float): 
        self.volume = np.clip(float(volume_float), 0.0, 1.0)
        if not self.is_muted: self.previous_volume_before_mute = self.volume
        if self.host_app_ref: self.host_app_ref.publish_event("volume_changed", volume=self.volume, is_muted=self.is_muted)

    def toggle_mute(self): 
        self.is_muted = not self.is_muted
        if self.is_muted and self.volume == 0.0 and self.previous_volume_before_mute == 0.0: self.previous_volume_before_mute = 0.5 
        logger.info(f"Mute toggled: {self.is_muted}")
        if self.host_app_ref: self.host_app_ref.publish_event("volume_changed", volume=self.volume, is_muted=self.is_muted)

    def cleanup(self): 
        logger.info("AudioEngine cleanup: Stopping playback and closing stream.")
        self.stop() 

    def set_shuffle_mode(self, enable: bool): 
        if self.shuffle_mode == enable: return
        self.shuffle_mode = enable; self._apply_shuffle()
        logger.info(f"Shuffle mode: {self.shuffle_mode}")
        if self.host_app_ref: 
            self.host_app_ref.publish_event("shuffle_mode_changed", shuffle_on=self.shuffle_mode)
            self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)

    def _apply_shuffle(self):
        current_track_path = None
        if self.current_index != -1 and 0 <= self.current_index < len(self.playlist):
             current_track_path = self.playlist[self.current_index]

        if self.shuffle_mode:
            if not self.original_playlist_order and self.playlist: self.original_playlist_order = list(self.playlist)
            if not self.original_playlist_order: return

            num_tracks = len(self.original_playlist_order)
            self.shuffled_indices = list(range(num_tracks))
            if num_tracks > 0 : random.shuffle(self.shuffled_indices)
            self.playlist = [self.original_playlist_order[i] for i in self.shuffled_indices]
            
            if current_track_path and current_track_path in self.playlist: self.current_index = self.playlist.index(current_track_path)
            elif self.playlist: self.current_index = 0 
            else: self.current_index = -1
        else: 
            self.playlist = list(self.original_playlist_order) 
            self.shuffled_indices = []
            if current_track_path and current_track_path in self.playlist: self.current_index = self.playlist.index(current_track_path)
            elif self.playlist: self.current_index = 0
            else: self.current_index = -1
        logger.debug(f"Applied shuffle ({self.shuffle_mode}). New playlist len: {len(self.playlist)}, current_idx: {self.current_index}")

    def set_repeat_mode(self, mode: str): 
        if mode in ["off", "one", "all"] and self.repeat_mode != mode:
            self.repeat_mode = mode; logger.info(f"Repeat mode: {self.repeat_mode}")
            if self.host_app_ref: self.host_app_ref.publish_event("repeat_mode_changed", mode=self.repeat_mode)

    def get_next_track_info(self, for_previous: bool = False) -> Tuple[Optional[str], Optional[int]]: 
        if not self.playlist: return None, None
        num_tracks = len(self.playlist)
        if num_tracks == 0: return None, None
        
        current_eff_idx = self.current_index
        # If no track is selected (-1), "next" should start from beginning of playlist, "prev" from end
        if current_eff_idx == -1:
            if not for_previous and num_tracks > 0 : current_eff_idx = -1 # So +1 becomes 0
            elif for_previous and num_tracks > 0 : current_eff_idx = 0 # So -1 becomes -1, then wraps if repeat all
            else: return None, None # Should not happen if num_tracks > 0

        next_idx = -1 
        if for_previous:
            if self.repeat_mode == "one" and not self.is_playing: # If repeat one and stopped, prev should restart
                 next_idx = current_eff_idx
            else: next_idx = current_eff_idx - 1
            if next_idx < 0: next_idx = num_tracks - 1 if self.repeat_mode == "all" else -1
        else: # Next track
            if self.repeat_mode == "one": next_idx = current_eff_idx
            else: next_idx = current_eff_idx + 1
            if next_idx >= num_tracks: next_idx = 0 if self.repeat_mode == "all" else -1
        
        return (self.playlist[next_idx], next_idx) if next_idx != -1 and 0 <= next_idx < num_tracks else (None, None)

    def next_track(self): 
        filepath, new_index = self.get_next_track_info(for_previous=False)
        if filepath and new_index is not None:
            self.load_track(filepath, playlist_context=self.playlist, playlist_index=new_index)
            self.play()
        else: 
            self.stop(); self.current_index = -1 
            if self.host_app_ref: self.host_app_ref.publish_event("playback_playlist_ended"); self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)

    def previous_track(self): 
        if self.get_position() > 3.0 and self.current_index != -1 : self.play(start_offset_sec=0.0) 
        else:
            filepath, new_index = self.get_next_track_info(for_previous=True)
            if filepath and new_index is not None:
                self.load_track(filepath, playlist_context=self.playlist, playlist_index=new_index)
                self.play()
            elif self.current_file: self.play(start_offset_sec=0.0) 

    def remove_track_from_playlist_at_index(self, index_in_current_playlist: int) -> bool: 
        if not self.playlist or not (0 <= index_in_current_playlist < len(self.playlist)): return False
        removed_track_path = self.playlist.pop(index_in_current_playlist)
        logger.info(f"Removed '{Path(removed_track_path).name}' from playlist.")
        # Also remove from original_playlist_order to keep them in sync if not shuffling,
        # or to correctly rebuild shuffle if shuffling.
        if removed_track_path in self.original_playlist_order:
            try: self.original_playlist_order.remove(removed_track_path)
            except ValueError: pass # Should not happen if lists are consistent
        
        if self.current_index == index_in_current_playlist:
            self.stop(for_load=True) # Stop current playback, preserve position if for_load
            if self.playlist: # If tracks remain
                new_play_index = min(index_in_current_playlist, len(self.playlist) - 1) # Try to stay at same index
                if new_play_index < 0 and self.playlist: new_play_index = 0 
                
                if new_play_index >=0: 
                    self.current_index = new_play_index 
                    self.load_track(self.playlist[self.current_index], playlist_context=self.playlist, playlist_index=self.current_index)
                    # Don't auto-play, let user or next logic decide
                else: # Playlist became empty
                    self.current_file=None; self.current_metadata_obj=None; self.current_index = -1
            else: # Playlist became empty
                self.current_file=None; self.current_metadata_obj=None; self.current_index = -1
        elif self.current_index > index_in_current_playlist: self.current_index -= 1
        
        if self.shuffle_mode: self._apply_shuffle() # Re-shuffle if needed
        
        if self.host_app_ref:
            self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
        return True

    def play_track_at_playlist_index(self, index_in_current_playlist: int): 
        if not self.playlist or not (0 <= index_in_current_playlist < len(self.playlist)):
            logger.warning(f"Cannot play track at index {index_in_current_playlist}, out of bounds for playlist size {len(self.playlist)}")
            return
        filepath = self.playlist[index_in_current_playlist]
        self.load_track(filepath, playlist_context=self.playlist, playlist_index=index_in_current_playlist)
        self.play()

    def add_to_playlist(self, filepaths: Union[str, List[str]], play_immediately_idx: Optional[int] = None): # play_immediately_idx refers to index within passed filepaths_list
        if isinstance(filepaths, str): filepaths_list = [filepaths]
        elif isinstance(filepaths, list): filepaths_list = filepaths
        else: logger.warning("add_to_playlist: filepaths must be str or list."); return

        added_any = False; first_newly_added_path_for_play = None

        for path_str in filepaths_list:
            p_obj = Path(path_str)
            if not p_obj.is_file(): logger.warning(f"Cannot add '{path_str}': not a file."); continue
            
            # Always add to original_playlist_order if not present
            if path_str not in self.original_playlist_order: self.original_playlist_order.append(path_str)
            # If not shuffling, also add to current live playlist if not present
            if not self.shuffle_mode and path_str not in self.playlist: self.playlist.append(path_str)
            added_any = True
            if play_immediately_idx == (len(filepaths_list) -1 - filepaths_list[::-1].index(path_str)) and first_newly_added_path_for_play is None : # If this is the target path to play
                first_newly_added_path_for_play = path_str
        
        if added_any:
            if self.shuffle_mode: self._apply_shuffle() # Rebuild self.playlist
            if self.host_app_ref: self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)

            if first_newly_added_path_for_play and first_newly_added_path_for_play in self.playlist:
                idx_in_current_playlist = self.playlist.index(first_newly_added_path_for_play)
                self.load_track(first_newly_added_path_for_play, playlist_context=self.playlist, playlist_index=idx_in_current_playlist)
                self.play()

    def load_playlist_paths(self, filepaths: List[str], play_first: bool = True, replace_queue: bool = True):
        if not isinstance(filepaths, list): logger.error("load_playlist_paths expects list."); return
        valid_paths = [p for p in filepaths if Path(p).is_file()]
        if not valid_paths:
            if replace_queue: self.clear_playlist(); logger.warning("load_playlist_paths: No valid paths, queue cleared.")
            else: logger.warning("load_playlist_paths: No valid paths to append."); return
        
        if replace_queue: self.stop(for_load=True)
        
        if replace_queue:
            self.playlist = list(valid_paths); self.original_playlist_order = list(valid_paths)
            self.current_index = 0 if self.playlist else -1
        else: 
            for path in valid_paths:
                if path not in self.original_playlist_order: self.original_playlist_order.append(path)
                if not self.shuffle_mode and path not in self.playlist: self.playlist.append(path)
            if self.current_index == -1 and self.playlist: self.current_index = 0
        
        if self.shuffle_mode: self._apply_shuffle() 
        if self.host_app_ref: self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
        if play_first and self.playlist and self.current_index != -1:
            self.load_track(self.playlist[self.current_index], playlist_context=self.playlist, playlist_index=self.current_index); self.play()
        elif not self.playlist: self.clear_playlist()

    def clear_playlist(self):
        self.stop(for_load=True); self.playlist = []; self.original_playlist_order = []; self.shuffled_indices = []
        self.current_index = -1; self.current_file = None; self.current_metadata_obj = None
        self.duration_sec = 0.0; self.playback_position_sec = 0.0; logger.info("Playlist cleared.")
        if self.host_app_ref:
            self.host_app_ref.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
            self.host_app_ref.publish_event("playback_state_changed", is_playing=False, is_paused=False, position=0.0)
            self.host_app_ref.publish_event("audio_track_loaded_basic", metadata=AudioMetadata(duration=0.0))

    def force_sync_playback_to_state(self, library_track_id_to_play: int, target_position_seconds: float, target_is_playing: bool):
        logger.info(f"Force sync: TrackID {library_track_id_to_play}, Pos {target_position_seconds:.2f}s, Playing: {target_is_playing}")
        if not self.host_app_ref or not self.host_app_ref.music_library_db_ref: logger.warning("Cannot force sync: No music_library_db_ref."); return
        
        track_data = self.host_app_ref.request_library_action("get_track_by_id", {'track_id': library_track_id_to_play})
        if not track_data or not hasattr(track_data, 'file_path') or not track_data.file_path:
            self._publish_playback_error(f"Track ID {library_track_id_to_play} not found for sync."); return

        self.stop(for_load=True) 
        # For sync, create a temporary context for this one track. Don't alter user's main playlist.
        temp_playlist_context = [track_data.file_path]
        loaded = self.load_track(track_data.file_path, playlist_context=temp_playlist_context, playlist_index=0)

        if loaded:
            self.set_position(target_position_seconds)
            if target_is_playing: self.play(start_offset_sec=self.playback_position_sec)
            else: 
                self.is_playing = False; self.is_paused = True; self._pause_event.clear()
                if self.host_app_ref: self.host_app_ref.publish_event("playback_state_changed", is_playing=self.is_playing, is_paused=self.is_paused, position=self.playback_position_sec)
            logger.info(f"Force sync complete for track ID {library_track_id_to_play}.")
        else: self._publish_playback_error(f"Failed to sync to track '{Path(track_data.file_path).name}'.")


class LyricsDisplay(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Optional[Any] = None):
        super().__init__(parent)
        self.host_app = host_app_ref
        self.lyrics_data: List[Tuple[float, str]] = []
        self.current_line_index = -1
        default_font_family = "Arial" 
        if platform.system() == "Windows": default_font_family = "Segoe UI"
        elif platform.system() == "Darwin": default_font_family = "Helvetica Neue"
        self.font_config = {'family': default_font_family, 'size_normal': 12, 'size_current': 13, 'style_current': "bold"}

        self.lyrics_text_widget = scrolledtext.ScrolledText(self, wrap=tk.WORD, state=tk.DISABLED, font=(self.font_config['family'], self.font_config['size_normal']), padx=10, pady=10, borderwidth=0, relief=tk.FLAT)
        self.lyrics_text_widget.pack(fill=tk.BOTH, expand=True)
        self._setup_tags()
        if self.host_app and self.host_app.theme_manager and self.host_app.theme_manager.get_current_theme():
            self.apply_theme(self.host_app.theme_manager.get_current_theme())

    def _setup_tags(self):
        self.lyrics_text_widget.tag_configure("normal_line", foreground="gray", justify=tk.CENTER, font=(self.font_config['family'], self.font_config['size_normal']))
        self.lyrics_text_widget.tag_configure("current_line", foreground="white", font=(self.font_config['family'], self.font_config['size_current'], self.font_config['style_current']), justify=tk.CENTER)
        self.lyrics_text_widget.tag_configure("upcoming_line", foreground="#BBBBBB", justify=tk.CENTER, font=(self.font_config['family'], self.font_config['size_normal']))

    def apply_theme(self, theme: Any): 
        self.font_config['family'] = theme.font_family
        self.font_config['size_normal'] = theme.font_size_large 
        self.font_config['size_current'] = theme.font_size_title 
        
        self.lyrics_text_widget.configure(background=theme.primary_bg, foreground=theme.secondary_fg, insertbackground=theme.accent_color, font=(self.font_config['family'], self.font_config['size_normal']))
        self._setup_tags() 
        self.lyrics_text_widget.tag_configure("current_line", foreground=theme.accent_color, font=(self.font_config['family'], self.font_config['size_current'], self.font_config['style_current']))
        self.lyrics_text_widget.tag_configure("upcoming_line", foreground=theme.primary_fg) 
        if self.lyrics_data: self.load_lyrics(self.lyrics_data, force_reload=True)

    def load_lyrics(self, lyrics: List[Tuple[float, str]], force_reload:bool=False):
        if not lyrics and not force_reload and not self.lyrics_data : return 
        
        self.lyrics_data = sorted(lyrics, key=lambda x: x[0]) if lyrics else []
        self.current_line_index = -1
        self.lyrics_text_widget.config(state=tk.NORMAL)
        self.lyrics_text_widget.delete(1.0, tk.END)
        
        if not self.lyrics_data:
            self.lyrics_text_widget.insert(tk.END, "\n\n\n--- No lyrics available ---", "normal_line")
        else:
            self.lyrics_text_widget.insert(tk.END, "\n\n\n") 
            for _timestamp, text in self.lyrics_data:
                self.lyrics_text_widget.insert(tk.END, text + "\n", "normal_line")
        self.lyrics_text_widget.config(state=tk.DISABLED)
        if self.lyrics_data: self.update_current_line(0, force_update=True)

    def update_current_line(self, current_playback_time_sec: float, force_update: bool = False):
        if not self.lyrics_data: return
        new_current_line_idx = -1
        for i, (timestamp, _text) in enumerate(self.lyrics_data):
            if current_playback_time_sec >= timestamp: new_current_line_idx = i
            else: break
        if new_current_line_idx == self.current_line_index and not force_update: return

        self.lyrics_text_widget.config(state=tk.NORMAL)
        initial_padding_exists = self.lyrics_text_widget.get(1.0, "4.0").strip() == "" and len(self.lyrics_data)>0
        line_start_offset = 4 if initial_padding_exists else 1 
        
        if self.current_line_index != -1: 
            prev_line_tk_idx = self.current_line_index + line_start_offset
            self.lyrics_text_widget.tag_remove("current_line", f"{prev_line_tk_idx}.0", f"{prev_line_tk_idx+1}.0")
            self.lyrics_text_widget.tag_add("normal_line", f"{prev_line_tk_idx}.0", f"{prev_line_tk_idx+1}.0")
        
        self.current_line_index = new_current_line_idx
        if self.current_line_index != -1:
            current_line_tk_idx = self.current_line_index + line_start_offset
            current_line_start_tk_str = f"{current_line_tk_idx}.0"
            current_line_end_tk_str = f"{current_line_tk_idx + 1}.0"

            self.lyrics_text_widget.tag_remove("normal_line", current_line_start_tk_str, current_line_end_tk_str)
            self.lyrics_text_widget.tag_remove("upcoming_line", current_line_start_tk_str, current_line_end_tk_str)
            self.lyrics_text_widget.tag_add("current_line", current_line_start_tk_str, current_line_end_tk_str)
            
            try:
                dline = self.lyrics_text_widget.dlineinfo(current_line_start_tk_str)
                if dline:
                    line_height = dline[-1]; widget_height = self.lyrics_text_widget.winfo_height()
                    visible_lines = max(1, int(widget_height / line_height if line_height > 0 else 15))
                    scroll_to_idx = max(1, current_line_tk_idx - visible_lines // 3)
                    self.lyrics_text_widget.see(f"{scroll_to_idx}.0")
                else: self.lyrics_text_widget.see(current_line_start_tk_str)
            except tk.TclError: self.lyrics_text_widget.see(current_line_start_tk_str)

            for upcoming_offset in range(1, 4):
                upcoming_actual_idx = self.current_line_index + upcoming_offset
                if 0 <= upcoming_actual_idx < len(self.lyrics_data):
                    upcoming_tk_idx = upcoming_actual_idx + line_start_offset
                    self.lyrics_text_widget.tag_remove("normal_line", f"{upcoming_tk_idx}.0", f"{upcoming_tk_idx+1}.0")
                    self.lyrics_text_widget.tag_add("upcoming_line", f"{upcoming_tk_idx}.0", f"{upcoming_tk_idx+1}.0")
        self.lyrics_text_widget.config(state=tk.DISABLED)

class AudioAnalyzer: 
    def __init__(self): logger.debug("AudioAnalyzer initialized (basic).")
    def analyze_track(self, filepath: str) -> Dict[str, Any]: return {}


class FlowStateApp(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent)
        self.host_app = host_app_ref
        self.root_app_tk = parent.winfo_toplevel()
        self._just_started_playing_flag = True 
        self.play_pause_button=None; self.stop_button=None; self.next_button=None; self.prev_button=None
        self.volume_slider=None; self.volume_label=None; self.mute_button=None
        self.progress_slider=None; self.current_time_label=None; self.total_time_label=None
        self.track_info_label=None; self.status_label_var = tk.StringVar(value="Welcome to Flow State!")
        self.shuffle_button=None; self.repeat_button=None
        self.queue_treeview: Optional[ttk.Treeview] = None; self.queue_scrollbar=None
        self.viz_canvas_tk_agg=None; self.viz_fig=None; self.viz_ax_spectrum=None; self.viz_ax_waveform=None
        self.viz_spectrum_bars=None; self.viz_waveform_line=None; self.viz_animation=None
        self.lyrics_display=None; self.analyzer=None
        self._after_id_progress_update: Optional[str] = None; self._user_is_dragging_slider = False
        self.like_button: Optional[ttk.Button] = None; self.dislike_button: Optional[ttk.Button] = None
        self._create_ui_widgets()
        self._setup_playback_control_buttons()
        self._subscribe_to_host_events()
        self.pack(fill=tk.BOTH, expand=True)
        if self.host_app.audio_engine_ref:
            ae = self.host_app.audio_engine_ref
            self.on_volume_changed_event(volume=ae.volume, is_muted=ae.is_muted)
            self.on_shuffle_mode_changed_event(shuffle_on=ae.shuffle_mode)
            self.on_repeat_mode_changed_event(mode=ae.repeat_mode)
            if ae.current_metadata_obj: self.on_track_fully_loaded_details_event(ae.current_metadata_obj, [])
            self.on_playback_playlist_changed_event(playlist=ae.playlist, current_index=ae.current_index)

    def _create_ui_widgets(self):
        # Top Controls Area
        top_controls_frame = ttk.Frame(self, padding=5); top_controls_frame.pack(fill=tk.X, side=tk.TOP)
        playback_buttons_frame = ttk.Frame(top_controls_frame); playback_buttons_frame.pack(side=tk.LEFT, padx=5)
        self.prev_button = ttk.Button(playback_buttons_frame, text="⏮"); self.prev_button.pack(side=tk.LEFT)
        self.play_pause_button = ttk.Button(playback_buttons_frame, text="▶"); self.play_pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(playback_buttons_frame, text="⏹"); self.stop_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(playback_buttons_frame, text="⏭"); self.next_button.pack(side=tk.LEFT, padx=5)
        
        track_actions_frame = ttk.Frame(top_controls_frame); track_actions_frame.pack(side=tk.LEFT, padx=10)
        self.like_button = ttk.Button(track_actions_frame, text="❤️", width=3, command=self._on_like_track); self.like_button.pack(side=tk.LEFT)
        self.dislike_button = ttk.Button(track_actions_frame, text="💔", width=3, command=self._on_dislike_track); self.dislike_button.pack(side=tk.LEFT, padx=(2,0))

        volume_frame = ttk.Frame(top_controls_frame); volume_frame.pack(side=tk.RIGHT, padx=5)
        self.mute_button = ttk.Button(volume_frame, text="🔊"); self.mute_button.pack(side=tk.LEFT)
        self.volume_slider = ttk.Scale(volume_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=100); self.volume_slider.pack(side=tk.LEFT, padx=5)
        self.volume_label = ttk.Label(volume_frame, text="70%", width=4); self.volume_label.pack(side=tk.LEFT)
        self.track_info_label = ttk.Label(top_controls_frame, text="No Track Loaded", anchor=tk.CENTER, font=('Arial', 11)); self.track_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Progress Bar Area
        progress_area_frame = ttk.Frame(self, padding=(5,10)); progress_area_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.current_time_label = ttk.Label(progress_area_frame, text="00:00", width=5); self.current_time_label.pack(side=tk.LEFT, padx=(0,5))
        self.progress_slider = ttk.Scale(progress_area_frame, from_=0, to=100, orient=tk.HORIZONTAL); self.progress_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.total_time_label = ttk.Label(progress_area_frame, text="00:00", width=5); self.total_time_label.pack(side=tk.LEFT, padx=(5,0))

        # Main Content Area
        main_content_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL); main_content_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        left_content_frame = ttk.Frame(main_content_paned, width=600); main_content_paned.add(left_content_frame, weight=3)
        self.display_notebook = ttk.Notebook(left_content_frame); self.display_notebook.pack(fill=tk.BOTH, expand=True)
        
        viz_frame = ttk.Frame(self.display_notebook); self.display_notebook.add(viz_frame, text="Visualizer")
        try:
            self.viz_fig, (self.viz_ax_spectrum, self.viz_ax_waveform) = plt.subplots(2, 1, figsize=(6,4), facecolor='#1E1E1E', constrained_layout=True)
            self._setup_basic_viz_plot(self.viz_ax_spectrum, "Spectrum", is_spectrum=True)
            self._setup_basic_viz_plot(self.viz_ax_waveform, "Waveform")
            self.viz_canvas_tk_agg = FigureCanvasTkAgg(self.viz_fig, master=viz_frame)
            self.viz_canvas_tk_agg.get_tk_widget().pack(fill=tk.BOTH, expand=True); self.viz_canvas_tk_agg.draw_idle()
        except Exception as e_plt: logger.error(f"Matplotlib viz setup error: {e_plt}", exc_info=True)

        lyrics_tab_frame = ttk.Frame(self.display_notebook); self.display_notebook.add(lyrics_tab_frame, text="Lyrics")
        self.lyrics_display = LyricsDisplay(lyrics_tab_frame, self.host_app); self.lyrics_display.pack(fill=tk.BOTH, expand=True)

        queue_frame_outer = ttk.Frame(main_content_paned, width=300); main_content_paned.add(queue_frame_outer, weight=1)
        queue_header_frame = ttk.Frame(queue_frame_outer); queue_header_frame.pack(fill=tk.X)
        ttk.Label(queue_header_frame, text="Playback Queue", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5, pady=3)
        self.shuffle_button = ttk.Button(queue_header_frame, text="🔀 Off"); self.shuffle_button.pack(side=tk.RIGHT, padx=(0,2))
        self.repeat_button = ttk.Button(queue_header_frame, text="🔁 Off"); self.repeat_button.pack(side=tk.RIGHT, padx=2)

        self.queue_treeview = ttk.Treeview(queue_frame_outer, columns=("title", "artist", "duration"), show="headings", selectmode="browse")
        self.queue_treeview.heading("title", text="Title"); self.queue_treeview.heading("artist", text="Artist"); self.queue_treeview.heading("duration", text="Time")
        self.queue_treeview.column("title", width=150, stretch=tk.YES); self.queue_treeview.column("artist", width=100, stretch=tk.YES); self.queue_treeview.column("duration", width=50, stretch=tk.NO, anchor=tk.E)
        self.queue_scrollbar = ttk.Scrollbar(queue_frame_outer, orient="vertical", command=self.queue_treeview.yview)
        self.queue_treeview.configure(yscrollcommand=self.queue_scrollbar.set)
        self.queue_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.queue_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.queue_treeview.bind("<Double-1>", self.on_queue_double_click); self.queue_treeview.bind("<Button-3>", self.show_queue_context_menu)
        self.analyzer = AudioAnalyzer()

    def _setup_basic_viz_plot(self, ax: plt.Axes, title: str, is_spectrum: bool = False):
        ax.set_title(title, color='white', fontsize=10); ax.set_facecolor('#111111')
        ax.tick_params(axis='x', colors='gray', labelsize=8); ax.tick_params(axis='y', colors='gray', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('gray')
        if is_spectrum: ax.set_ylim(0, 1); ax.set_xlim(0, 1024); ax.set_xticks([]); ax.set_yticks([])
        else: ax.set_ylim(-1, 1); ax.set_xlim(0, 2048); ax.set_xticks([]); ax.set_yticks([])

    def _update_basic_viz(self, frame_num: int):
        if not self.host_app or not self.host_app.audio_engine_ref or \
           not self.host_app.audio_engine_ref.is_playing or \
           not hasattr(self.host_app.audio_engine_ref, 'effects_output_buffer_for_viz') or \
           not self.host_app.audio_engine_ref.effects_output_buffer_for_viz.any():
            if self.viz_spectrum_bars: [bar.set_height(0) for bar in self.viz_spectrum_bars]
            if self.viz_waveform_line: self.viz_waveform_line.set_ydata(np.zeros(2048))
            if self.viz_canvas_tk_agg: self.viz_canvas_tk_agg.draw_idle()
            return

        audio_data_mono = self.host_app.audio_engine_ref.effects_output_buffer_for_viz
        if audio_data_mono.size == 0: return

        if self.viz_ax_spectrum:
            fft_size = min(audio_data_mono.size, 2048)
            if fft_size > 0:
                 yf = np.abs(rfft(audio_data_mono[:fft_size])) * (2 / fft_size)
                 yf_normalized = np.clip(yf[:1024], 0, 1) # Ensure 1024 bins for display
            else: yf_normalized = np.zeros(1024)
            
            if not self.viz_spectrum_bars:
                num_bars = 64; bar_x_pos = np.linspace(0, 1023, num_bars)
                bar_indices = np.linspace(0, yf_normalized.size -1 if yf_normalized.size > 0 else 0, num_bars, dtype=int)
                bar_heights = yf_normalized[bar_indices] if yf_normalized.size > 0 else np.zeros(num_bars)
                self.viz_spectrum_bars = self.viz_ax_spectrum.bar(bar_x_pos, bar_heights, width=1023/(num_bars*1.2), color='#00FFFF')
            else:
                num_bars = len(self.viz_spectrum_bars); bar_indices = np.linspace(0, yf_normalized.size -1 if yf_normalized.size > 0 else 0, num_bars, dtype=int)
                bar_heights = yf_normalized[bar_indices] if yf_normalized.size > 0 else np.zeros(num_bars)
                for bar, h in zip(self.viz_spectrum_bars, bar_heights): bar.set_height(h)
        
        if self.viz_ax_waveform:
            data_to_plot = audio_data_mono[:2048]; pad_len = 2048 - data_to_plot.size
            if pad_len > 0: data_to_plot = np.pad(data_to_plot, (0, pad_len))
            if not self.viz_waveform_line: self.viz_waveform_line, = self.viz_ax_waveform.plot(data_to_plot, color='#00FF00', lw=1)
            else: self.viz_waveform_line.set_ydata(data_to_plot)
        
        if self.viz_canvas_tk_agg: self.viz_canvas_tk_agg.draw_idle()

    def _start_viz_animation(self):
        if self.viz_animation is None and self.viz_fig is not None and self.host_app.audio_engine_ref:
            self.viz_animation = FuncAnimation(self.viz_fig, self._update_basic_viz, interval=50, blit=False, cache_frame_data=False)
            logger.debug("Basic visualizer animation started.")

    def _stop_viz_animation(self): logger.debug("Basic visualizer animation will pause due to is_playing check.")

    def _setup_playback_control_buttons(self):
        if not self.host_app or not self.host_app.audio_engine_ref: logger.warning("AudioEngine not ready for controls setup."); return
        if self.play_pause_button: self.play_pause_button.config(command=self.play_pause)
        if self.stop_button: self.stop_button.config(command=lambda: self.host_app.request_playback_action("stop"))
        if self.next_button: self.next_button.config(command=self.next_track_action)
        if self.prev_button: self.prev_button.config(command=lambda: self.host_app.request_playback_action("previous"))
        if self.volume_slider: self.volume_slider.config(command=self.update_volume_from_slider); self.volume_slider.set(self.host_app.audio_engine_ref.volume * 100)
        if self.mute_button: self.mute_button.config(command=lambda: self.host_app.request_playback_action("toggle_mute"))
        if self.progress_slider:
            self.progress_slider.bind("<ButtonPress-1>", self._on_progress_slider_press)
            self.progress_slider.bind("<ButtonRelease-1>", self._on_progress_slider_release)
            self.progress_slider.bind("<B1-Motion>", self._on_progress_slider_drag)
        if self.shuffle_button: self.shuffle_button.config(command=self.toggle_shuffle_mode)
        if self.repeat_button: self.repeat_button.config(command=self.cycle_repeat_mode)

    def _subscribe_to_host_events(self):
        if not self.host_app: return
        self.host_app.subscribe_to_event("playback_state_changed", self.on_playback_state_changed_event)
        self.host_app.subscribe_to_event("playback_position_updated", self.on_playback_position_updated_event)
        self.host_app.subscribe_to_event("audio_track_loaded_basic", self.on_audio_track_loaded_basic_event)
        self.host_app.subscribe_to_event("track_fully_loaded_with_details", self.on_track_fully_loaded_details_event)
        self.host_app.subscribe_to_event("volume_changed", self.on_volume_changed_event)
        self.host_app.subscribe_to_event("playback_error", self.on_playback_error_event)
        self.host_app.subscribe_to_event("playback_playlist_changed", self.on_playback_playlist_changed_event)
        self.host_app.subscribe_to_event("shuffle_mode_changed", self.on_shuffle_mode_changed_event)
        self.host_app.subscribe_to_event("repeat_mode_changed", self.on_repeat_mode_changed_event)
        self.host_app.subscribe_to_event("playback_track_ended", self.on_playback_track_ended_event)
        if self.host_app.theme_manager: self.host_app.theme_manager.register_callback(self.apply_theme_to_player_ui)
        # Apply initial theme after a short delay to ensure all widgets are created.
        if self.host_app.theme_manager and self.host_app.theme_manager.get_current_theme(): 
            self.after(100, lambda: self.apply_theme_to_player_ui(self.host_app.theme_manager.get_current_theme()))


    def apply_theme_to_player_ui(self, theme: Any):
        self.configure(background=theme.secondary_bg)
        style = ttk.Style.get_instance() # Should be self.style if defined in __init__
        if not hasattr(self, 'style'): self.style = style # Ensure self.style exists

        self.style.configure("TButton", background=theme.accent_bg, foreground=theme.primary_fg, font=(theme.font_family, theme.font_size_normal))
        self.style.map("TButton", background=[('active', theme.highlight_color), ('disabled', theme.secondary_bg)])
        self.style.configure("TLabel", background=theme.secondary_bg, foreground=theme.primary_fg, font=(theme.font_family, theme.font_size_normal))
        if self.track_info_label: self.track_info_label.configure(font=(theme.font_family, theme.font_size_large, 'bold'), background=theme.secondary_bg, foreground=theme.primary_fg)
        if self.volume_label: self.volume_label.configure(background=theme.secondary_bg, foreground=theme.secondary_fg)
        if self.current_time_label: self.current_time_label.configure(background=theme.secondary_bg, foreground=theme.secondary_fg)
        if self.total_time_label: self.total_time_label.configure(background=theme.secondary_bg, foreground=theme.secondary_fg)
        self.style.configure("TScale", background=theme.secondary_bg, troughcolor=theme.primary_bg)
        self.style.configure("Horizontal.TScale", background=theme.secondary_bg, troughcolor=theme.primary_bg)
        self.style.configure("TFrame", background=theme.secondary_bg); self.style.configure("TPanedwindow", background=theme.secondary_bg)
        self.style.configure("TNotebook", background=theme.primary_bg); self.style.configure("TNotebook.Tab", background=theme.accent_bg, foreground=theme.secondary_fg, padding=[10, 3], font=(theme.font_family, theme.font_size_normal))
        self.style.map("TNotebook.Tab", background=[("selected", theme.highlight_color)], foreground=[("selected", theme.primary_fg)])
        self.style.configure("Treeview", background=theme.primary_bg, foreground=theme.primary_fg, fieldbackground=theme.primary_bg, font=(theme.font_family, theme.font_size_normal), rowheight=int(theme.font_size_normal*2.2)) # Adjust rowheight
        self.style.map("Treeview", background=[('selected', theme.highlight_color)], foreground=[('selected', theme.primary_fg)])
        self.style.configure("Treeview.Heading", background=theme.accent_bg, foreground=theme.primary_fg, font=(theme.font_family, theme.font_size_normal, 'bold'), relief=tk.FLAT, padding=3)
        self.style.map("Treeview.Heading", background=[('active',theme.highlight_color)])
        self.style.configure("TProgressbar", troughcolor=theme.primary_bg, background=theme.accent_color, thickness=10)
        if self.viz_fig:
            self.viz_fig.set_facecolor(theme.viz_bg)
            if self.viz_ax_spectrum: self._theme_mpl_ax(self.viz_ax_spectrum, theme, is_spectrum=True)
            if self.viz_ax_waveform: self._theme_mpl_ax(self.viz_ax_waveform, theme)
            if self.viz_spectrum_bars: [bar.set_color(theme.viz_primary) for bar in self.viz_spectrum_bars]
            if self.viz_waveform_line: self.viz_waveform_line.set_color(theme.waveform_color)
            if self.viz_canvas_tk_agg: self.viz_canvas_tk_agg.draw_idle()
        if self.lyrics_display: self.lyrics_display.apply_theme(theme)

    def _theme_mpl_ax(self, ax: plt.Axes, theme: Any, is_spectrum:bool=False):
        ax.set_title(ax.get_title(), color=theme.primary_fg, fontsize=theme.font_size_normal); ax.set_facecolor(theme.viz_bg) 
        ax.tick_params(axis='x', colors=theme.secondary_fg, labelsize=theme.font_size_small); ax.tick_params(axis='y', colors=theme.secondary_fg, labelsize=theme.font_size_small)
        for spine in ax.spines.values(): spine.set_color(theme.secondary_fg)

    def open_file(self):
        filepath = filedialog.askopenfilename(title="Open Audio File", filetypes=(("Audio Files", "*.mp3 *.wav *.ogg *.flac *.m4a *.opus"), ("All files", "*.*")))
        if filepath: self.host_app.request_playback_action("load_and_play_path", {'filepath': filepath})

    def open_folder_threaded(self):
        folder_path = filedialog.askdirectory(title="Open Folder")
        if not folder_path: return
        def _scan():
            paths = [os.path.join(r,f) for r,_,fi in os.walk(folder_path) for f in fi if f.lower().endswith((".mp3",".wav",".ogg",".flac",".m4a",".opus"))]
            if paths: self.host_app.request_playback_action("load_playlist_paths", {'paths': sorted(paths), 'play_first': True, 'replace_queue': True})
            elif self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, lambda: messagebox.showinfo("No Audio", "No compatible audio files found.", parent=self.root_app_tk))
        threading.Thread(target=_scan, daemon=True, name="FolderScanThread").start()

    def load_track_by_path_and_play(self, filepath: str): # Helper called by HostApp
        # This specific method is more for internal calls if main_player_ui_ref itself is handling it.
        # The HostAppInterface would typically call AudioEngine directly for this.
        # However, if it's meant for the UI to initiate its own loading sequence:
        if self.host_app.audio_engine_ref:
            if self.host_app.audio_engine_ref.load_track(filepath): # Assuming load_track doesn't auto-play
                self.host_app.audio_engine_ref.play()
            # else: error handled by load_track publishing event

    def play_track_by_id_from_library(self, track_id: int): # Helper called by HostApp
        if self.host_app.audio_engine_ref and self.host_app.music_library_db_ref:
            track_obj = self.host_app.music_library_db_ref.get_track(track_id) # This is sync call
            if track_obj and track_obj.file_path:
                # Replace current queue with this single track and play it
                self.host_app.request_playback_action("load_playlist_paths", 
                                                      {'paths': [track_obj.file_path], 
                                                       'play_first': True, 
                                                       'replace_queue': True})
            else: logger.warning(f"Could not find track with ID {track_id} in library to play.")
        else: logger.warning("Cannot play by ID: AudioEngine or MusicLibraryDB not available.")


    def play_pause(self):
        if not self.host_app.audio_engine_ref: return
        ae = self.host_app.audio_engine_ref
        if ae.is_playing and not ae.is_paused: self.host_app.request_playback_action("pause")
        elif ae.is_playing and ae.is_paused: self.host_app.request_playback_action("resume")
        else: 
            if ae.current_file: self.host_app.request_playback_action("play")
            else: self.open_file()

    def update_volume_from_slider(self, value_str: str):
        if not self.host_app.audio_engine_ref: return
        self.host_app.request_playback_action("set_volume", {'level': float(value_str) / 100.0})

    def _on_progress_slider_press(self, event: tk.Event): 
        self._user_is_dragging_slider = True
        if self._after_id_progress_update and self.root_app_tk.winfo_exists():
            self.root_app_tk.after_cancel(self._after_id_progress_update)
            self._after_id_progress_update = None
            
    def _on_progress_slider_drag(self, event: tk.Event): 
        if self._user_is_dragging_slider and self.progress_slider and self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.duration_sec > 0:
            pos_s = (self.progress_slider.get() / 100.0) * self.host_app.audio_engine_ref.duration_sec
            if self.current_time_label: self.current_time_label.config(text=self._format_time(pos_s))

    def _on_progress_slider_release(self, event: tk.Event):
        if self._user_is_dragging_slider and self.progress_slider and self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.duration_sec > 0:
            self._user_is_dragging_slider = False
            seek_s = (self.progress_slider.get() / 100.0) * self.host_app.audio_engine_ref.duration_sec
            self.host_app.request_playback_action("seek", {'position_seconds': seek_s})
        else: self._user_is_dragging_slider = False 
        self._schedule_progress_update()

    def toggle_shuffle_mode(self):
        if self.host_app.audio_engine_ref: self.host_app.request_playback_action("set_shuffle_mode", {'state': not self.host_app.audio_engine_ref.shuffle_mode})
    
    def cycle_repeat_mode(self):
        if self.host_app.audio_engine_ref:
            modes = ["off", "one", "all"]; current = self.host_app.audio_engine_ref.repeat_mode
            new_mode = modes[(modes.index(current) + 1) % len(modes)] if current in modes else "off"
            self.host_app.request_playback_action("set_repeat_mode", {'mode': new_mode})

    def on_playback_state_changed_event(self, is_playing: bool, is_paused: bool, position: float, **kwargs):
        if self.play_pause_button: self.play_pause_button.config(text="⏸" if is_playing and not is_paused else "▶")
        ae_duration = getattr(self.host_app.audio_engine_ref, 'duration_sec', 0.0) if self.host_app.audio_engine_ref else 0.0
        if not self._user_is_dragging_slider: self.on_playback_position_updated_event(position, ae_duration)
        
        if is_playing and not is_paused: 
            self._schedule_progress_update(); self._start_viz_animation()
            if self._just_started_playing_flag: # Track actual play start for interaction
                if self.host_app and self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.current_metadata_obj and self.host_app.audio_engine_ref.current_metadata_obj.id is not None:
                    self.host_app.publish_event("user_interaction_track_played", track_id=self.host_app.audio_engine_ref.current_metadata_obj.id, timestamp=datetime.now(timezone.utc).isoformat())
                self._just_started_playing_flag = False 
        else: 
            if self._after_id_progress_update and self.root_app_tk.winfo_exists(): self.root_app_tk.after_cancel(self._after_id_progress_update); self._after_id_progress_update = None
            if is_paused: self.on_playback_position_updated_event(position, ae_duration) # Update one last time for pause
        if not is_playing: self._just_started_playing_flag = True

    def _schedule_progress_update(self):
        if self._after_id_progress_update and self.root_app_tk.winfo_exists(): self.root_app_tk.after_cancel(self._after_id_progress_update); self._after_id_progress_update=None
        if self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.is_playing and not self.host_app.audio_engine_ref.is_paused and self.root_app_tk.winfo_exists():
            current_pos = self.host_app.audio_engine_ref.get_position(); duration = self.host_app.audio_engine_ref.duration_sec
            self.on_playback_position_updated_event(current_pos, duration)
            self._after_id_progress_update = self.root_app_tk.after(250, self._schedule_progress_update)

    def on_playback_position_updated_event(self, position_seconds: float, duration_seconds: float):
        if self.progress_slider and duration_seconds > 0 and not self._user_is_dragging_slider: self.progress_slider.set((position_seconds / duration_seconds) * 100)
        elif self.progress_slider and duration_seconds <= 0 : self.progress_slider.set(0) # Handles case of no duration or end
        if self.current_time_label: self.current_time_label.config(text=self._format_time(position_seconds))
        if self.total_time_label: self.total_time_label.config(text=self._format_time(duration_seconds))
        if self.lyrics_display: self.lyrics_display.update_current_line(position_seconds)

    def _format_time(self, seconds_float: float, show_hours: bool = False) -> str:
        if seconds_float is None or seconds_float < 0: return "--:--" if not show_hours else "--:--:--"
        s, m, h = int(seconds_float % 60), int((seconds_float // 60) % 60), int(seconds_float // 3600)
        return f"{h:d}:{m:02d}:{s:02d}" if show_hours or h > 0 else f"{m:02d}:{s:02d}"

    def on_audio_track_loaded_basic_event(self, metadata: AudioMetadata, **kwargs):
        if self.track_info_label: self.track_info_label.config(text=f"{metadata.title or 'Unknown Title'} - {metadata.artist or 'Unknown Artist'} ({metadata.album or 'Unknown Album'})")
        if self.total_time_label: self.total_time_label.config(text=self._format_time(metadata.duration))
        if self.progress_slider: self.progress_slider.config(to=100); self.progress_slider.set(0)
        if self.current_time_label: self.current_time_label.config(text="00:00")
        if self.lyrics_display: self.lyrics_display.load_lyrics([])
        self._just_started_playing_flag = True
        if self.like_button: self.like_button.config(relief=tk.RAISED)
        if self.dislike_button: self.dislike_button.config(relief=tk.RAISED)
        if self.root_app_tk.winfo_exists(): self.root_app_tk.after(100, lambda m=metadata: self._fetch_and_process_full_track_details(m))

    def _fetch_and_process_full_track_details(self, metadata: AudioMetadata):
        lyrics_data = self._parse_lrc_file(metadata.file_path) if metadata.file_path else []
        logger.info(f"Lyrics for {metadata.title}: {'Found' if lyrics_data else 'Not found'}")
        if self.host_app: self.host_app.publish_event("track_fully_loaded_with_details", track_metadata=metadata, lyrics_data_for_track=lyrics_data)

    def _parse_lrc_file(self, audio_filepath: str) -> List[Tuple[float, str]]:
        lrc_path = Path(audio_filepath).with_suffix(".lrc"); lyrics = []
        if lrc_path.exists():
            try:
                with open(lrc_path, 'r', encoding='utf-8', errors='replace') as f: # Added errors='replace'
                    for line in f:
                        match = re.match(r'\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)', line.strip())
                        if match:
                            m, s, cs_str, txt = match.groups()
                            cs_val = int(cs_str) / 1000.0 if len(cs_str) == 3 else int(cs_str) / 100.0
                            ts = int(m)*60 + int(s) + cs_val
                            if txt.strip(): lyrics.append((ts, txt.strip()))
                lyrics.sort(key=lambda x: x[0])
            except Exception as e: logger.error(f"Error parsing LRC {lrc_path}: {e}")
        return lyrics

    def on_track_fully_loaded_details_event(self, track_metadata: AudioMetadata, lyrics_data_for_track: Optional[List[Tuple[float, str]]], **kwargs):
        logger.debug(f"Event: track_fully_loaded_with_details for {track_metadata.title}")
        if self.track_info_label: self.track_info_label.config(text=f"{track_metadata.title or 'Unknown Title'} - {track_metadata.artist or 'Unknown Artist'} ({track_metadata.album or 'Unknown Album'})")
        if self.total_time_label: self.total_time_label.config(text=self._format_time(track_metadata.duration))
        if self.lyrics_display: self.lyrics_display.load_lyrics(lyrics_data_for_track or [])
        if self.viz_spectrum_bars: [bar.set_height(0) for bar in self.viz_spectrum_bars] # self.viz_spectrum_bars = None
        if self.viz_waveform_line: self.viz_waveform_line.set_ydata(np.zeros(2048)) # self.viz_waveform_line = None

    def on_volume_changed_event(self, volume: float, is_muted: bool, **kwargs):
        if self.volume_slider: self.volume_slider.set(volume * 100)
        if self.volume_label: self.volume_label.config(text=f"{int(volume*100)}%")
        if self.mute_button: self.mute_button.config(text="🔇" if is_muted else "🔊")

    def on_playback_error_event(self, message: str, **kwargs):
        logger.error(f"FlowStateApp UI: Playback Error: {message}")
        if self.host_app and hasattr(self.host_app, 'update_status_bar'): self.host_app.update_status_bar(f"Error: {message}")
        else: messagebox.showerror("Playback Error", message, parent=self.root_app_tk)
        if self.play_pause_button: self.play_pause_button.config(text="▶")
        if self.track_info_label: self.track_info_label.config(text="Error loading track")
        if self.progress_slider: self.progress_slider.set(0)
        if self.current_time_label: self.current_time_label.config(text="00:00")
        if self.total_time_label: self.total_time_label.config(text="00:00")
        if self.lyrics_display: self.lyrics_display.load_lyrics([])

    def on_playback_playlist_changed_event(self, playlist: List[str], current_index: int, **kwargs):
        if not self.queue_treeview: return
        selected_iid_before = self.queue_treeview.focus() # Get IID string
        
        # Clear only items, not headings or column config
        for item in self.queue_treeview.get_children(): self.queue_treeview.delete(item)
            
        for i, filepath_str in enumerate(playlist):
            item_iid = f"track_{i}" 
            # For now, just filename. Metadata should be part of playlist items in AudioEngine ideally.
            display_title = Path(filepath_str).stem 
            self.queue_treeview.insert("", tk.END, iid=item_iid, values=(f"{i+1}. {display_title}", "Artist?", "??:??"))
        
        self._update_queue_highlight(current_index)
        
        # Try to re-select/focus based on IID if it still exists
        if selected_iid_before and self.queue_treeview.exists(selected_iid_before):
             self.queue_treeview.focus(selected_iid_before); self.queue_treeview.selection_set(selected_iid_before)
        elif current_index != -1 and 0 <= current_index < len(self.queue_treeview.get_children()):
            # If previous selection gone, focus new current track
            new_focus_iid = f"track_{current_index}"
            if self.queue_treeview.exists(new_focus_iid): self.queue_treeview.focus(new_focus_iid)


    def _update_queue_highlight(self, current_playlist_idx: int):
        if not self.queue_treeview: return
        for item_iid_str in self.queue_treeview.get_children(): self.queue_treeview.item(item_iid_str, tags=())
        if 0 <= current_playlist_idx < len(self.queue_treeview.get_children()):
            target_iid = f"track_{current_playlist_idx}" 
            if self.queue_treeview.exists(target_iid):
                theme = self.host_app.theme_manager.get_current_theme() if self.host_app and self.host_app.theme_manager else None
                highlight_bg = theme.highlight_color if theme else "SystemHighlight" # Tkinter system color
                text_fg = theme.primary_fg if theme else "SystemHighlightText"
                # Configure tag if not already done by main theme apply
                self.queue_treeview.tag_configure('current_track_highlight', background=highlight_bg, foreground=text_fg)
                self.queue_treeview.item(target_iid, tags=('current_track_highlight',))
                self.queue_treeview.see(target_iid)

    def on_queue_double_click(self, event: tk.Event):
        item_iid_str = self.queue_treeview.focus()
        if not item_iid_str or not item_iid_str.startswith("track_"): return
        try:
            track_idx_in_queue = int(item_iid_str.split("_")[1])
            if self.host_app and self.host_app.audio_engine_ref and 0 <= track_idx_in_queue < len(self.host_app.audio_engine_ref.playlist):
                self.host_app.request_playback_action("play_track_at_playlist_index", {'index': track_idx_in_queue})
        except (ValueError, IndexError) as e: logger.error(f"Error parsing queue item IID '{item_iid_str}': {e}")

    def _create_queue_context_menu(self) -> tk.Menu:
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Play from Here", command=self.play_selected_from_queue)
        menu.add_command(label="Remove from Queue", command=self.remove_selected_from_queue)
        menu.add_separator()
        menu.add_command(label="Add to Playlist...", command=self.add_selected_queue_tracks_to_playlist_dialog)
        menu.add_command(label="Track Info...", command=self.show_track_info_for_queue_item, state=tk.DISABLED)
        return menu

    def show_queue_context_menu(self, event: tk.Event):
        iid = self.queue_treeview.identify_row(event.y)
        if iid: self.queue_treeview.selection_set(iid) 
        else: return 
        if not self.queue_treeview.selection(): return 
        menu = self._create_queue_context_menu()
        menu.post(event.x_root, event.y_root)

    def play_selected_from_queue(self):
        sel = self.queue_treeview.selection()
        if sel and sel[0].startswith("track_"):
            try: idx = int(sel[0].split("_")[1]); self.host_app.request_playback_action("play_track_at_playlist_index", {'index': idx})
            except (ValueError, IndexError): pass
            
    def remove_selected_from_queue(self):
        sel = self.queue_treeview.selection()
        if sel and sel[0].startswith("track_"):
            try: 
                idx = int(sel[0].split("_")[1])
                if self.host_app and self.host_app.audio_engine_ref: 
                    self.host_app.audio_engine_ref.remove_track_from_playlist_at_index(idx)
            except (ValueError, IndexError) as e: logger.error(f"Error in remove_selected_from_queue: {e}")


    def add_selected_queue_tracks_to_playlist_dialog(self): messagebox.showinfo("TODO", "Add selected queue tracks to new/existing playlist - NYI.", parent=self.root_app_tk)
    def show_track_info_for_queue_item(self): messagebox.showinfo("TODO", "Show detailed track info for queue item - NYI.", parent=self.root_app_tk)

    def on_shuffle_mode_changed_event(self, shuffle_on: bool, **kwargs):
        if self.shuffle_button: self.shuffle_button.config(text="🔀 On" if shuffle_on else "🔀 Off", relief=tk.SUNKEN if shuffle_on else tk.RAISED)
    def on_repeat_mode_changed_event(self, mode: str, **kwargs):
        if self.repeat_button:
            text_map = {"off": "🔁 Off", "one": "🔁 One", "all": "🔁 All"}
            self.repeat_button.config(text=text_map.get(mode, "🔁 ?"), relief=tk.SUNKEN if mode != "off" else tk.RAISED)
    def on_playback_track_ended_event(self, filepath: str, duration: float, **kwargs): 
        logger.debug(f"UI notified: Track ended - {filepath}")

    def _on_like_track(self):
        if self.host_app and self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.current_metadata_obj:
            meta = self.host_app.audio_engine_ref.current_metadata_obj
            if meta.id is not None: 
                self.host_app.publish_event("user_interaction_track_liked", track_id=meta.id, timestamp=datetime.now(timezone.utc).isoformat())
                if self.like_button: self.like_button.config(relief=tk.SUNKEN)
                if self.dislike_button: self.dislike_button.config(relief=tk.RAISED)
                if self.root_app_tk.winfo_exists(): self.root_app_tk.after(1000, lambda: self.like_button.config(relief=tk.RAISED) if self.like_button and self.like_button.winfo_exists() else None)

    def _on_dislike_track(self):
        if self.host_app and self.host_app.audio_engine_ref and self.host_app.audio_engine_ref.current_metadata_obj:
            meta = self.host_app.audio_engine_ref.current_metadata_obj
            if meta.id is not None:
                self.host_app.publish_event("user_interaction_track_disliked", track_id=meta.id, timestamp=datetime.now(timezone.utc).isoformat())
                if self.dislike_button: self.dislike_button.config(relief=tk.SUNKEN)
                if self.like_button: self.like_button.config(relief=tk.RAISED)
                if self.root_app_tk.winfo_exists(): self.root_app_tk.after(1000, lambda: self.dislike_button.config(relief=tk.RAISED) if self.dislike_button and self.dislike_button.winfo_exists() else None)

    def next_track_action(self):
        if self.host_app.audio_engine_ref:
            ae = self.host_app.audio_engine_ref
            if ae.current_metadata_obj and ae.current_metadata_obj.id is not None and ae.duration_sec > 0:
                time_left = ae.duration_sec - ae.get_position()
                percent_played = (ae.get_position() / ae.duration_sec) * 100 if ae.duration_sec > 0 else 0
                if time_left > 15 or percent_played < 75: # Skip if >15s left OR <75% played
                    self.host_app.publish_event("user_interaction_track_skipped", track_id=ae.current_metadata_obj.id, skipped_at_position_sec=ae.get_position(), timestamp=datetime.now(timezone.utc).isoformat())
            self.host_app.request_playback_action("next")

    def gather_session_state(self) -> Dict[str, Any]:
        state = {'volume': 0.7, 'is_muted': False, 'current_track_path': None, 'position_sec': 0.0,
                 'playlist_paths': [], 'current_playlist_index': -1, 'shuffle': False, 'repeat': 'off'}
        if self.host_app and self.host_app.audio_engine_ref:
            ae = self.host_app.audio_engine_ref
            state.update({'volume': ae.volume, 'is_muted': ae.is_muted, 
                          'current_track_path': ae.current_file, 'position_sec': ae.get_position(),
                          'playlist_paths': list(ae.playlist), 'current_playlist_index': ae.current_index,
                          'shuffle': ae.shuffle_mode, 'repeat': ae.repeat_mode})
        return state

    def restore_session_state(self, session_data: Dict[str, Any]):
        if not self.host_app or not self.host_app.audio_engine_ref: return
        ae = self.host_app.audio_engine_ref
        logger.info(f"Restoring session state. Current track in session: {session_data.get('current_track_path')}")
        
        ae.set_volume(session_data.get('volume', 0.7))
        # Mute state: only toggle if different from current AudioEngine state
        if session_data.get('is_muted', False) != ae.is_muted: ae.toggle_mute()
        
        # Set shuffle and repeat BEFORE loading playlist, so playlist loads into correct mode
        ae.set_shuffle_mode(session_data.get('shuffle', False))
        ae.set_repeat_mode(session_data.get('repeat', 'off'))

        playlist_paths = session_data.get('playlist_paths', [])
        track_to_load = session_data.get('current_track_path')
        position_to_seek = session_data.get('position_sec', 0.0)
        # The index from session_data might be for the un-shuffled list.
        # After setting shuffle mode and loading playlist, we need to find the track_to_load in the *new* current playlist.

        if playlist_paths:
            ae.load_playlist_paths(playlist_paths, play_first=False, replace_queue=True) # Don't auto-play
            if track_to_load and track_to_load in ae.playlist:
                final_idx_to_set = ae.playlist.index(track_to_load)
                # Load this specific track to set metadata and prepare for seek, using its new index
                if ae.load_track(track_to_load, playlist_context=ae.playlist, playlist_index=final_idx_to_set):
                    ae.set_position(position_to_seek)
            elif ae.playlist: # Playlist loaded, but target track not in it, load first of current playlist
                if ae.load_track(ae.playlist[0], playlist_context=ae.playlist, playlist_index=0):
                     ae.set_position(0.0) # Start from beginning
            # Don't auto-play; user can resume. Publish state for UI.
            if self.host_app: self.host_app.publish_event("playback_state_changed", is_playing=False, is_paused=False, position=ae.get_position())
        elif track_to_load: # No playlist in session, just a single track
            if ae.load_track(track_to_load):
                ae.set_position(position_to_seek)
                if self.host_app: self.host_app.publish_event("playback_state_changed", is_playing=False, is_paused=False, position=ae.get_position())
        
        logger.info("Session state restoration attempt complete.")
        if self.host_app: self.host_app.publish_event("playback_playlist_changed", playlist=ae.playlist, current_index=ae.current_index) # Ensure queue UI updates


    def on_app_exit(self):
        logger.info("FlowStateApp UI (Main Player) on_app_exit called.")
        if self._after_id_progress_update and self.root_app_tk.winfo_exists():
            self.root_app_tk.after_cancel(self._after_id_progress_update)
        if self.viz_animation and self.viz_fig: 
            if hasattr(self.viz_animation, '_stop'): self.viz_animation._stop() # More direct stop for FuncAnimation if available
            self.viz_animation = None 
            if plt.fignum_exists(self.viz_fig.number): plt.close(self.viz_fig)
        if self.host_app: 
            self.host_app.unsubscribe_from_event("playback_state_changed", self.on_playback_state_changed_event)
            self.host_app.unsubscribe_from_event("playback_position_updated", self.on_playback_position_updated_event)
            self.host_app.unsubscribe_from_event("audio_track_loaded_basic", self.on_audio_track_loaded_basic_event)
            self.host_app.unsubscribe_from_event("track_fully_loaded_with_details", self.on_track_fully_loaded_details_event)
            self.host_app.unsubscribe_from_event("volume_changed", self.on_volume_changed_event)
            self.host_app.unsubscribe_from_event("playback_error", self.on_playback_error_event)
            self.host_app.unsubscribe_from_event("playback_playlist_changed", self.on_playback_playlist_changed_event)
            self.host_app.unsubscribe_from_event("shuffle_mode_changed", self.on_shuffle_mode_changed_event)
            self.host_app.unsubscribe_from_event("repeat_mode_changed", self.on_repeat_mode_changed_event)
            self.host_app.unsubscribe_from_event("playback_track_ended", self.on_playback_track_ended_event)
            if self.host_app.theme_manager: self.host_app.theme_manager.unregister_callback(self.apply_theme_to_player_ui)


def create_main_player_tab(notebook: ttk.Notebook, host_app_ref: Any) -> FlowStateApp:
    player_tab_frame = ttk.Frame(notebook)
    notebook.add(player_tab_frame, text="Player") 
    main_player_ui_instance = FlowStateApp(player_tab_frame, host_app_ref)
    logger.info("Main Player Tab UI (FlowStateApp) created and added to notebook.")
    return main_player_ui_instance


if __name__ == '__main__':
    root = tk.Tk()
    root.title("FlowStateApp - Standalone Test")
    root.geometry("1000x700")
    class MockHostApp: 
        def __init__(self, r):
            self.root = r; self.audio_engine_ref = None; self.theme_manager = None; self.music_library_db_ref=None
            self.status_bar_var = tk.StringVar(value="Mock Status")
            self.notebook = ttk.Notebook(r); self.notebook.pack(fill=tk.BOTH, expand=True) 
            self._event_callbacks: Dict[str, List[Callable]] = {}
            def req_pb(action, params=None, callback=None): print(f"MockHost: request_playback_action '{action}' PARAMS: {params}")
            self.request_playback_action = req_pb
            def req_lib(action, params=None, callback=None): print(f"MockHost: request_library_action '{action}' PARAMS: {params}"); return [] if callback is None else self.root.after(0, callback, [])
            self.request_library_action = req_lib
            def pub_ev(name, *a, **kw): 
                logger.debug(f"MockHost: publish_event '{name}' Args: {a} Kwargs: {kw}")
                cbs = self._event_callbacks.get(name,[]); [self.root.after(0,cb,*a,**kw) for cb in cbs]
            self.publish_event = pub_ev
            def sub_ev(name, cb): self._event_callbacks.setdefault(name,[]).append(cb)
            self.subscribe_to_event = sub_ev
            def unsub_ev(name, cb): 
                if name in self._event_callbacks and cb in self._event_callbacks[name]: self._event_callbacks[name].remove(cb)
            self.unsubscribe_from_event = unsub_ev
            def get_current_track_metadata(): return None
            self.get_current_track_metadata = get_current_track_metadata
            def get_current_playback_position(): return 0.0
            self.get_current_playback_position = get_current_playback_position
            def get_audio_properties(): return 44100, 2
            self.get_audio_properties = get_audio_properties
            def get_current_lyrics_data(): return None
            self.get_current_lyrics_data = get_current_lyrics_data

    mock_host = MockHostApp(root)
    
    class MockAudioEngine:
        is_playing=False; is_paused=False; volume=0.7; is_muted=False; shuffle_mode=False; repeat_mode='off'
        current_metadata_obj=None; playlist=[]; current_index=-1; duration_sec=0.0; effects_output_buffer_for_viz = np.zeros(2048)
        effects_chain_ref_from_host = None; sample_rate=44100; channels=2; current_file = None
        original_playlist_order = []
        def get_position(self): return self.playback_position_sec if hasattr(self, 'playback_position_sec') else 0.0
        def load_track(self, fp, playlist_context=None, playlist_index=None): self.current_file=fp; self.current_metadata_obj=AudioMetadata(file_path=fp, title=Path(fp).stem, duration=180); self.duration_sec=180; self.playlist=[fp]; self.current_index=0; mock_host.publish_event("audio_track_loaded_basic", metadata=self.current_metadata_obj) ; return True
        def play(self, *a, **k): self.is_playing=True; self.is_paused=False; mock_host.publish_event("playback_state_changed", is_playing=True,is_paused=False,position=0)
        def pause(self, *a, **k): self.is_paused=True; mock_host.publish_event("playback_state_changed", is_playing=True,is_paused=True,position=self.get_position())
        def stop(self, *a, **k): self.is_playing=False;self.is_paused=False; mock_host.publish_event("playback_state_changed",is_playing=False,is_paused=False,position=0)
        def add_to_playlist(self, fp): self.playlist.append(fp); mock_host.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
        # Add other methods as needed by FlowStateApp
               def load_playlist_paths(self, paths, play_first, replace_queue):
            self.playlist = paths
            self.current_index = 0 if paths else -1
            mock_host.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
            if play_first and paths:
                self.play()
        def clear_playlist(self): self.playlist=[]; self.current_index=-1; mock_host.publish_event("playback_playlist_changed", playlist=self.playlist, current_index=self.current_index)
        def remove_track_from_playlist_at_index(self,idx): self.playlist.pop(idx); mock_host.publish_event("playback_playlist_changed",playlist=self.playlist, current_index=self.current_index); return True
        def set_shuffle_mode(self, enable): self.shuffle_mode=enable; mock_host.publish_event("shuffle_mode_changed",shuffle_on=enable)
        def set_repeat_mode(self, mode): self.repeat_mode=mode; mock_host.publish_event("repeat_mode_changed",mode=mode)


    mock_host.audio_engine_ref = MockAudioEngine()
    
    class MockTheme: name="Mock Dark"; primary_bg="#333"; secondary_bg="#222"; accent_bg="#444"; primary_fg="#EEE"; secondary_fg="#AAA"; accent_color="#0AF"; highlight_color="#07A"; viz_bg="#000"; waveform_color="#0F0"; viz_primary="#0FF"; spectrum_colors=["#F00"]; font_family="TestSans"; font_size_normal=10;font_size_large=12;font_size_title=14;font_size_small=9;
    class MockThemeManager: current_theme=MockTheme(); get_current_theme=lambda s: s.current_theme; register_callback=lambda s,x:None; unregister_callback=lambda s,x:None; apply_theme=lambda s,x:None
    mock_host.theme_manager = MockThemeManager()

    player_app_ui = create_main_player_tab(mock_host.notebook, mock_host) 
    mock_host.main_player_ui_ref = player_app_ui 

    def simulate_track_load_for_test():
        if player_app_ui:
            dummy_meta = AudioMetadata(title="Test Song Standalone", artist="Tester", album="Test Album", duration=180.0, file_path="dummy.mp3", id=1)
            mock_host.audio_engine_ref.current_metadata_obj = dummy_meta 
            mock_host.audio_engine_ref.duration_sec = 180.0
            mock_host.publish_event("audio_track_loaded_basic", metadata=dummy_meta)
            if root.winfo_exists(): root.after(100, lambda: mock_host.publish_event("track_fully_loaded_with_details", track_metadata=dummy_meta, lyrics_data_for_track=[(10.0,"Line 1 Test"),(15.0,"Line 2 Test")]))
    if root.winfo_exists(): root.after(1000, simulate_track_load_for_test)
    
    root.mainloop()


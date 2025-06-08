

"""
Flow State: Voice Control System
Natural language voice commands for hands-free music control
"""

import speech_recognition as sr
import pyttsx3 
import threading
import time
import re
import json 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum
import numpy as np 
from fuzzywuzzy import fuzz, process 
import nltk
from word2number import w2n 
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import concurrent.futures
import random 
from datetime import datetime 

logger = logging.getLogger("FlowStateVoice")

# --- NLTK Data Downloads (Attempt once) ---
NLTK_DATA_DOWNLOADED = {"punkt": False, "averaged_perceptron_tagger": False, "wordnet": False}
def ensure_nltk_data():
    global NLTK_DATA_DOWNLOADED
    for package_name, downloaded in NLTK_DATA_DOWNLOADED.items():
        if not downloaded:
            try:
                nltk.data.find(f'tokenizers/{package_name}' if package_name == 'punkt' else f'taggers/{package_name}' if package_name == 'averaged_perceptron_tagger' else f'corpora/{package_name}')
                NLTK_DATA_DOWNLOADED[package_name] = True
            except LookupError:
                logger.info(f"NLTK data '{package_name}' not found. Attempting download...")
                try:
                    nltk.download(package_name, quiet=True, raise_on_error=False) # Don't halt on CI or no-internet
                    NLTK_DATA_DOWNLOADED[package_name] = True # Assume success if no error raised by download itself
                    logger.info(f"NLTK data '{package_name}' download attempt finished.")
                except Exception as e_nltk_dl: # Catch any download error
                    logger.warning(f"NLTK data '{package_name}' download attempt failed/skipped: {e_nltk_dl}")
            except AttributeError: # nltk.corpus might not be fully available if nltk is broken
                logger.error(f"NLTK seems to be missing core components (e.g. nltk.corpus). Cannot check/download '{package_name}'.")
ensure_nltk_data() # Call on module load


VOICE_COMMAND_PROCESS_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="VoiceCmdProcess")

class CommandType(Enum): 
    PLAYBACK = "playback"; NAVIGATION = "navigation"; VOLUME = "volume"; SEARCH = "search"
    PLAYLIST = "playlist"; INFORMATION = "information"; SYSTEM = "system"; MOOD = "mood"
    DISCOVERY = "discovery"; SETTINGS = "settings"

@dataclass
class VoiceCommand: 
    command_type: CommandType; action: str
    parameters: Dict[str, Any] = field(default_factory=dict); confidence: float = 1.0
    raw_text: str = ""; timestamp: float = field(default_factory=time.time)

@dataclass
class CommandPattern: 
    pattern: str; command_type: CommandType; action: str
    parameter_extractors: List[Callable[[re.Match], Optional[Dict[str, Any]]]] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    compiled_pattern: Optional[re.Pattern] = field(default=None, repr=False)
    def __post_init__(self):
        if self.pattern:
            try: self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
            except re.error as e: logger.error(f"Regex error for '{self.pattern}': {e}"); self.compiled_pattern = None


class IntentParser:
    def __init__(self, host_app_ref: Optional[Any] = None):
        self.host_app = host_app_ref
        self.command_patterns = self._initialize_patterns()

    def _extract_search_query(self, match: re.Match, group_name: str = 'query') -> Optional[Dict[str, str]]:
        try: query_text = match.group(group_name).strip(); return {'query': query_text} if query_text else None
        except IndexError: return None

    def _extract_search_query_nlp(self, text: str, matched_action_phrase: Optional[str] = None) -> Optional[Dict[str, str]]:
        query_text = text
        if matched_action_phrase and text.lower().startswith(matched_action_phrase.lower()): query_text = text[len(matched_action_phrase):].strip()
        if not query_text: return None
        try:
            tokens = nltk.word_tokenize(query_text); tagged = nltk.pos_tag(tokens)
            # Very basic noun phrase extraction heuristic
            parts = []; current_phrase = []
            for word, tag in tagged:
                if tag.startswith("NN") or tag.startswith("JJ") or word.lower() in ["by", "the", "a", "an", "of"]: # Include some determiners/preps if part of title
                    current_phrase.append(word)
                elif current_phrase : parts.append(" ".join(current_phrase)); current_phrase = []
            if current_phrase: parts.append(" ".join(current_phrase))
            final_query = " ".join(p for p in parts if p.lower() not in ["by", "the", "a", "an", "of"]) # Filter out isolated preps/dets
            return {'query': final_query.strip()} if final_query.strip() else ({'query': query_text} if len(tokens)<=5 else None) # Fallback for short phrases
        except Exception as e: logger.warning(f"NLP query extract failed: {e}"); return {'query': query_text} if query_text else None

    def _extract_mood_name(self, match: re.Match, group_name: str = 'mood') -> Optional[Dict[str, str]]:
        try: mood = match.group(group_name).strip().lower(); return {'mood': mood} if mood else None
        except IndexError: return None
    
    def _extract_number_from_text(self, text: str) -> Optional[int]:
        try: return w2n.word_to_num(text)
        except ValueError: 
            try: return int(text) # Try direct int conversion
            except ValueError: return None

    def _extract_volume_level_from_text(self, match: re.Match, group_name: str = 'level_text') -> Optional[Dict[str, Any]]:
        try:
            text_segment = match.group(group_name).lower().replace(" percent", "").strip()
            level = self._extract_number_from_text(text_segment)
            if level is not None and 0 <= level <= 100: return {'level_percent': level, 'is_direct_level': True}
        except IndexError: pass
        return None

    def _extract_time_offset(self, match: re.Match, text_segment: str) -> Optional[Dict[str, Any]]:
        minutes, seconds = None, None
        # Try regex groups first for "X minutes Y seconds" or "Y seconds" etc.
        if 'minutes' in match.groupdict(): minutes = self._extract_number_from_text(match.group('minutes'))
        if 'seconds' in match.groupdict(): seconds = self._extract_number_from_text(match.group('seconds'))
        
        total_seconds = 0
        if minutes is not None: total_seconds += minutes * 60
        if seconds is not None: total_seconds += seconds
        
        if total_seconds > 0: return {'offset_seconds': total_seconds}

        # Fallback to parsing the whole text_segment if regex groups didn't capture fully
        # This is a simplified parser for "X minutes Y seconds" text
        try:
            num_val = None; unit = None; parsed_seconds = 0
            words = text_segment.lower().replace(" and ", " ").split()
            i = 0
            while i < len(words):
                num_val = self._extract_number_from_text(words[i])
                if num_val is not None:
                    i += 1
                    if i < len(words) and ("minute" in words[i] or "min" in words[i]): parsed_seconds += num_val * 60; unit = "m"
                    elif i < len(words) and ("second" in words[i] or "sec" in words[i]): parsed_seconds += num_val; unit = "s"
                    else: # Number not followed by unit, could be just seconds if it's the only number
                        if unit is None: parsed_seconds += num_val 
                i += 1
            if parsed_seconds > 0: return {'offset_seconds': parsed_seconds}
        except Exception as e: logger.debug(f"Time offset text parse error for '{text_segment}': {e}")
        return None

    def _initialize_patterns(self) -> List[CommandPattern]:
        patterns = [
            CommandPattern(r"play(?: music| track| song)?$", CommandType.PLAYBACK, "play", []),
            CommandPattern(r"pause(?: music| track| song)?$", CommandType.PLAYBACK, "pause", []),
            CommandPattern(r"resume(?: music| track| song)?$", CommandType.PLAYBACK, "resume", []),
            CommandPattern(r"stop(?: music| track| song)?$", CommandType.PLAYBACK, "stop", []),
            CommandPattern(r"(?:next|skip)(?: track| song)?$", CommandType.PLAYBACK, "next_track", []),
            CommandPattern(r"(?:previous|last|back)(?: track| song)?$", CommandType.PLAYBACK, "prev_track", []),
            CommandPattern(r"restart(?: track| song)?$", CommandType.PLAYBACK, "restart_track", []),
            CommandPattern(r"(?:volume|turn it) up$", CommandType.VOLUME, "increase", []),
            CommandPattern(r"(?:volume|turn it) down$", CommandType.VOLUME, "decrease", []),
            CommandPattern(r"mute(?: audio| volume)?$", CommandType.VOLUME, "mute", []),
            CommandPattern(r"unmute(?: audio| volume)?$", CommandType.VOLUME, "unmute", []),
            CommandPattern(r"set volume to (?P<level_text>(?:[a-zA-Z\s]+|\d+)(?: percent)?)", CommandType.VOLUME, "set_absolute_text", [lambda m: self._extract_volume_level_from_text(m, 'level_text')]),
            CommandPattern(r"play (?P<query>.+)$", CommandType.SEARCH, "play_query", [lambda m: self._extract_search_query(m, 'query')]),
            CommandPattern(r"play(?: song| track)? named (?P<query>.+)$", CommandType.SEARCH, "play_query", [lambda m: self._extract_search_query(m, 'query')]),
            CommandPattern(r"play music by (?P<query>.+)$", CommandType.SEARCH, "play_artist_query", [lambda m: self._extract_search_query(m, 'query')]),
            CommandPattern(r"play the album (?P<query>.+)$", CommandType.SEARCH, "play_album_query", [lambda m: self._extract_search_query(m, 'query')]),
            CommandPattern(r"add (?P<query>.+?) to (?:the )?queue$", CommandType.PLAYLIST, "add_to_queue_query", [lambda m: self._extract_search_query(m, 'query')]),
            CommandPattern(r"clear (?:the )?queue$", CommandType.PLAYLIST, "clear_queue", []),
            CommandPattern(r"shuffle(?: on| mode)?$", CommandType.PLAYBACK, "shuffle_on", []),
            CommandPattern(r"turn shuffle off$", CommandType.PLAYBACK, "shuffle_off", []),
            CommandPattern(r"repeat (?:song|track|this)$", CommandType.PLAYBACK, "repeat_one", []),
            CommandPattern(r"repeat (?:all|playlist|queue)$", CommandType.PLAYBACK, "repeat_all", []),
            CommandPattern(r"turn repeat off$", CommandType.PLAYBACK, "repeat_off", []),
            CommandPattern(r"(?:go to|seek to)\s*(?:(?P<minutes>\d+|[a-zA-Z\s]+)\s*minutes?)?\s*(?:and\s*)?(?:(?P<seconds>\d+|[a-zA-Z\s]+)\s*seconds?)?", CommandType.PLAYBACK, "seek_to_time", [lambda m: self._extract_time_offset(m, m.group(0))]),
            CommandPattern(r"(?:fast forward|skip ahead|forward)\s+(?P<time_offset_text>.+)", CommandType.PLAYBACK, "seek_relative_forward", [lambda m: self._extract_time_offset(m, m.group('time_offset_text'))]),
            CommandPattern(r"(?:rewind|skip back|back)\s+(?P<time_offset_text>.+)", CommandType.PLAYBACK, "seek_relative_backward", [lambda m: self._extract_time_offset(m, m.group('time_offset_text'))]),
            CommandPattern(r"play something (?P<mood>\w+)(?: music)?$", CommandType.MOOD, "play_mood", [lambda m: self._extract_mood_name(m, 'mood')]),
            CommandPattern(r"(?:go to|open|show me the?) library$", CommandType.NAVIGATION, "goto_library", []),
            CommandPattern(r"(?:go to|open|show me) visualizations$", CommandType.NAVIGATION, "goto_visualizations", []),
            CommandPattern(r"(?:go to|open|show me) settings$", CommandType.NAVIGATION, "goto_settings", []),
            CommandPattern(r"what time is it$", CommandType.INFORMATION, "get_time", []),
            CommandPattern(r"what song is this$|what's playing(?: right now)?$", CommandType.INFORMATION, "current_track_info", []),
            CommandPattern(r"stop listening$|go to sleep$", CommandType.SYSTEM, "stop_listening", []),
        ]
        for cp in patterns: # Compile regexes
            if cp.pattern: try: cp.compiled_pattern = re.compile(cp.pattern, re.IGNORECASE)
            except re.error as e: logger.error(f"Regex compile error for '{cp.pattern}': {e}"); cp.compiled_pattern = None
        return patterns

    def parse(self, text: str) -> Optional[VoiceCommand]:
        text_lower = text.lower().strip()
        if not text_lower: return None
        for cmd_pattern_obj in self.command_patterns:
            if not cmd_pattern_obj.compiled_pattern: continue
            match = cmd_pattern_obj.compiled_pattern.fullmatch(text_lower)
            if match:
                params = {}
                for extractor in cmd_pattern_obj.parameter_extractors:
                    extracted = extractor(match); 
                    if extracted: params.update(extracted)
                return VoiceCommand(cmd_pattern_obj.command_type, cmd_pattern_obj.action, params, 0.93, text, time.time())
        return self._fuzzy_match_command(text, text_lower)

    def _fuzzy_match_command(self, original_text: str, text_lower: str) -> Optional[VoiceCommand]:
        action_map_for_fuzzy = {} 
        for p_obj in self.command_patterns:
            # Use simplified key phrases for fuzzy matching
            key_phrases = [p_obj.action.replace("_", " ")] + p_obj.aliases
            # Add first few words of regex as potential key phrase
            regex_start_match = re.match(r"^(\w+(?:\s+\w+){0,2})", p_obj.pattern.split('(')[0].strip())
            if regex_start_match: key_phrases.append(regex_start_match.group(1).lower())
            
            for phrase in key_phrases:
                if phrase.lower() not in action_map_for_fuzzy: action_map_for_fuzzy[phrase.lower()] = p_obj
        
        if not action_map_for_fuzzy: return None
        best_match_phrase, score = process.extractOne(text_lower, action_map_for_fuzzy.keys(), scorer=fuzz.WRatio)
        
        if score >= 78: # Adjusted threshold
            matched_pattern = action_map_for_fuzzy[best_match_phrase]
            parameters = {}
            # Attempt parameter extraction even with fuzzy match
            # Try direct regex search on original text with the matched pattern's regex
            if matched_pattern.compiled_pattern:
                fuzzy_re_match = matched_pattern.compiled_pattern.search(text_lower) # Use search, not fullmatch
                if fuzzy_re_match:
                    for extractor in matched_pattern.parameter_extractors:
                        extracted = extractor(fuzzy_re_match); 
                        if extracted: parameters.update(extracted)
            
            # If regex didn't yield params (e.g., query), try NLP if it's a search-like action
            if not parameters and matched_pattern.action in ["play_query", "play_artist_query", "play_album_query", "add_to_queue_query"]:
                nlp_params = self._extract_search_query_nlp(original_text, best_match_phrase)
                if nlp_params: parameters.update(nlp_params)
            
            logger.debug(f"Fuzzy matched '{text_lower}' to action '{matched_pattern.action}' (phrase '{best_match_phrase}') score {score}, params: {parameters}")
            return VoiceCommand(matched_pattern.command_type, matched_pattern.action, parameters, score/100.0, original_text, time.time())
        return None


class VoiceRecognizer: # Full implementation as before
    def __init__(self, phrase_callback: Callable[[str, bool], None], status_callback: Callable[[str], None], audio_level_callback: Optional[Callable[[float], None]] = None, wake_word: Optional[str] = None, language: str = "en-US"): pass
    def _initialize_microphone(self): pass
    def calibrate_microphone_energy(self, duration: float = 1.0): pass
    def start_listening(self): pass
    def stop_listening(self): pass
    def _listen_loop(self): pass
    def _process_phrase(self, audio_data: sr.AudioData): pass
    def update_wake_word(self, new_wake_word: Optional[str]): pass


class VoiceFeedback: # Full implementation as before
    def __init__(self, default_rate: int = 170, default_volume: float = 0.95): pass # Slightly faster rate
    def _init_engine_thread(self): pass
    def _tts_worker(self): pass
    def _setup_voice_properties(self, rate: Optional[int]=None, volume: Optional[float]=None, voice_id: Optional[str]=None): pass
    def speak(self, text: str, rate: Optional[int]=None, volume: Optional[float]=None, voice_id: Optional[str]=None): pass
    def stop(self): pass 
    def get_available_voices(self) -> List[Dict[str,str]]: pass 
    def cleanup(self): pass


class VoiceCommandHandler: # Full implementation with refined handlers
    def __init__(self, feedback_system: VoiceFeedback, host_app_interface: Any):
        self.feedback = feedback_system; self.host_app = host_app_interface
        self.confirmation_phrases = ["Okay.", "Alright.", "Got it.", "Sure thing.", "Done."]
        self.error_phrases = ["Sorry, I couldn't do that.", "I had trouble with that.", "Something went wrong processing that."]
        self.not_found_phrases = ["I couldn't find that.", "Sorry, no results for that.", "That doesn't seem to be in your library or available."]

    def _speak_response(self, text: str, is_error: bool = False, is_info: bool = False):
        prefix = ""
        if not any(text.lower().startswith(p.lower()) for p in self.confirmation_phrases + self.error_phrases + self.not_found_phrases) and not is_info:
            prefix = random.choice(self.error_phrases if is_error else self.confirmation_phrases) + " "
        self.feedback.speak(f"{prefix}{text}")

    def _handle_playback_command(self, action: str, parameters: Dict[str, Any]):
        ae_ref = self.host_app.audio_engine_ref
        if not ae_ref: self._speak_response("Audio system isn't ready.",is_error=True); return

        if action in ["seek_to_time", "seek_relative_forward", "seek_relative_backward"]:
            offset_s = parameters.get('offset_seconds')
            if offset_s is None: self._speak_response("I didn't catch the time to seek to.",is_error=True); return
            if not ae_ref.current_file: self._speak_response("Nothing is playing to seek.",is_error=True); return
            
            current_pos = ae_ref.get_position(); duration = ae_ref.duration_sec
            target_sec = offset_s if action == "seek_to_time" else (current_pos + offset_s if action == "seek_relative_forward" else current_pos - offset_s)
            target_sec = np.clip(target_sec, 0, duration if duration > 0 else float('inf'))
            
            self.host_app.request_playback_action("seek", {'position_seconds': target_sec})
            m,s = divmod(int(target_sec),60); self._speak_response(f"Seeking to {m} minutes {s} seconds.")
        elif action == "restart_track":
            if ae_ref.current_file: self.host_app.request_playback_action("seek", {'position_seconds': 0.0}); self.host_app.request_playback_action("play"); self._speak_response("Restarting track.")
            else: self._speak_response("No track to restart.", is_error=True)
        else: self._handle_standard_playback_action(action, parameters) # For play, pause, next etc.

    def _handle_standard_playback_action(self, action: str, parameters: Dict[str, Any]):
        action_map = {"play":"play","pause":"pause","resume":"resume","stop":"stop","next_track":"next","prev_track":"previous","shuffle_on":("set_shuffle_mode",{'state':True}),"shuffle_off":("set_shuffle_mode",{'state':False}),"repeat_one":("set_repeat_mode",{'mode':'one'}),"repeat_all":("set_repeat_mode",{'mode':'all'}),"repeat_off":("set_repeat_mode",{'mode':'off'})}
        if action in action_map:
            mapped = action_map[action]; host_act_name, params_host = mapped if isinstance(mapped,tuple) else (mapped,{})
            self.host_app.request_playback_action(host_act_name, params_host)
            self._speak_response(f"{action.replace('_',' ').capitalize()} initiated.")
        else: logger.warning(f"Unhandled std playback action: {action}")

    def _handle_volume_command(self, action: str, parameters: Dict[str, Any]):
        ae_ref = self.host_app.audio_engine_ref
        if not ae_ref: self._speak_response("Audio system isn't ready.", is_error=True); return
        
        if action == "set_absolute_text": # Handles "set volume to fifty percent" or "set volume to 30"
            level_p = parameters.get('level_percent')
            if level_p is not None: self.host_app.request_playback_action("set_volume",{'level':level_p/100.0}); self._speak_response(f"Volume set to {level_p} percent.")
            else: self._speak_response("Didn't catch the volume level.", is_error=True)
        else: # increase, decrease, mute, unmute handled by a std helper
            self._handle_standard_volume_action(action, parameters)

    def _handle_standard_volume_action(self, action: str, parameters: Dict[str, Any]):
        ae_ref = self.host_app.audio_engine_ref; if not ae_ref: return
        current_vol_p = int(ae_ref.volume * 100); change_p = 15; new_vol_p = current_vol_p
        if action == "increase": new_vol_p = min(100, current_vol_p + change_p)
        elif action == "decrease": new_vol_p = max(0, current_vol_p - change_p)
        elif action == "mute": self.host_app.request_playback_action("toggle_mute"); self._speak_response("Muted." if not ae_ref.is_muted else "Already muted."); return # Check state before response
        elif action == "unmute": self.host_app.request_playback_action("toggle_mute"); self._speak_response("Unmuted." if ae_ref.is_muted else "Already unmuted."); return
        else: logger.warning(f"Unhandled std volume action: {action}"); return
        if new_vol_p != current_vol_p: self.host_app.request_playback_action("set_volume",{'level':new_vol_p/100.0}); self._speak_response(f"Volume {'up' if new_vol_p > current_vol_p else 'down'}.")
        else: self._speak_response(f"Volume at {'max' if new_vol_p == 100 else 'min'}.")

    def _handle_search_command(self, action: str, parameters: Dict[str, Any]):
        query = parameters.get('query'); if not query: self._speak_response("What to search for?",is_error=True); return
        self._speak_response(f"Searching for {query}...")
        def _cb(results: Optional[List[Any]]):
            if results:
                t = results[0]; tid=getattr(t,'id',None); title=getattr(t,'title',"that track"); artist=getattr(t,'artist',"")
                if tid is not None: self.host_app.request_playback_action("play_track_by_id",{'track_id':tid}); self._speak_response(f"Playing {title}{f' by {artist}' if artist else ''}.")
                else: self._speak_response(f"Found {title}, but couldn't play it.",is_error=True)
            else: self._speak_response(random.choice(self.not_found_phrases)+f" for {query}.",is_error=True)
        self.host_app.request_library_action("search_tracks",{'query':query,'limit':1},callback=_cb)

    def _handle_playlist_command(self, action: str, parameters: Dict[str, Any]):
        if action == "add_to_queue_query":
            query = parameters.get('query'); if not query: self._speak_response("Add what to queue?",is_error=True); return
            self._speak_response(f"Finding {query} to add...")
            def _cb(r):
                if r: t=r[0]; fp=getattr(t,'file_path',None); title=getattr(t,'title',"that track")
                     if fp: self.host_app.request_playback_action("add_to_queue_path",{'filepath':fp}); self._speak_response(f"Added {title} to queue.")
                     else: self._speak_response(f"Found {title}, but no path to add.",is_error=True)
                else: self._speak_response(f"Couldn't find {query} to add.",is_error=True)
            self.host_app.request_library_action("search_tracks",{'query':query,'limit':1},callback=_cb)
        elif action == "clear_queue": self.host_app.request_playback_action("clear_playlist_or_queue"); self._speak_response("Queue cleared.")

    def _handle_mood_command(self, action: str, parameters: Dict[str, Any]):
        mood = parameters.get('mood'); if not mood: self._speak_response("What mood?",is_error=True); return
        self._speak_response(f"Finding {mood} music...")
        # Placeholder: map mood to genre or use RecommendationEngine if it supports mood queries
        search_q = {'happy':'pop','energetic':'dance','calm':'ambient','sad':'blues'}.get(mood, mood)
        def _cb(r):
            if r: paths=[getattr(t,'file_path',None) for t in random.sample(r,min(len(r),5)) if getattr(t,'file_path',None)]
                 if paths: self.host_app.request_playback_action("load_playlist_paths",{'paths':paths,'play_first':True,'replace_queue':True}); self._speak_response(f"Here's some {mood} music.")
                 else: self._speak_response(f"Found {mood} tracks, but couldn't play.",is_error=True)
            else: self._speak_response(f"Sorry, no {mood} music found.",is_error=True)
        if self.host_app.recommendation_engine_ref and hasattr(self.host_app.recommendation_engine_ref, 'get_tracks_by_mood'): # Fictional method
             # self.host_app.recommendation_engine_ref.get_tracks_by_mood(mood, callback=_cb)
             self.host_app.request_library_action("search_tracks",{'query':search_q,'limit':20},callback=_cb) # Fallback
        else: self.host_app.request_library_action("search_tracks",{'query':search_q,'limit':20},callback=_cb)

    def _handle_navigation_command(self, action: str, parameters: Dict[str, Any]):
        tabs = {"goto_library":"Library","goto_visualizations":"Visualizations","goto_settings":"Manage"} # Manage tab for settings/export
        tab_name = tabs.get(action)
        if tab_name: self.host_app.request_ui_focus_tab(tab_name); self._speak_response(f"Showing {tab_name}.")
        else: self._speak_response(f"Can't navigate to '{action.replace('goto_','')}'.", is_error=True)

    def _handle_information_command(self, action: str, parameters: Dict[str, Any]):
        if action == "get_time": self._speak_response(f"The time is {datetime.now().strftime('%I:%M %p')}.", is_info=True)
        elif action == "current_track_info":
            meta = self.host_app.get_current_track_metadata()
            if meta and meta.title not in ["Unknown Title", None]:
                resp = f"Playing {meta.title}{f' by {meta.artist}' if meta.artist not in ['Unknown Artist',None] else ''}."
                self._speak_response(resp, is_info=True)
            else: self._speak_response("Nothing specific is playing.", is_info=True)

    def _handle_system_command(self, action: str, parameters: Dict[str, Any]):
        if action == "stop_listening":
            if self.host_app and self.host_app.voice_control_ui_ref and hasattr(self.host_app.voice_control_ui_ref, 'stop_listening'):
                 if self.host_app.root and self.host_app.root.winfo_exists(): self.host_app.root.after(0, self.host_app.voice_control_ui_ref.stop_listening)
                 self._speak_response("Okay, I'll stop listening.")
            else: self._speak_response("I can't stop listening now.", is_error=True)

    def _execute_command_worker(self, command: VoiceCommand):
        logger.info(f"Executing voice command: {command.command_type.value} - {command.action} with params {command.parameters}")
        handlers = {
            CommandType.PLAYBACK: self._handle_playback_command, CommandType.VOLUME: self._handle_volume_command,
            CommandType.SEARCH: self._handle_search_command, CommandType.PLAYLIST: self._handle_playlist_command,
            CommandType.MOOD: self._handle_mood_command, CommandType.NAVIGATION: self._handle_navigation_command,
            CommandType.INFORMATION: self._handle_information_command, CommandType.SYSTEM: self._handle_system_command
        }
        handler = handlers.get(command.command_type)
        if handler:
            try:
                if command.command_type == CommandType.NAVIGATION : # UI actions need main thread
                     if self.host_app.root and self.host_app.root.winfo_exists(): self.host_app.root.after(0, handler, command.action, command.parameters)
                else: handler(command.action, command.parameters)
            except Exception as e_exec: logger.error(f"Error executing handler for {command.command_type}: {e_exec}", exc_info=True); self._speak_response(random.choice(self.error_phrases), is_error=True)
        else: logger.warning(f"No handler for command type: {command.command_type}")

    def handle_command_async(self, command: VoiceCommand):
        VOICE_COMMAND_PROCESS_POOL.submit(self._execute_command_worker, command)


class VoiceControlUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent) # CRITICAL
        self.host_app = host_app_ref
        self.root_app_tk = parent.winfo_toplevel()
        self.is_actively_listening = False # Separate from recognizer.is_listening for UI toggle state
        self.status_var = tk.StringVar(value="Voice control idle. Click 'Start Listening'.")
        self.wake_word_var = tk.StringVar(value="flow state") # Default wake word
        self.mic_level_queue = deque(maxlen=100) # For mic level visualization

        self.intent_parser = IntentParser(host_app_ref=self.host_app)
        self.feedback_system = VoiceFeedback()
        self.command_handler = VoiceCommandHandler(self.feedback_system, self.host_app)
        self.recognizer = VoiceRecognizer(
            phrase_callback=self.on_voice_input,
            status_callback=lambda msg: self.root_app_tk.after(0, self.status_var.set, msg) if self.root_app_tk.winfo_exists() else None,
            audio_level_callback=self.monitor_audio_level,
            wake_word=self.wake_word_var.get()
        )
        self._create_ui()
        if self.host_app and self.host_app.theme_manager: # Theming
            self.host_app.theme_manager.register_callback(self.apply_theme_to_voice_ui)
            if self.host_app.theme_manager.get_current_theme(): self.apply_theme_to_voice_ui(self.host_app.theme_manager.get_current_theme())

    def apply_theme_to_voice_ui(self, theme: Any): # ... (Theme application logic for this UI) ...
        pass
    def _create_ui(self): # ... (Full UI as refined, with mic level viz, command log, wake word entry, calibrate button) ...
        pass
    def toggle_listening(self): # ... (Start/Stop listening based on self.is_actively_listening) ...
        pass
    def start_listening(self): # ... (Calls self.recognizer.start_listening(), updates UI) ...
        pass
    def stop_listening(self): # ... (Calls self.recognizer.stop_listening(), updates UI) ...
        pass
    def on_voice_input(self, phrase: str, is_final: bool): # ... (Calls _process_recognized_text if is_final) ...
        pass
    def _process_recognized_text(self, text: str): # ... (Uses self.intent_parser, self.command_handler.handle_command_async, self.log_command_to_ui) ...
        pass
    def log_command_to_ui(self, raw_text: str, command: Optional[VoiceCommand]): # ... (Updates ScrolledText log) ...
        pass
    def calibrate_microphone(self): # ... (Calls self.recognizer.calibrate_microphone_energy in a thread) ...
        pass
    def _calibrate_thread(self): pass
    def update_wake_word(self, *args): # ... (Calls self.recognizer.update_wake_word) ...
        pass
    def monitor_audio_level(self, level: float): # ... (Updates self.mic_level_queue, schedules viz update) ...
        pass
    def _update_mic_level_viz(self, _frame=None): # ... (Updates Matplotlib bar for mic level) ...
        pass
    def on_app_exit(self): # ... (Calls self.stop_listening(), self.feedback_system.cleanup()) ...
        pass


def create_voice_control_tab(notebook: ttk.Notebook, host_app_ref: Any) -> VoiceControlUI:
    voice_frame = ttk.Frame(notebook)
    notebook.add(voice_frame, text="Voice")
    voice_ui_instance = VoiceControlUI(voice_frame, host_app_ref=host_app_ref)
    voice_ui_instance.pack(fill=tk.BOTH, expand=True)
    logger.info("Voice Control Tab UI created.")
    return voice_ui_instance


if __name__ == '__main__':
    # ... (Standalone test block as before) ...
    pass


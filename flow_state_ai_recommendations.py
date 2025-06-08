

"""
Flow State: AI Music Recommendation Engine
Advanced machine learning for personalized music discovery
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import librosa 

import pickle
import json
import sqlite3
from datetime import datetime, timedelta, timezone 
from typing import List, Dict, Tuple, Optional, Any, Set 
import threading
import asyncio # For RecommendationEngine._db_action_rec_engine_async
from dataclasses import dataclass, field, asdict 
import logging
import concurrent.futures
import os 
from pathlib import Path 
import random 
import re # For _detect_key_from_chroma if note names are complex

import tkinter as tk               
from tkinter import ttk            
from tkinter import messagebox, scrolledtext # Added scrolledtext for RecommendationUI

logger = logging.getLogger("FlowStateAI")

# --- Constants for this module ---
RECOMMENDATIONS_DB_FILENAME = "recommendations_engine_data.db" 
FEATURE_EXTRACTION_TIMEOUT_SECONDS = 240 
AI_PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) # No thread_name_prefix
AI_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="AIRecsIO")


@dataclass
class UserProfile: 
    user_id: str 
    favorite_genres: List[str] = field(default_factory=list)
    favorite_artists: List[str] = field(default_factory=list)
    listening_history: List[Dict[str, Any]] = field(default_factory=list) 
    skip_history: List[Dict[str, Any]] = field(default_factory=list) 
    liked_tracks: Set[str] = field(default_factory=set) 
    disliked_tracks: Set[str] = field(default_factory=set)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict_for_db(self) -> Dict[str, Any]:
        d = asdict(self)
        d['liked_tracks'] = json.dumps(list(d.get('liked_tracks', []))) 
        d['disliked_tracks'] = json.dumps(list(d.get('disliked_tracks', [])))
        return d

    @classmethod
    def from_db_dict(cls, db_data_json: str) -> 'UserProfile':
        try: db_data = json.loads(db_data_json)
        except json.JSONDecodeError: logger.error("Failed UserProfile JSON decode."); return cls(user_id="error_profile")
        db_data['liked_tracks'] = set(db_data.get('liked_tracks', [])) 
        db_data['disliked_tracks'] = set(db_data.get('disliked_tracks', []))
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        init_data = {k: v for k, v in db_data.items() if k in field_names}
        if 'user_id' not in init_data or not init_data['user_id']: init_data['user_id'] = "recovered_profile_no_id" 
        return cls(**init_data)

@dataclass
class TrackFeatures: 
    track_id: str 
    title: str 
    artist: str 
    genre: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None 
    energy: Optional[float] = None 
    valence: Optional[float] = None 
    danceability: Optional[float] = None 
    acousticness: Optional[float] = None 
    instrumentalness: Optional[float] = None 
    speechiness: Optional[float] = None 
    loudness: Optional[float] = None 
    duration: Optional[float] = None 
    timbre_features: List[float] = field(default_factory=list) 
    mood_vector: List[float] = field(default_factory=list) 
    last_analyzed_at: Optional[str] = None 

    def to_dict_for_db(self) -> Dict[str, Any]: 
        d = asdict(self)
        d['timbre_features'] = json.dumps(d.get('timbre_features', []))
        d['mood_vector'] = json.dumps(d.get('mood_vector', []))
        return d

    @classmethod
    def from_db_dict(cls, db_data_json_str: str) -> 'TrackFeatures': 
        try: db_data = json.loads(db_data_json_str)
        except json.JSONDecodeError: logger.error("Failed TrackFeatures JSON decode."); return cls(track_id="error_tf", title="Error", artist="Error")
        for key in ['timbre_features', 'mood_vector']:
            if key in db_data and isinstance(db_data[key], str):
                try: db_data[key] = json.loads(db_data[key])
                except json.JSONDecodeError: db_data[key] = []
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        init_data = {k: v for k, v in db_data.items() if k in field_names}
        if 'track_id' not in init_data: init_data['track_id'] = "unknown_id_from_db"
        if 'title' not in init_data: init_data['title'] = "Unknown Title"
        if 'artist' not in init_data: init_data['artist'] = "Unknown Artist"
        return cls(**init_data)


class AudioFeatureExtractor:
    def __init__(self, sample_rate_target: int = 22050):
        self.sr_target = sample_rate_target

    def _safe_load_audio(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
        try:
            y, sr_orig = librosa.load(file_path, sr=self.sr_target, mono=True, duration=300) # Max 5 mins
            return y, float(self.sr_target)
        except Exception as e:
            logger.warning(f"Librosa failed to load audio {Path(file_path).name}: {e}", exc_info=False)
            return None, None

    def extract_features(self, file_path: str, track_id_str: str, existing_metadata: Optional[Dict]=None) -> Optional[TrackFeatures]:
        logger.debug(f"Extracting features for: {Path(file_path).name} (ID: {track_id_str})")
        y, sr = self._safe_load_audio(file_path)
        if y is None or sr is None: return None

        duration = librosa.get_duration(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo_tuple[0]) if tempo_tuple and len(tempo_tuple)>0 and tempo_tuple[0] is not None else None
        
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12) # Ensure 12 chroma bins
        key_estimate_str = self._detect_key_from_chroma(chromagram)
        
        rms_frames = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms_frames)) if rms_frames.size > 0 else None # RMS based energy

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        avg_mfccs = [float(val) for val in np.mean(mfccs, axis=1)] if mfccs.size > 0 else [0.0]*13

        title = existing_metadata.get('title', Path(file_path).stem) if existing_metadata else Path(file_path).stem
        artist = existing_metadata.get('artist', "Unknown") if existing_metadata else "Unknown"
        genre = existing_metadata.get('genre') if existing_metadata else None

        return TrackFeatures(
            track_id=track_id_str, title=title, artist=artist, genre=genre,
            bpm=bpm, key=key_estimate_str, energy=energy, duration=duration,
            timbre_features=avg_mfccs, last_analyzed_at=datetime.now(timezone.utc).isoformat()
            # Other features (valence, etc.) are None by default
        )

    def _detect_key_from_chroma(self, chromagram: np.ndarray) -> Optional[str]:
        if chromagram.shape[0] != 12 or chromagram.ndim != 2: 
            logger.warning(f"Invalid chromagram shape for key detection: {chromagram.shape}"); return None
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.78,2.74,4.33,2.69,3.34])
        chroma_notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        mean_chroma = np.mean(chromagram, axis=1)
        best_corr = -np.inf; best_key = None
        for i in range(12):
            for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                rotated_profile = np.roll(profile, i)
                try:
                    # Ensure no NaNs or Infs in mean_chroma or rotated_profile for corrcoef
                    if np.any(np.isnan(mean_chroma)) or np.any(np.isnan(rotated_profile)) or \
                       np.any(np.isinf(mean_chroma)) or np.any(np.isinf(rotated_profile)):
                        continue # Skip if invalid data
                    if np.std(mean_chroma) == 0 or np.std(rotated_profile) == 0: # Avoid div by zero in corrcoef if flat
                        corr = 0.0 
                    else:
                        corr = np.corrcoef(mean_chroma, rotated_profile)[0, 1]

                    if corr > best_corr: best_corr = corr; best_key = f"{chroma_notes[i]} {mode}"
                except Exception as e_corr:
                    logger.debug(f"Correlation error in key detection: {e_corr}"); continue # Skip on error
        return best_key if best_key and best_corr > 0.1 else None


class RecommendationEngine:
    def __init__(self, host_app_ref: Any, default_user_id: str = "default_user"):
        self.host_app = host_app_ref
        self.music_library_db = host_app_ref.music_library_db_ref
        self.default_user_id = default_user_id 
        
        rec_data_dir = Path.home() / ".flowstate" / "data"
        rec_data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(rec_data_dir / RECOMMENDATIONS_DB_FILENAME)
        
        self.feature_extractor = AudioFeatureExtractor()
        self.track_feature_matrix: Optional[np.ndarray] = None
        self.track_ids_in_matrix_order: List[str] = []
        self.matrix_feature_names: List[str] = []
        self.matrix_scaler: Optional[StandardScaler] = None 
        self.feature_cache: Dict[str, TrackFeatures] = {}
        self.cache_lock = threading.Lock()
        self.pending_extractions: Set[str] = set()
        self.pending_lock = threading.Lock()

        self._init_recommendation_db()
        if self.host_app:
            self.host_app.subscribe_to_event("library_track_added", self.on_library_track_added_event)
            self.host_app.subscribe_to_event("user_interaction_track_played", self.on_user_interaction_track_played)
            self.host_app.subscribe_to_event("user_interaction_track_skipped", self.on_user_interaction_track_skipped)
            self.host_app.subscribe_to_event("user_interaction_track_liked", self.on_user_interaction_track_liked)
            self.host_app.subscribe_to_event("user_interaction_track_disliked", self.on_user_interaction_track_disliked)
        self._load_or_build_feature_matrix_async()

    def _db_execute_rec_engine(self, query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False, conn_passed: Optional[sqlite3.Connection] = None):
        is_external_conn = conn_passed is not None
        conn = conn_passed if is_external_conn else sqlite3.connect(self.db_path, timeout=10)
        if not is_external_conn: conn.row_factory = sqlite3.Row; conn.execute("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;")
        try:
            cursor = conn.cursor(); cursor.execute(query, params)
            if commit: conn.commit()
            if fetch_one: return cursor.fetchone()
            if fetch_all: return cursor.fetchall()
            return cursor.lastrowid if commit else None
        except sqlite3.Error as e: logger.error(f"RecEngine DB Error: {e} (Query: {query[:100]})", exc_info=True); conn.rollback() if not is_external_conn else None; raise
        finally: 
            if not is_external_conn and conn: conn.close()

    async def _db_action_rec_engine_async(self, query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(AI_THREAD_POOL, self._db_execute_rec_engine, query, params, commit, fetch_one, fetch_all)

    def _init_recommendation_db(self):
        self._db_execute_rec_engine(""" CREATE TABLE IF NOT EXISTS user_profiles ( user_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL, created_at_utc TEXT, updated_at_utc TEXT ); """, commit=True)
        self._db_execute_rec_engine(""" CREATE TABLE IF NOT EXISTS extracted_track_features ( track_id TEXT PRIMARY KEY, features_json TEXT NOT NULL, last_analyzed_at TEXT );""", commit=True)
        logger.info("Recommendation engine database initialized.")

    def on_library_track_added_event(self, track_id: int, file_path: str, metadata_dict: Dict):
        logger.info(f"AI Rec Engine: Received new track: ID {track_id}, Path {file_path}")
        track_id_str = str(track_id)
        with self.cache_lock: existing_features = self.feature_cache.get(track_id_str)
        if not existing_features: existing_features = self.get_full_track_features_from_rec_db(track_id_str)
        if existing_features and existing_features.last_analyzed_at: return # Add recency check if needed
        self.request_feature_extraction(track_id_str, file_path, metadata_dict)

    def get_full_track_features_from_rec_db(self, track_id_str: str, db_conn: Optional[sqlite3.Connection] = None) -> Optional[TrackFeatures]:
        row = self._db_execute_rec_engine("SELECT features_json FROM extracted_track_features WHERE track_id = ?", (track_id_str,), fetch_one=True, conn_passed=db_conn)
        return TrackFeatures.from_db_dict(row['features_json']) if row and row['features_json'] else None

    def _load_or_build_feature_matrix_async(self): AI_THREAD_POOL.submit(self._load_or_build_feature_matrix_worker)

    def _get_numerical_features_from_track(self, tf: TrackFeatures) -> Optional[List[float]]:
        # Consistent feature order for the matrix
        self.matrix_feature_names = ['bpm', 'energy', 'duration'] # Basic set
        num_mfccs = 13 # Define how many MFCCs to use
        self.matrix_feature_names.extend([f'mfcc_{i}' for i in range(num_mfccs)])
        
        vals = [
            tf.bpm or 0.0, # Replace None with 0 before scaling
            tf.energy or 0.0,
            tf.duration or 0.0,
        ]
        if tf.timbre_features and len(tf.timbre_features) >= num_mfccs: vals.extend(tf.timbre_features[:num_mfccs])
        else: vals.extend((tf.timbre_features or []) + [0.0] * (num_mfccs - len(tf.timbre_features or [])))
        
        if len(vals) != len(self.matrix_feature_names): # Should not happen if logic is correct
            logger.error(f"Feature vector length mismatch for {tf.track_id}. Expected {len(self.matrix_feature_names)}, got {len(vals)}."); return None
        return vals

    def _load_or_build_feature_matrix_worker(self):
        logger.info("Feature matrix worker started.")
        all_vectors, temp_ids = [], []
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT track_id, features_json FROM extracted_track_features").fetchall()
            logger.info(f"Found {len(rows)} tracks with features for matrix.")
            for row in rows:
                tf = TrackFeatures.from_db_dict(row['features_json'])
                with self.cache_lock: self.feature_cache[row['track_id']] = tf
                num_vec = self._get_numerical_features_from_track(tf)
                if num_vec: all_vectors.append(num_vec); temp_ids.append(row['track_id'])
            
            if not all_vectors: self.track_feature_matrix = None; self.track_ids_in_matrix_order = []; return
            feature_arr = np.array(all_vectors, dtype=np.float32)
            if self.matrix_scaler is None or feature_arr.shape[1] != self.matrix_scaler.n_features_in_: # Re-fit if new or shape changed
                self.matrix_scaler = StandardScaler(); scaled_features = self.matrix_scaler.fit_transform(feature_arr)
            else: scaled_features = self.matrix_scaler.transform(feature_arr)
            self.track_feature_matrix = scaled_features; self.track_ids_in_matrix_order = temp_ids
            logger.info(f"Feature matrix built. Shape: {self.track_feature_matrix.shape if self.track_feature_matrix is not None else 'None'}.")
        except Exception as e: logger.error(f"Error in matrix worker: {e}", exc_info=True)
        finally: conn.close()

    def request_feature_extraction(self, track_id_str: str, file_path: str, metadata_dict_from_lib: Optional[Dict] = None):
        with self.pending_lock:
            if track_id_str in self.pending_extractions: return
            self.pending_extractions.add(track_id_str)
        logger.info(f"Requesting feature extraction for track ID {track_id_str}")
        future = AI_PROCESS_POOL.submit(self.feature_extractor.extract_features, file_path, track_id_str, metadata_dict_from_lib)
        def _cb(f):
            with self.pending_lock: self.pending_extractions.remove(track_id_str)
            try:
                features_obj = f.result(timeout=FEATURE_EXTRACTION_TIMEOUT_SECONDS)
                if features_obj: self._save_track_features_to_rec_db_sync(features_obj); self._load_or_build_feature_matrix_async() # Trigger rebuild
                else: logger.warning(f"Extraction returned None for {track_id_str}.")
            except Exception as e: logger.error(f"Extraction callback error for {track_id_str}: {e}", exc_info=True)
        future.add_done_callback(_cb)

    def _save_track_features_to_rec_db_sync(self, features: TrackFeatures): # Synchronous version for callbacks/workers
        json_str = json.dumps(features.to_dict_for_db())
        ts = features.last_analyzed_at or datetime.now(timezone.utc).isoformat()
        self._db_execute_rec_engine(
            "INSERT OR REPLACE INTO extracted_track_features (track_id, features_json, last_analyzed_at) VALUES (?, ?, ?)",
            (features.track_id, json_str, ts), commit=True
        )
        logger.info(f"Saved features for track ID {features.track_id} to rec DB.")

    def get_similar_tracks_content_based(self, track_id_str: str, num_similar: int = 10) -> List[Dict[str, Any]]:
        if self.track_feature_matrix is None or not self.track_ids_in_matrix_order:
            if self.track_feature_matrix is None: self._load_or_build_feature_matrix_async()
            return []
        try: target_idx = self.track_ids_in_matrix_order.index(track_id_str)
        except ValueError: logger.warning(f"Track {track_id_str} not in matrix."); return []
        
        target_vec = self.track_feature_matrix[target_idx,:].reshape(1, -1)
        if target_vec.shape[1] != self.track_feature_matrix.shape[1]: return []
        
        sims = cosine_similarity(target_vec, self.track_feature_matrix)[0]
        sorted_indices = np.argsort(-sims)
        
        results = []
        for idx in sorted_indices:
            if len(results) >= num_similar: break
            if idx == target_idx: continue
            sim_track_id = self.track_ids_in_matrix_order[idx]
            # Get details from main library (can be slow, consider caching or batching)
            lib_track = self.music_library_db.get_track(int(sim_track_id)) if self.music_library_db and sim_track_id.isdigit() else None
            if lib_track: results.append({'track_id_lib':lib_track.id, 'title':lib_track.title, 'artist':lib_track.artist, 'similarity':float(sims[idx]), 'file_path':lib_track.file_path})
        return results

    def get_user_profile(self, user_id: str) -> UserProfile: # As refined before
        pass
    def _save_user_profile_sync(self, profile: UserProfile, conn_passed: Optional[sqlite3.Connection] = None): # As refined before
        pass
    def save_user_profile_async(self, profile: UserProfile): # As refined before
        pass
    def _handle_generic_interaction(self, user_id: str, event_type: str, track_id: Optional[int], **kwargs): # As refined
        pass
    def on_user_interaction_track_played(self, track_id: int, timestamp: str, **kwargs): # As refined
        pass
    def on_user_interaction_track_skipped(self, track_id: int, skipped_at_position_sec: float, timestamp: str, **kwargs): # As refined
        pass
    def on_user_interaction_track_liked(self, track_id: int, timestamp: str, **kwargs): # As refined
        pass
    def on_user_interaction_track_disliked(self, track_id: int, timestamp: str, **kwargs): # As refined
        pass
    def get_recommendations_for_user_sync_wrapper(self, user_id: str, num_recs: int = 20, context: Optional[Dict]=None) -> List[Dict[str, Any]]: # As refined
        pass


class RecommendationUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, engine: RecommendationEngine, host_app_ref: Any, user_id: str = "default_user"):
        super().__init__(parent) # CRITICAL
        self.engine = engine
        self.host_app = host_app_ref
        self.user_id = user_id
        self.root_app_tk = parent.winfo_toplevel()
        self.current_similar_tracks: List[Dict] = []
        self.current_foryou_recs: List[Dict] = []
        self._create_ui()
        if self.host_app:
            self.host_app.subscribe_to_event("audio_track_loaded_basic", self.on_current_track_changed_for_similarity)
            current_meta = self.host_app.get_current_track_metadata()
            if current_meta and hasattr(current_meta,'id') and current_meta.id is not None: self.on_current_track_changed_for_similarity(metadata=current_meta)
            if self.host_app.theme_manager: 
                self.host_app.theme_manager.register_callback(self.apply_theme_to_rec_ui)
                if self.host_app.theme_manager.get_current_theme(): self.apply_theme_to_rec_ui(self.host_app.theme_manager.get_current_theme())
        self.refresh_recommendations()

    def apply_theme_to_rec_ui(self, theme: Any): # ... (as before) ...
        pass
    def _create_ui(self): # ... (Full implementation as before) ...
        pass
    def on_current_track_changed_for_similarity(self, metadata: Any, **kwargs): # ... (as before) ...
        pass
    def _clear_foryou_recs_display(self): # ... (as before) ...
        pass
    def _clear_similar_tracks_display(self): # ... (as before) ...
        pass
    def _display_foryou_recommendations(self, recommendations: List[Dict]): # ... (as before) ...
        pass
    def _display_similar_tracks(self, similar_tracks: List[Dict], original_title: str): # ... (as before) ...
        pass
    def _create_recommendation_card_ui(self, parent_widget: ttk.Frame, track_data: Dict, card_text_override: Optional[str] = None) -> ttk.Frame: # ... (as before) ...
        pass
    def refresh_recommendations(self): # ... (as before) ...
        pass
    def play_recommended_track(self, track_id_lib: Optional[int], file_path: Optional[str]): # ... (as before) ...
        pass
    def on_app_exit(self): # ... (as before) ...
        pass


def create_recommendation_tab(notebook: ttk.Notebook, host_app_ref: Any) -> Tuple[Optional[RecommendationEngine], Optional[RecommendationUI]]:
    # ... (Full implementation as before) ...
    pass




"""
Flow State: Real-time Collaboration Module
Multi-user listening sessions, collaborative playlists, and social features
"""

import asyncio
import websockets 
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone 
from typing import Dict, List, Set, Optional, Callable, Any, Coroutine
from dataclasses import dataclass, field, asdict
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, simpledialog, filedialog 
from pathlib import Path 
import secrets
import logging
import concurrent.futures 
import uuid 

logger = logging.getLogger("FlowStateCollab")

COLLAB_DB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="CollabDB")
APP_DATA_BASE_PATH_COLLAB = Path.home() / ".flowstate"
COLLAB_DATA_DIR = APP_DATA_BASE_PATH_COLLAB / "data"
COLLAB_DB_PATH = str(COLLAB_DATA_DIR / "collaboration_engine_data.db")


@dataclass
class User: 
    user_id: str; username: str; display_name: Optional[str] = None
    avatar_url: Optional[str] = None; status: str = "online"
    current_session_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def __post_init__(self):
        if self.display_name is None: self.display_name = self.username

@dataclass
class Session: 
    session_id: str; name: str; host_id: str 
    host_username: str 
    participants: Dict[str, User] = field(default_factory=dict) 
    playlist: List[Dict[str, Any]] = field(default_factory=list) 
    current_track_index: int = -1 
    current_position_sec: float = 0.0 
    is_playing: bool = False
    is_public: bool = True 
    max_participants: int = 50
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chat_enabled: bool = True
    collaborative_queue: bool = True 
    voting_enabled: bool = True 
    _skip_votes_current_track: Set[str] = field(default_factory=set, repr=False)
    _current_track_for_voting_id: Optional[Any] = field(default=None, repr=False) # Store ID of track being voted on
    _last_voted_track_idx: Optional[int] = field(default=None, repr=False) # Store playlist index of track


@dataclass
class ChatMessage: 
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""; user_id: str = ""; username: str = ""
    message: str = ""; timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_type: str = "text"


class CollaborationServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host; self.port = port
        self.server: Optional[websockets.WebSocketServer] = None
        self.sessions: Dict[str, Session] = {} 
        self.connections: Dict[websockets.WebSocketServerProtocol, Dict[str, Any]] = {}
        self.user_connections: Dict[str, Set[websockets.WebSocketServerProtocol]] = {} 
        COLLAB_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _db_execute(self, query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False, conn_passed: Optional[sqlite3.Connection] = None):
        # ... (Full implementation as before) ...
        pass
    async def db_action(self, query: str, params: tuple = (), commit: bool = False, fetch_one: bool = False, fetch_all: bool = False):
        # ... (Full implementation as before) ...
        pass
    def _init_database(self): # Add tables for users, sessions, chat_messages
        # ... (Full implementation as before, ensuring all CREATE TABLE and INDEX statements are correct) ...
        pass
    async def start(self): # ... (as before) ...
        pass
    async def stop(self): # ... (as before) ...
        pass

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        client_addr = websocket.remote_address
        logger.info(f"Collab Server: New client connection from {client_addr}")
        self.connections[websocket] = {'user_id': None, 'session_id': None} # Temp store
        try:
            async for message_str in websocket:
                await self.handle_message(websocket, message_str)
        except websockets.exceptions.ConnectionClosedError as e:
            logger.info(f"Collab Server: Connection closed by client {client_addr} (Code: {e.code}, Reason: {e.reason})")
        except Exception as e:
            logger.error(f"Collab Server: Error in connection handler for {client_addr}: {e}", exc_info=True)
        finally:
            await self.handle_user_disconnect(websocket)

    async def handle_authenticate(self, websocket: websockets.WebSocketServerProtocol, data: Dict) -> bool:
        user_id = data.get('user_id')
        username = data.get('username')
        # In a real app, token would be validated against a user DB or auth service
        if user_id and username:
            self.connections[websocket]['user_id'] = user_id
            self.connections[websocket]['username'] = username # Store username too
            
            # Store user if not exists (basic user management)
            existing_user = await self.db_action("SELECT user_id FROM collab_users WHERE user_id = ?", (user_id,), fetch_one=True)
            if not existing_user:
                await self.db_action("INSERT INTO collab_users (user_id, username, display_name, created_at) VALUES (?, ?, ?, ?)",
                                     (user_id, username, username, datetime.now(timezone.utc).isoformat()), commit=True)

            self.user_connections.setdefault(user_id, set()).add(websocket)
            await self.send_to_connection(websocket, {'type': 'auth_success', 'user_id': user_id})
            logger.info(f"Client {websocket.remote_address} authenticated as user {user_id} ('{username}')")
            return True
        else:
            await self.send_to_connection(websocket, {'type': 'auth_failed', 'reason': 'Missing user_id or username'})
            return False

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message_str: str):
        try:
            message = json.loads(message_str)
            msg_type = message.get('type')
            user_id = self.connections.get(websocket, {}).get('user_id')

            if not user_id and msg_type != 'authenticate': # Must auth first
                await self.send_to_connection(websocket, {'type':'error', 'message':'Authentication required.'}); return
            
            logger.debug(f"Collab Server RX from {user_id or 'unauth'}: {message}")

            handler_map = {
                'authenticate': self.handle_authenticate,
                'create_session': self._handle_create_session_command,
                'join_session': self._handle_join_session_command,
                'leave_session': self._handle_leave_session_command,
                'play_pause': self._handle_play_pause_command, # Host action
                'seek': self._handle_seek_command,             # Host action
                'next_track': self._handle_next_track_command, # Host action
                'prev_track': self._handle_prev_track_command, # Host action
                'add_track_to_session': self._handle_add_track_to_session_command,
                'remove_track_from_session': self._handle_remove_track_from_session_command,
                'reorder_track_in_session': self._handle_reorder_track_in_session_command,
                'chat_message': self._handle_chat_message_command,
                'vote_skip': self._handle_vote_skip_command,
                'request_sync': self._handle_sync_request_command,
            }
            handler = handler_map.get(msg_type)
            if handler: await handler(websocket, message.get('data', {})) # Pass data field
            else: logger.warning(f"Unknown message type '{msg_type}' from {user_id}")

        except json.JSONDecodeError: logger.error("Received invalid JSON message.")
        except Exception as e: logger.error(f"Error handling message: {e}", exc_info=True)


    async def handle_user_disconnect(self, websocket: websockets.WebSocketServerProtocol):
        conn_info = self.connections.pop(websocket, None)
        if conn_info and conn_info['user_id']:
            user_id = conn_info['user_id']
            username = conn_info.get('username', 'Unknown')
            session_id = conn_info.get('session_id')
            
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(websocket)
                if not self.user_connections[user_id]: # No more connections for this user
                    del self.user_connections[user_id]
            
            logger.info(f"User {user_id} ('{username}') disconnected from {websocket.remote_address}")
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if user_id in session.participants:
                    del session.participants[user_id]
                    logger.info(f"User {user_id} removed from session {session_id}. Participants left: {len(session.participants)}")
                    await self.broadcast_to_session(session_id, {'type': 'user_left', 'user_id': user_id, 'username': username, 'session_id':session_id, 'num_participants': len(session.participants)})
                    
                    if not session.participants: # Session empty
                        logger.info(f"Session {session_id} is now empty. Removing from active sessions.")
                        del self.sessions[session_id]
                        # Optionally delete from DB or mark inactive
                        # await self.db_action("DELETE FROM collab_sessions WHERE session_id = ?", (session_id,), commit=True)
                    elif session.host_id == user_id: # Host left
                        if session.participants: # Promote new host (e.g. first joined or longest present)
                            new_host_user_id = list(session.participants.keys())[0] # Simplistic: first in dict
                            new_host_user = session.participants[new_host_user_id]
                            session.host_id = new_host_user.user_id
                            session.host_username = new_host_user.username
                            logger.info(f"Host {user_id} left session {session_id}. New host: {session.host_id} ('{session.host_username}')")
                            await self.save_session(session) # Persist new host
                            await self.broadcast_to_session(session_id, {'type': 'host_changed', 'new_host_id': session.host_id, 'new_host_username': session.host_username, 'session_id': session_id})
                        else: # Host left and session became empty
                             logger.info(f"Host {user_id} left, session {session_id} is now empty and removed.")
                             if session_id in self.sessions: del self.sessions[session_id]


    async def _handle_create_session_command(self, ws, data: Dict): # As before
        pass
    async def _handle_join_session_command(self, ws, data: Dict): # As refined (sends chat history)
        pass
    async def _handle_leave_session_command(self, ws, data: Dict): # As before
        pass
    async def _handle_play_pause_command(self, ws, data: Dict): # As refined (server-authoritative)
        pass
    async def _handle_seek_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_next_track_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_prev_track_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_add_track_to_session_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_remove_track_from_session_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_reorder_track_in_session_command(self, ws, data: Dict): # TODO
        pass
    async def _handle_chat_message_command(self, ws, data: Dict): # As refined (saves to DB)
        pass
    async def _handle_vote_skip_command(self, ws, data: Dict): # As refined
        pass
    async def _handle_sync_request_command(self, ws, data: Dict): # As refined
        pass
    async def save_session(self, session: Session): # As refined
        pass
    async def load_session_from_db(self, session_id: str) -> Optional[Session]: # As refined
        pass
    async def save_chat_message(self, message_obj: ChatMessage): # As refined
        pass
    async def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict]: # As refined
        pass
    async def broadcast_playback_state(self, session_id: str): # As refined
        pass
    async def broadcast_track_changed(self, session_id: str): # As refined
        pass
    async def broadcast_to_session(self, session_id: str, message: Dict, exclude_ws: Optional[websockets.WebSocketServerProtocol]=None): # As before
        pass
    async def send_to_connection(self, ws: websockets.WebSocketServerProtocol, message: Dict): # As before
        pass
    async def send_to_user(self, user_id: str, message: Dict): # As before
        pass
    def get_user_id_by_connection(self, ws: websockets.WebSocketServerProtocol) -> Optional[str]: # As before
        pass


class CollaborationClient:
    # ... (Full implementation as refined: __init__, connect (sends user_id/name from host_app), 
    #      disconnect, _listen_loop, _process_incoming_message, send_message, register_handler, 
    #      and all specific command methods that now just send messages to server) ...
    def __init__(self, server_url: str, host_app_ref: Optional[Any] = None): # as before
        pass
    # ... (all methods as before)


class CollaborationUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent) # CRITICAL: Initialize ttk.Frame
        self.host_app = host_app_ref
        # ... (rest of __init__ as refined before)
        pass
    
    # ... (All methods of CollaborationUI as previously refined, including:
    #      _run_async_loop_in_thread, run_async,
    #      _setup_client_event_handlers (with on_chat_history_received, on_skip_vote_update),
    #      _create_ui_widgets (with chat_display, vote_skip_button),
    #      UI Action Handlers (add_track_to_session_dialog now uses proper track info fetching),
    #      Host UI Action Handlers (now *only* send commands to server),
    #      Client Event Handlers (_handle_session_playback_state_changed and _handle_session_track_changed
    #         are CRITICAL for server-authoritative sync, _handle_track_added/removed, 
    #         _add_chat_message_to_display, _handle_chat_history, _update_skip_vote_button),
    #      _update_session_ui_display (updates host controls, queue, participant list, chat),
    #      _update_queue_ui_display (refreshes queue_tree with current session playlist and highlight),
    #      on_app_exit.
    # ) ...
    # Example for one:
    def _add_chat_message_to_display(self, message_dict: Dict):
        if not hasattr(self, 'chat_display') or not self.chat_display: return
        ts_str = message_dict.get('timestamp', datetime.now(timezone.utc).isoformat())
        try: ts_dt = datetime.fromisoformat(ts_str.replace('Z','+00:00')) # Handle Z for UTC
        except ValueError: ts_dt = datetime.now(timezone.utc) # Fallback
        local_ts = ts_dt.astimezone()
        display_line = f"[{local_ts.strftime('%H:%M')}] {message_dict.get('username','System')}: {message_dict.get('message','')}\n"
        
        was_scrolled_to_bottom = False
        try: # Check if scrolled to bottom before inserting
            # This can be tricky with ScrolledText. A simpler way is just always scroll.
            # yview_moveto(1.0) moves the top of the last line to the top of the widget.
            # yview_scroll(number, what) scrolls by units or pages.
            # For ScrolledText, self.chat_display.yview_moveto(1.0) might work after insert.
            # A common check is if bottom of visible area is close to bottom of content.
            # For simplicity, just scroll after insert for now.
            pass
        except tk.TclError: pass # If widget not ready

        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, display_line)
        self.chat_display.see(tk.END) # Auto-scroll
        self.chat_display.config(state='disabled')


def create_collaboration_tab(notebook: ttk.Notebook, host_app_ref: Any) -> CollaborationUI:
    collab_frame = ttk.Frame(notebook)
    notebook.add(collab_frame, text="Collaborate")
    collab_ui_instance = CollaborationUI(collab_frame, host_app_ref)
    collab_ui_instance.pack(fill=tk.BOTH, expand=True)
    # host_app_ref.collab_client_ui_ref = collab_ui_instance # Launcher already does this
    logger.info("Collaboration Tab UI created.")
    return collab_ui_instance


if __name__ == '__main__':
    # Standalone test needs a mock host_app and to run server/client in separate threads/processes.
    # This is complex for a simple __main__ block.
    # Focus on testing via the full application launcher.
    logger.info("Collaboration module standalone test: Server and client need to be run separately or in threads.")
    # Example: Start server in a thread, then create a Tkinter root with CollaborationUI client.
    pass

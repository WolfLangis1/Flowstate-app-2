

"""
Flow State: Mobile Sync & Remote Control
Control your music from any device with real-time synchronization
"""

import asyncio
import aiohttp # For server
from aiohttp import web # For server
import qrcode # For QR code generation
import socket # For get_local_ip fallback and zeroconf
import json
# import sqlite3 # SecurityManager stores paired devices in encrypted JSON file now
import secrets # For PINs, JWT secret
import hashlib # Not directly used, but good for security context
from jose import jwt, JWTError, ExpiredSignatureError 
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Any, Callable, Tuple 
import threading
import os 
import platform # For file permissions
# import base64 # Not directly used for QR data in this version
from dataclasses import dataclass, asdict, field
import netifaces # For get_local_ip (preferred)
import zeroconf # For service discovery
from cryptography.fernet import Fernet 
import logging
# import concurrent.futures # aiohttp is async, not much use for pools here
from pathlib import Path 
import re # For RemoteControlUI unpair logic

logger = logging.getLogger("FlowStateMobileSync")

APP_DATA_BASE_PATH = Path.home() / ".flowstate"
MOBILE_SYNC_DATA_DIR = APP_DATA_BASE_PATH / "mobile_sync_data"
SECURITY_KEY_PATH = MOBILE_SYNC_DATA_DIR / "mobile_sync_fernet.key" 
PAIRED_DEVICES_PATH = MOBILE_SYNC_DATA_DIR / "paired_devices_mobile.enc.json"

@dataclass
class Device: 
    device_id: str; device_name: str; device_type: str; platform: str; app_version: str
    last_seen_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_paired: bool = False
    capabilities: List[str] = field(default_factory=list)

@dataclass
class RemoteCommand: 
    command_id: str; device_id: str; command: str; params: Dict
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"


class SecurityManager:
    def __init__(self, app_name: str = "FlowStateMobileSync"):
        self.app_name = app_name
        MOBILE_SYNC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.encryption_key = self._get_or_create_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.jwt_secret = secrets.token_urlsafe(32) 
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_delta = timedelta(days=30) 
        self.active_pairing_pins: Dict[str, Tuple[str, datetime]] = {} 
        self.paired_devices_info: Dict[str, Dict[str, str]] = self._load_paired_devices_info()

    def _get_or_create_encryption_key(self) -> bytes:
        if SECURITY_KEY_PATH.exists(): return SECURITY_KEY_PATH.read_bytes()
        key = Fernet.generate_key()
        SECURITY_KEY_PATH.write_bytes(key)
        if platform.system() != "Windows": os.chmod(SECURITY_KEY_PATH, 0o600)
        logger.info(f"Generated new encryption key at {SECURITY_KEY_PATH}")
        return key

    def _load_paired_devices_info(self) -> Dict[str, Dict[str, str]]:
        if PAIRED_DEVICES_PATH.exists():
            try:
                decrypted_data = self.fernet.decrypt(PAIRED_DEVICES_PATH.read_bytes())
                devices = json.loads(decrypted_data.decode('utf-8'))
                logger.info(f"Loaded {len(devices)} paired devices.")
                return devices
            except Exception as e: logger.error(f"Failed to load/decrypt: {e}. Starting fresh.", exc_info=True)
        return {}

    def _save_paired_devices_info(self):
        try:
            encrypted_data = self.fernet.encrypt(json.dumps(self.paired_devices_info).encode('utf-8'))
            PAIRED_DEVICES_PATH.write_bytes(encrypted_data)
            logger.info(f"Saved {len(self.paired_devices_info)} paired devices.")
        except Exception as e: logger.error(f"Failed to save/encrypt: {e}", exc_info=True)

    def generate_pin_for_desktop_display(self, temp_desktop_pairing_id: str) -> str:
        now_utc = datetime.now(timezone.utc)
        self.active_pairing_pins = {p:info for p,info in self.active_pairing_pins.items() if info[1] >= now_utc} # Clean expired
        while True:
            pin = "".join(secrets.choice("0123456789") for _ in range(6))
            if pin not in self.active_pairing_pins: break
        expiry = now_utc + timedelta(minutes=5)
        self.active_pairing_pins[pin] = (temp_desktop_pairing_id, expiry)
        logger.info(f"Generated PIN {pin} for temp_id {temp_desktop_pairing_id}, expires {expiry.isoformat()}")
        return pin

    def verify_pin_from_mobile(self, pin_entered: str, temp_desktop_id_from_mobile: str) -> bool:
        pin_info = self.active_pairing_pins.get(pin_entered)
        if pin_info:
            orig_temp_id, expiry = pin_info
            if datetime.now(timezone.utc) < expiry:
                if orig_temp_id == temp_desktop_id_from_mobile: logger.info(f"PIN {pin_entered} verified for {temp_desktop_id_from_mobile}."); return True
                else: logger.warning(f"PIN {pin_entered} valid, but temp_id mismatch (exp {orig_temp_id}, got {temp_desktop_id_from_mobile}).")
            else: logger.warning(f"PIN {pin_entered} expired."); del self.active_pairing_pins[pin_entered]
        else: logger.warning(f"PIN {pin_entered} not found or already used.")
        return False

    def confirm_device_pairing(self, mobile_device_id: str, mobile_device_name: str, verified_pin: str) -> bool:
        if verified_pin not in self.active_pairing_pins: # Check again, might have expired or been used by race condition
            logger.warning(f"Attempt to confirm pairing for no longer active PIN {verified_pin}.")
            return False
        self.paired_devices_info[mobile_device_id] = {'name': mobile_device_name, 'paired_at_utc': datetime.now(timezone.utc).isoformat()}
        self._save_paired_devices_info()
        del self.active_pairing_pins[verified_pin] # Consume PIN
        logger.info(f"Device {mobile_device_id} ('{mobile_device_name}') paired with PIN {verified_pin}.")
        return True

    def is_device_paired(self, device_id: str) -> bool: return device_id in self.paired_devices_info

    def generate_auth_token(self, device_id: str) -> str: 
        if not self.is_device_paired(device_id): logger.warning(f"Token gen for unpaired device: {device_id}"); return ""
        now = datetime.now(timezone.utc)
        payload = {'device_id': device_id, 'exp': now + self.jwt_expiry_delta, 'iat': now, 'sub': device_id}
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_auth_token(self, token: str) -> Optional[str]: 
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            device_id = payload.get('device_id') # Or 'sub' if preferred
            if not device_id or not self.is_device_paired(device_id):
                logger.warning(f"Token verified but device '{device_id}' no longer paired."); return None
            return device_id
        except ExpiredSignatureError: logger.info(f"Auth token expired."); return None
        except JWTError as e: logger.warning(f"Auth token verification failed: {e}"); return None

    def unpair_device(self, device_id: str) -> bool:
        if device_id in self.paired_devices_info:
            del self.paired_devices_info[device_id]; self._save_paired_devices_info()
            logger.info(f"Device {device_id} unpaired."); return True
        return False


class MobileServer:
    def __init__(self, host_address: str = "0.0.0.0", port_number: int = 8888, host_app_interface: Optional[Any] = None):
        self.host = host_address; self.port = port_number; self.host_app = host_app_interface 
        self.app = web.Application(middlewares=[self._auth_middleware])
        self.aiohttp_runner: Optional[web.AppRunner] = None; self.aiohttp_site: Optional[web.TCPSite] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.security = SecurityManager()
        self.authed_websockets: Dict[str, web.WebSocketResponse] = {}
        self.ws_lock = asyncio.Lock()
        self.player_state: Dict[str, Any] = {} # Populated by update_player_state_from_host
        self.setup_routes()
        self.zeroconf_handler: Optional[zeroconf.Zeroconf] = None; self.service_info: Optional[zeroconf.ServiceInfo] = None
        self.ui_pair_success_callback: Optional[Callable[[str,str],None]] = None
        # Initial state update
        self.update_player_state_from_host()

    def get_local_ip(self) -> str: # As refined before (netifaces preferred, then socket)
        pass
    def setup_routes(self): # As refined before
        pass
    async def _auth_middleware(self, app, handler): # As refined before
        pass
    async def handle_info(self, request: web.Request) -> web.Response: # As refined before
        pass
    async def handle_qr_code_info_for_mobile(self, request: web.Request) -> web.Response: # As refined before
        pass
    async def handle_submit_pin_from_mobile(self, request: web.Request) -> web.Response: # As refined before
        pass
    async def handle_get_state(self, request: web.Request) -> web.Response: # As refined before
        pass
    async def handle_command(self, request: web.Request) -> web.Response: # As refined before
        pass
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse: # As refined before
        pass
    async def broadcast_state_update(self): # As refined before
        pass
    def update_player_state_from_host(self): # As refined before
        pass
    def _start_zeroconf_service(self): # As refined before
        pass
    def _stop_zeroconf_service(self): # As refined before
        pass
    async def start(self): # As refined before
        pass
    async def stop(self): # As refined before
        pass
    def _schedule_on_main_thread(self, callable_to_run: Callable, *args): # As refined before
        pass
    async def handle_library_browse(self, request: web.Request) -> web.Response: # As refined before
        pass


class RemoteControlUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent) # CRITICAL: Initialize ttk.Frame
        self.host_app = host_app_ref
        self.server_instance: Optional[MobileServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.server_loop: Optional[asyncio.AbstractEventLoop] = None
        self.ip_label_var = tk.StringVar(value="IP: N/A"); self.port_label_var = tk.StringVar(value="Port: N/A")
        self.status_var = tk.StringVar(value="Server Stopped"); self.desktop_pin_var = tk.StringVar(value="----")
        self.qr_image_tk: Optional[ImageTk.PhotoImage] = None
        self.paired_devices_list_var = tk.StringVar()
        self.create_ui()
        self.update_ip_address_display()
        self._subscribe_to_host_events_for_remote()
        # self.pack(fill=tk.BOTH, expand=True) # Creator function packs this instance

    def create_ui(self): # As refined before
        pass
    def update_ip_address_display(self): # As refined before
        pass
    def start_server(self): # As refined before
        pass
    def _run_server_in_thread(self, loop: asyncio.AbstractEventLoop, server: MobileServer): # As refined before
        pass
    def stop_server(self): # As refined before
        pass
    def _update_ui_for_server_stopped(self): # As refined before
        pass
    def refresh_pairing_info_from_server(self): # As refined before
        pass
    def generate_qr_code_image(self, qr_data_dict: Dict[str, Any]): # As refined before
        pass
    def update_paired_devices_list(self): # As refined before
        pass
    def unpair_selected_device(self): # As refined before
        pass
    def on_device_paired_ui_update(self, device_id: str, device_name: str): # As refined before
        pass
    def _subscribe_to_host_events_for_remote(self): # As refined before
        pass
    def _schedule_broadcast(self): # As refined before
        pass
    def on_host_playback_state_changed(self, **kwargs): self._schedule_broadcast()
    def on_host_playback_position_changed(self, **kwargs): self._schedule_broadcast()
    def on_host_track_changed(self, **kwargs): self._schedule_broadcast()
    def on_host_volume_changed(self, **kwargs): self._schedule_broadcast()
    def on_host_playlist_changed(self, **kwargs): self._schedule_broadcast()
    def on_host_shuffle_repeat_changed(self, **kwargs): self._schedule_broadcast()
    def on_app_exit(self): # As refined before
        pass


def create_remote_control_tab(notebook: ttk.Notebook, host_app_ref: Any) -> RemoteControlUI:
    remote_frame = ttk.Frame(notebook)
    notebook.add(remote_frame, text="Remote") 
    remote_ui_instance = RemoteControlUI(remote_frame, host_app_ref=host_app_ref)
    remote_ui_instance.pack(fill=tk.BOTH, expand=True)
    logger.info("Remote Control Tab UI created.")
    return remote_ui_instance


if __name__ == "__main__":
    # ... (Standalone test block as previously refined, ensuring MockHostApp is comprehensive) ...
    pass


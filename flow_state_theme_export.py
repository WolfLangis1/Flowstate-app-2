

"""
Flow State: Theme System & Export Module
Customizable themes and export/streaming capabilities
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np 
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageColor 
import threading 
import queue 
import subprocess 
import datetime 
import tempfile 
import shutil 
import logging
import concurrent.futures 
import re 
import platform 
import http.server # For conceptual StreamingServer
import socketserver # For conceptual StreamingServer

# For ExportManager's dynamic viz export
try:
    from flow_state_advanced_viz import VisualizationConfig, VisualizationEngine # For type hints and instantiation
except ImportError:
    VisualizationConfig = None # type: ignore
    VisualizationEngine = None # type: ignore
    # Define a dummy logger if this module is run standalone and advanced_viz isn't found
    if not logging.getLogger("FlowStateThemeExport").hasHandlers():
        logging.basicConfig(level=logging.DEBUG)
    logger_for_fallback = logging.getLogger("FlowStateThemeExport")
    logger_for_fallback.warning("Advanced Visualization module not found. Dynamic video export will be limited/unavailable.")

logger = logging.getLogger("FlowStateThemeExport")

if 'THEME_EXPORT_PROCESS_POOL' not in globals():
    THEME_EXPORT_PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2 -1)) # No thread_name_prefix
if 'THEME_EXPORT_THREAD_POOL' not in globals():
    THEME_EXPORT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="ThemeExportIO")


@dataclass
class Theme: 
    name: str = "Unnamed Theme"
    primary_bg: str = "#1e1e1e"; secondary_bg: str = "#2e2e2e"; accent_bg: str = "#3e3e3e"
    primary_fg: str = "#e0e0e0"; secondary_fg: str = "#b0b0b0"
    accent_color: str = "#00aaff"; highlight_color: str = "#0077cc"
    error_color: str = "#ff4444"; success_color: str = "#44dd44"; warning_color: str = "#ffaa00"
    viz_bg: str = "#000000"; viz_primary: str = "#00ffff"; viz_secondary: str = "#ff00ff"; viz_tertiary: str = "#ffff00"
    waveform_color: str = "#33ff33"
    spectrum_colors: List[str] = field(default_factory=lambda: ["#ff0000","#ff8800","#ffff00","#00ff00","#00ffff","#0088ff","#ff00ff"])
    font_family: str = "Segoe UI" if platform.system() == "Windows" else "Arial"
    font_size_small: int = 9; font_size_normal: int = 10; font_size_large: int = 12; font_size_title: int = 14
    def to_dict(self) -> Dict: return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict) -> 'Theme':
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        valid_data = {k: data[k] for k in data if k in field_names}
        return cls(**valid_data)


class ThemeManager:
    def __init__(self, app_root_tk_widget: tk.Tk):
        self.root = app_root_tk_widget
        self.style = ttk.Style(self.root) # Global ttk style object
        self.current_theme_obj: Optional[Theme] = None # Renamed from current_theme
        self.themes: Dict[str, Theme] = {}
        self.base_app_dir = Path.home() / ".flowstate"
        self.theme_dir = self.base_app_dir / "themes"
        self.theme_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks: List[Callable[[Theme], None]] = []
        self.load_builtin_themes()
        self.load_custom_themes()
        
        default_theme_name = "Dark Cyan" 
        if default_theme_name in self.themes: self.apply_theme(default_theme_name)
        elif self.themes: self.apply_theme(list(self.themes.keys())[0])
        else: 
            logger.error("No themes loaded. UI will use Tkinter defaults.")
            self.current_theme_obj = Theme(name="Fallback Default (Unthemed)") # Basic fallback

    def load_builtin_themes(self):
        # Define a few built-in themes
        builtin = {
            "Dark Cyan": Theme(name="Dark Cyan", primary_bg="#1A242F", secondary_bg="#243447", accent_bg="#2E4053", primary_fg="#EAECEE", secondary_fg="#BDC3C7", accent_color="#00BCD4", highlight_color="#0097A7", viz_primary="#00BCD4", waveform_color="#4CAF50"),
            "Light Solar": Theme(name="Light Solar", primary_bg="#FDF6E3", secondary_bg="#F5E8C0", accent_bg="#EEE8D5", primary_fg="#657B83", secondary_fg="#586E75", accent_color="#268BD2", highlight_color="#2AA198", viz_bg="#FDF6E3", viz_primary="#D33682", waveform_color="#859900"),
            "Dracula": Theme(name="Dracula", primary_bg="#282A36", secondary_bg="#383C4A", accent_bg="#44475A", primary_fg="#F8F8F2", secondary_fg="#BFBFBF", accent_color="#BD93F9", highlight_color="#6272A4", viz_primary="#FF79C6", waveform_color="#50FA7B"),
        }
        self.themes.update(builtin)

    def load_custom_themes(self):
        if not self.theme_dir.exists(): return
        for filepath in self.theme_dir.glob("*.theme.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    theme_data = json.load(f)
                    if 'name' in theme_data:
                        theme_name = theme_data['name']
                        self.themes[theme_name] = Theme.from_dict(theme_data)
                        logger.info(f"Loaded custom theme: {theme_name} from {filepath.name}")
                    else: logger.warning(f"Skipping theme file {filepath.name}: missing 'name' field.")
            except json.JSONDecodeError: logger.error(f"Error decoding theme file {filepath.name}")
            except Exception as e: logger.error(f"Error loading theme {filepath.name}: {e}")

    def save_theme(self, theme: Theme, is_custom:bool = True):
        if not theme.name: logger.error("Cannot save theme: theme name is empty."); return False
        self.themes[theme.name] = theme # Update in-memory cache
        if is_custom: # Only save custom themes to file system
            # Sanitize theme name for filename
            filename_base = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in theme.name).rstrip()
            filename = self.theme_dir / f"{filename_base}.theme.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(theme.to_dict(), f, indent=2)
                logger.info(f"Saved custom theme '{theme.name}' to {filename}")
                return True
            except IOError as e:
                logger.error(f"Error saving theme '{theme.name}' to file {filename}: {e}"); return False
        return True # Built-in themes updated in memory only

    def apply_theme(self, theme_name_or_theme_obj: Union[str, Theme]):
        theme_to_apply: Optional[Theme] = None
        if isinstance(theme_name_or_theme_obj, Theme):
            theme_to_apply = theme_name_or_theme_obj
        elif isinstance(theme_name_or_theme_obj, str):
            theme_to_apply = self.themes.get(theme_name_or_theme_obj)
        
        if not theme_to_apply:
            logger.warning(f"Theme '{str(theme_name_or_theme_obj)}' not found. Cannot apply."); return

        self.current_theme_obj = theme_to_apply
        logger.info(f"Applying theme: {self.current_theme_obj.name}")

        # --- Apply to TTK Style ---
        # Base style for all ttk widgets
        self.style.theme_use('clam') # Start with a base theme that's somewhat stylable
        
        # General widget styling
        self.style.configure(".", background=theme_to_apply.secondary_bg, foreground=theme_to_apply.primary_fg,
                             font=(theme_to_apply.font_family, theme_to_apply.font_size_normal),
                             fieldbackground=theme_to_apply.primary_bg, # For Entry, Combobox backgrounds
                             borderwidth=1, relief=tk.FLAT) 
        self.style.map(".", background=[('active', theme_to_apply.highlight_color), ('disabled', theme_to_apply.accent_bg)])

        # Specific widget styling
        self.style.configure("TButton", padding=5, background=theme_to_apply.accent_bg, foreground=theme_to_apply.primary_fg)
        self.style.map("TButton", background=[('active', theme_to_apply.highlight_color), ('pressed', theme_to_apply.accent_color)])
        self.style.configure("TLabel", padding=2, background=theme_to_apply.secondary_bg, foreground=theme_to_apply.primary_fg)
        self.style.configure("Header.TLabel", font=(theme_to_apply.font_family, theme_to_apply.font_size_title, 'bold'), foreground=theme_to_apply.accent_color)
        self.style.configure("Accent.TLabel", foreground=theme_to_apply.accent_color)
        
        self.style.configure("TFrame", background=theme_to_apply.secondary_bg)
        self.style.configure("TLabelframe", background=theme_to_apply.secondary_bg, bordercolor=theme_to_apply.accent_color, lightcolor=theme_to_apply.secondary_bg, darkcolor=theme_to_apply.secondary_bg)
        self.style.configure("TLabelframe.Label", background=theme_to_apply.secondary_bg, foreground=theme_to_apply.accent_color, font=(theme_to_apply.font_family, theme_to_apply.font_size_normal, 'bold'))

        self.style.configure("TEntry", insertcolor=theme_to_apply.primary_fg, fieldbackground=theme_to_apply.primary_bg)
        self.style.configure("TCombobox", arrowcolor=theme_to_apply.accent_color, fieldbackground=theme_to_apply.primary_bg, selectbackground=theme_to_apply.highlight_color)
        self.style.map("TCombobox", fieldbackground=[('readonly', theme_to_apply.primary_bg)])

        self.style.configure("TScrollbar", background=theme_to_apply.accent_bg, troughcolor=theme_to_apply.primary_bg, arrowcolor=theme_to_apply.primary_fg)
        self.style.map("TScrollbar", background=[('active', theme_to_apply.highlight_color)])

        self.style.configure("TNotebook", background=theme_to_apply.primary_bg, tabmargins=[2,5,2,0])
        self.style.configure("TNotebook.Tab", background=theme_to_apply.accent_bg, foreground=theme_to_apply.secondary_fg, padding=[10, 3], font=(theme_to_apply.font_family, theme_to_apply.font_size_normal))
        self.style.map("TNotebook.Tab", background=[("selected", theme_to_apply.highlight_color)], foreground=[("selected", theme_to_apply.primary_fg)])

        self.style.configure("Treeview", background=theme_to_apply.primary_bg, foreground=theme_to_apply.primary_fg, fieldbackground=theme_to_apply.primary_bg, rowheight=25)
        self.style.map("Treeview", background=[('selected', theme_to_apply.highlight_color)], foreground=[('selected', theme_to_apply.primary_fg)])
        self.style.configure("Treeview.Heading", background=theme_to_apply.accent_bg, foreground=theme_to_apply.primary_fg, font=(theme_to_apply.font_family, theme_to_apply.font_size_normal, 'bold'), relief=tk.FLAT, padding=3)
        self.style.map("Treeview.Heading", background=[('active',theme_to_apply.highlight_color)])
        
        self.style.configure("TProgressbar", troughcolor=theme_to_apply.primary_bg, background=theme_to_apply.accent_color, thickness=10)
        self.style.configure("Horizontal.TScale", troughcolor=theme_to_apply.primary_bg, background=theme_to_apply.accent_color) # Slider color
        self.style.configure("Vertical.TScale", troughcolor=theme_to_apply.primary_bg, background=theme_to_apply.accent_color)

        # Apply to root window itself for overall background
        if self.root: self.root.configure(background=theme_to_apply.primary_bg)
        
        # Notify subscribers
        for callback in list(self.callbacks): # Iterate copy
            try: callback(self.current_theme_obj)
            except Exception as e_cb: logger.error(f"Error in theme change callback {callback}: {e_cb}", exc_info=True)
        
        if self.root: self.root.update_idletasks() # Force Tkinter to redraw with new styles

    def register_callback(self, callback: Callable[[Theme], None]):
        if callback not in self.callbacks: self.callbacks.append(callback)
    def unregister_callback(self, callback: Callable[[Theme], None]):
        if callback in self.callbacks: self.callbacks.remove(callback)
    def get_theme_names(self) -> List[str]: return sorted(list(self.themes.keys()))
    def get_current_theme(self) -> Optional[Theme]: return self.current_theme_obj
    def get_theme_by_name(self, name: str) -> Optional[Theme]: return self.themes.get(name)


class ThemeEditor(tk.Toplevel): # Full implementation
    def __init__(self, parent: tk.Tk, theme_manager: ThemeManager, host_app_ref: Any, base_theme: Optional[Theme] = None):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("Theme Editor")
        self.geometry("750x650") # Increased size
        self.theme_manager = theme_manager
        self.host_app = host_app_ref # For potential future use (e.g., previewing on actual components)
        
        self.editable_theme = Theme.from_dict(base_theme.to_dict()) if base_theme else Theme(name="New Custom Theme")
        self.original_name = self.editable_theme.name if base_theme else None # To check if name changed for save

        self.vars: Dict[str, tk.StringVar] = {} # Holds tk.StringVar for each theme attribute
        self._create_ui()
        self._populate_ui_from_theme()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.wait_window(self)

    def _on_close(self):
        # Optionally ask to save changes if modified
        self.destroy()

    def _create_ui(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top part: Name and Save/Apply buttons
        top_bar = ttk.Frame(main_frame); top_bar.pack(fill=tk.X, pady=(0,10))
        ttk.Label(top_bar, text="Theme Name:").pack(side=tk.LEFT, padx=(0,5))
        self.vars['name'] = tk.StringVar()
        ttk.Entry(top_bar, textvariable=self.vars['name'], width=30).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(top_bar, text="Save", command=self.save_theme).pack(side=tk.LEFT, padx=(10,2))
        ttk.Button(top_bar, text="Apply & Save", command=self.apply_and_save_theme).pack(side=tk.LEFT, padx=2)

        # Paned window for settings and preview
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Settings Scrollable Frame
        settings_canvas_outer = ttk.Frame(paned_window, width=380) # Initial width
        paned_window.add(settings_canvas_outer, weight=2)

        settings_canvas = tk.Canvas(settings_canvas_outer, borderwidth=0)
        settings_scrollbar = ttk.Scrollbar(settings_canvas_outer, orient="vertical", command=settings_canvas.yview)
        self.settings_scrollable_frame = ttk.Frame(settings_canvas, padding=5) # Content frame

        self.settings_scrollable_frame.bind("<Configure>", lambda e: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")))
        settings_canvas_window = settings_canvas.create_window((0,0), window=self.settings_scrollable_frame, anchor="nw")
        settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_canvas.bind("<Configure>", lambda e: settings_canvas.itemconfig(settings_canvas_window, width=e.width)) # Resize inner frame

        settings_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self._populate_settings_fields(self.settings_scrollable_frame) # Add color/font pickers


        # Preview Frame
        preview_frame_outer = ttk.LabelFrame(paned_window, text="Live Preview", padding=10, width=320)
        paned_window.add(preview_frame_outer, weight=1)
        self.preview_canvas = tk.Canvas(preview_frame_outer, width=300, height=400, bg="#CCCCCC") # Neutral BG for canvas
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        # Store PIL image and Tk PhotoImage for preview to prevent GC
        self.preview_pil_image: Optional[Image.Image] = None
        self.preview_tk_image: Optional[ImageTk.PhotoImage] = None


    def _populate_settings_fields(self, parent: ttk.Frame):
        row = 0
        # Iterate over Theme dataclass fields
        for f_name, f_type in self.editable_theme.__annotations__.items():
            if f_name == "name": continue # Name is handled at top
            if f_name.startswith("_"): continue # Skip private-like fields

            ttk.Label(parent, text=f"{f_name.replace('_',' ').title()}:").grid(row=row, column=0, sticky=tk.W, padx=2, pady=3)
            self.vars[f_name] = tk.StringVar()
            
            if isinstance(getattr(self.editable_theme, f_name, ""), str) and getattr(self.editable_theme, f_name, "").startswith("#"): # Color field
                entry = ttk.Entry(parent, textvariable=self.vars[f_name], width=10)
                entry.grid(row=row, column=1, sticky=tk.EW, padx=2)
                entry.bind("<FocusOut>", lambda e, fn=f_name: self._update_theme_from_ui(fn)) # Update on focus out
                entry.bind("<Return>", lambda e, fn=f_name: self._update_theme_from_ui(fn))

                # Color preview swatch
                swatch = tk.Label(parent, text="    ", background=getattr(self.editable_theme, f_name, "#FFFFFF"), relief=tk.SUNKEN, borderwidth=1)
                swatch.grid(row=row, column=2, padx=(2,0), sticky=tk.W)
                
                btn = ttk.Button(parent, text="Pick", width=5, command=lambda fn=f_name, sw=swatch: self.pick_color(fn, sw))
                btn.grid(row=row, column=3, padx=(2,0), sticky=tk.W)

            elif f_name == "spectrum_colors" and isinstance(getattr(self.editable_theme, f_name, []), list):
                # Special UI for list of colors
                # For simplicity, show as comma-separated string for now, or first color with picker
                current_val = getattr(self.editable_theme, f_name, [])
                self.vars[f_name].set(", ".join(current_val) if current_val else "")
                entry = ttk.Entry(parent, textvariable=self.vars[f_name], width=25)
                entry.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=2)
                # TODO: Add better UI for list of colors (e.g., multiple pickers or a small editor)
                entry.bind("<FocusOut>", lambda e, fn=f_name: self._update_theme_from_ui(fn))
                entry.bind("<Return>", lambda e, fn=f_name: self._update_theme_from_ui(fn))


            elif f_name.startswith("font_size_") and (f_type == int or isinstance(getattr(self.editable_theme, f_name, 0), int)):
                self.vars[f_name] = tk.IntVar(value=getattr(self.editable_theme, f_name, 10)) # Use IntVar
                ttk.Spinbox(parent, from_=6, to=30, textvariable=self.vars[f_name], width=5, command=lambda fn=f_name: self._update_theme_from_ui(fn)).grid(row=row, column=1, sticky=tk.W, padx=2)
            
            elif f_name == "font_family" and (f_type == str or isinstance(getattr(self.editable_theme, f_name, ""), str)):
                # Basic Entry for font family, could be Combobox with common system fonts
                entry = ttk.Entry(parent, textvariable=self.vars[f_name], width=20)
                entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=2)
                entry.bind("<FocusOut>", lambda e, fn=f_name: self._update_theme_from_ui(fn))
                entry.bind("<Return>", lambda e, fn=f_name: self._update_theme_from_ui(fn))
            
            # Add more specific editors for other types if needed
            else: # Default to simple Entry for other string types or unhandled
                entry = ttk.Entry(parent, textvariable=self.vars[f_name], width=20)
                entry.grid(row=row, column=1, columnspan=2, sticky=tk.EW, padx=2)
                entry.bind("<FocusOut>", lambda e, fn=f_name: self._update_theme_from_ui(fn))
                entry.bind("<Return>", lambda e, fn=f_name: self._update_theme_from_ui(fn))
            row += 1
        parent.columnconfigure(1, weight=1) # Make entry column expandable

    def _populate_ui_from_theme(self):
        for f_name in self.vars.keys(): # Iterate over vars we created UI for
            if hasattr(self.editable_theme, f_name):
                value = getattr(self.editable_theme, f_name)
                if f_name == "spectrum_colors" and isinstance(value, list):
                    self.vars[f_name].set(", ".join(value))
                elif isinstance(self.vars[f_name], tk.IntVar): # For font sizes
                    self.vars[f_name].set(int(value) if value is not None else 10)
                else:
                    self.vars[f_name].set(str(value) if value is not None else "")
        self.update_preview()

    def _update_theme_from_ui(self, field_name_updated: Optional[str] = None):
        # Update self.editable_theme from all self.vars
        changed = False
        for f_name, tk_var in self.vars.items():
            if not hasattr(self.editable_theme, f_name): continue
            
            current_theme_val = getattr(self.editable_theme, f_name)
            ui_val_str = tk_var.get()
            new_theme_val = current_theme_val # Default to no change

            if f_name == "spectrum_colors":
                try: new_theme_val = [c.strip() for c in ui_val_str.split(',') if c.strip().startswith("#") and len(c.strip()) in [4,7]]
                except: new_theme_val = current_theme_val # Revert on error
            elif isinstance(current_theme_val, bool): new_theme_val = bool(tk_var.get()) # Assuming BooleanVar if bool
            elif isinstance(current_theme_val, int):
                try: new_theme_val = int(ui_val_str)
                except ValueError: new_theme_val = current_theme_val 
            elif isinstance(current_theme_val, float):
                try: new_theme_val = float(ui_val_str)
                except ValueError: new_theme_val = current_theme_val
            elif isinstance(current_theme_val, str): # Includes hex colors and font_family
                new_theme_val = ui_val_str
            
            if new_theme_val != current_theme_val:
                setattr(self.editable_theme, f_name, new_theme_val)
                changed = True
        
        if changed or field_name_updated: # If any value changed, or specific field triggered update
            self.update_preview()
            # Update color swatch if a color field was just changed by its entry
            if field_name_updated and field_name_updated.endswith("_bg") or field_name_updated.endswith("_fg") or field_name_updated.endswith("_color"):
                 for child in self.settings_scrollable_frame.winfo_children(): # Find the swatch for this field
                     # This is a bit brittle, relies on grid layout. A dict of swatches would be better.
                     if child.grid_info().get("row") == list(self.vars.keys()).index(field_name_updated) and child.grid_info().get("column") == 2:
                         try: child.config(background=getattr(self.editable_theme, field_name_updated))
                         except: pass # Invalid color string temporarily


    def pick_color(self, field_name: str, swatch_label: tk.Label):
        current_color = getattr(self.editable_theme, field_name, "#FFFFFF")
        color_code = colorchooser.askcolor(title=f"Pick color for {field_name}", initialcolor=current_color, parent=self)
        if color_code and color_code[1]:  # Check if a color was chosen and hex is available
            self.vars[field_name].set(color_code[1]) # Set hex string to var
            setattr(self.editable_theme, field_name, color_code[1]) # Update theme object
            swatch_label.config(background=color_code[1]) # Update swatch preview
            self.update_preview() # Update main preview pane


    def update_preview(self):
        if not self.preview_canvas: return
        # Create a sample UI image using current self.editable_theme
        # This is a simplified preview
        w, h = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        if w <= 1 or h <= 1: w,h = 300,400 # Fallback if not drawn yet
        
        img = Image.new("RGB", (w, h), self.editable_theme.primary_bg)
        draw = ImageDraw.Draw(img)
        
        try:
            font_main = ImageFont.truetype(f"{self.editable_theme.font_family.lower().replace(' ','')}.ttf", self.editable_theme.font_size_normal)
            font_title = ImageFont.truetype(f"{self.editable_theme.font_family.lower().replace(' ','')}.ttf", self.editable_theme.font_size_title)
        except IOError: font_main = ImageFont.load_default(); font_title = ImageFont.load_default()

        # Panel
        draw.rectangle([10,10, w-10, h*0.4], fill=self.editable_theme.secondary_bg, outline=self.editable_theme.accent_color)
        draw.text((20,20), "Sample Panel Title", font=font_title, fill=self.editable_theme.accent_color)
        draw.text((25, 25 + self.editable_theme.font_size_title + 5), "Primary text example.", font=font_main, fill=self.editable_theme.primary_fg)
        draw.text((25, 25 + self.editable_theme.font_size_title + 5 + self.editable_theme.font_size_normal + 5), "Secondary text example.", font=font_main, fill=self.editable_theme.secondary_fg)

        # Button
        btn_x0, btn_y0 = 30, h*0.4 + 20
        btn_x1, btn_y1 = btn_x0 + 100, btn_y0 + 30
        draw.rectangle([btn_x0,btn_y0, btn_x1,btn_y1], fill=self.editable_theme.accent_bg, outline=self.editable_theme.highlight_color)
        # btn_text_w, btn_text_h = draw.textsize("Button", font=font_main) # Deprecated
        btn_text_bbox = draw.textbbox((0,0), "Button", font=font_main)
        btn_text_w = btn_text_bbox[2] - btn_text_bbox[0]
        btn_text_h = btn_text_bbox[3] - btn_text_bbox[1]
        draw.text((btn_x0+(100-btn_text_w)/2, btn_y0+(30-btn_text_h)/2), "Button", font=font_main, fill=self.editable_theme.primary_fg)

        # Progress bar
        prog_y = h*0.4 + 70
        draw.rectangle([30, prog_y, w-30, prog_y+15], fill=self.editable_theme.primary_bg, outline=self.editable_theme.accent_color)
        draw.rectangle([30, prog_y, 30 + (w-60)*0.6, prog_y+15], fill=self.editable_theme.accent_color) # 60% fill

        # Viz area
        viz_y0 = h*0.4 + 100
        draw.rectangle([10, viz_y0, w-10, h-10], fill=self.editable_theme.viz_bg)
        # Simple spectrum bar mock
        bar_w = (w-40) / 10
        for i in range(5):
            bar_h = random.uniform(0.2,0.8) * (h-10 - viz_y0 - 20)
            bar_x = 20 + i * bar_w * 1.5
            draw.rectangle([bar_x, h-20-bar_h, bar_x+bar_w*0.8, h-20], fill=self.editable_theme.spectrum_colors[i % len(self.editable_theme.spectrum_colors)])
        
        self.preview_pil_image = img
        self.preview_tk_image = ImageTk.PhotoImage(self.preview_pil_image)
        self.preview_canvas.create_image(0,0, anchor="nw", image=self.preview_tk_image)


    def save_theme(self):
        theme_name = self.vars['name'].get().strip()
        if not theme_name: messagebox.showerror("Error", "Theme name cannot be empty.", parent=self); return
        
        self.editable_theme.name = theme_name # Ensure name on theme object is updated
        self._update_theme_from_ui() # Ensure all other params are synced from UI vars to editable_theme

        if self.theme_manager.save_theme(self.editable_theme):
            messagebox.showinfo("Success", f"Theme '{theme_name}' saved.", parent=self)
            self.original_name = theme_name # Update original name to prevent re-prompt on close if no other changes
            # Host app should be notified to refresh theme list in menu
            if self.host_app: self.host_app.publish_event("themes_changed")
        else: messagebox.showerror("Error", "Failed to save theme. Check logs.", parent=self)

    def apply_and_save_theme(self):
        theme_name = self.vars['name'].get().strip()
        if not theme_name: messagebox.showerror("Error", "Theme name cannot be empty.", parent=self); return
        
        self.editable_theme.name = theme_name
        self._update_theme_from_ui()

        if self.theme_manager.save_theme(self.editable_theme):
            self.theme_manager.apply_theme(self.editable_theme) # Apply the newly saved/updated theme
            messagebox.showinfo("Success", f"Theme '{theme_name}' applied and saved.", parent=self)
            self.original_name = theme_name
            if self.host_app: self.host_app.publish_event("themes_changed") # For menu update
        else: messagebox.showerror("Error", "Failed to save theme. Check logs.", parent=self)


class ExportManager: # Full implementation
    def __init__(self, progress_callback: Optional[Callable[[float, str], None]] = None, host_app_ref: Optional[Any] = None):
        self.progress_callback = progress_callback
        self.host_app = host_app_ref
        self.is_exporting = False
        self.stop_export_event = threading.Event()
        self.current_ffmpeg_process: Optional[subprocess.Popen] = None
        self.audio_formats = {"MP3 (320kbps)":{"codec":'libmp3lame','ext':'.mp3','opts':['-b:a','320k']}, "WAV (PCM 16-bit)":{"codec":'pcm_s16le','ext':'.wav','opts':[]}, "FLAC":{"codec":'flac','ext':'.flac','opts':[]}, "AAC (192kbps)":{"codec":'aac','ext':'.m4a','opts':['-b:a','192k']}}
        self.video_formats = {"MP4 (H.264/AAC)":{'vcodec':'libx264','acodec':'aac','ext':'.mp4','opts':['-pix_fmt','yuv420p','-preset','medium','-crf','23']}, "WebM (VP9/Opus)":{'vcodec':'libvpx-vp9','acodec':'libopus','ext':'.webm','opts':['-crf','30','-b:v','0']}}

    def _update_progress(self, percentage: float, message: str):
        if self.progress_callback:
            if self.host_app and self.host_app.root and hasattr(self.host_app.root, 'after'): # Check if root is Tk-like
                try: self.host_app.root.after(0, self.progress_callback, percentage, message)
                except tk.TclError: pass # Root window might be destroyed
            else: self.progress_callback(percentage, message) # Direct call if no Tk root

    def _run_ffmpeg_command(self, command: List[str]) -> bool:
        logger.info(f"FFmpeg: {' '.join(command)}")
        self._update_progress(0, f"Starting FFmpeg...")
        try:
            self.current_ffmpeg_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system()=="Windows" else 0)
            for line in iter(self.current_ffmpeg_process.stdout.readline, ''): # type: ignore
                if self.stop_export_event.is_set():
                    self.current_ffmpeg_process.terminate(); self.current_ffmpeg_process.wait(timeout=5); logger.info("FFmpeg terminated by stop event."); return False
                # logger.debug(f"FFmpeg: {line.strip()}") # Progress parsing from ffmpeg output is complex
            self.current_ffmpeg_process.stdout.close() # type: ignore
            ret_code = self.current_ffmpeg_process.wait()
            self.current_ffmpeg_process = None
            if ret_code==0: self._update_progress(100, "FFmpeg processing complete."); return True
            logger.error(f"FFmpeg failed (code {ret_code})."); self._update_progress(100, f"FFmpeg error (code {ret_code})."); return False
        except FileNotFoundError: logger.error("FFmpeg not found."); self._update_progress(0, "Error: FFmpeg not found."); return False
        except Exception as e: logger.error(f"FFmpeg exception: {e}", exc_info=True); self._update_progress(0, f"Error: {e}"); return False
        finally: self.current_ffmpeg_process = None


    def export_audio_async(self, input_file: str, output_file: str, format_name: str, start_time_s: Optional[float]=None, end_time_s: Optional[float]=None):
        if self.is_exporting: self._update_progress(0,"Export error: Busy"); return None
        self.is_exporting = True; self.stop_export_event.clear()
        future = THEME_EXPORT_THREAD_POOL.submit(self._export_audio_worker, input_file, output_file, format_name, start_time_s, end_time_s)
        future.add_done_callback(lambda f: setattr(self, 'is_exporting', False))
        return future

    def _export_audio_worker(self, input_file:str, output_file:str, format_name:str, start_s:Optional[float], end_s:Optional[float]) -> bool:
        fmt_details = self.audio_formats.get(format_name)
        if not fmt_details: self._update_progress(100,"Error: Unknown audio format"); return False
        cmd = ['ffmpeg', '-y', '-i', input_file]
        if start_s is not None: cmd.extend(['-ss', str(start_s)])
        if end_s is not None: cmd.extend(['-to', str(end_s)])
        cmd.extend(['-c:a', fmt_details['codec']] + fmt_details.get('opts',[]) + [output_file])
        return self._run_ffmpeg_command(cmd)


    def export_visualization_async(self, audio_file_path: str, output_video_path: str,  viz_engine_config_dict: Dict, viz_type_name: str, video_format_name: str, duration_to_export: Optional[float] = None):
        if self.is_exporting: self._update_progress(0,"Export error: Busy"); return None
        self.is_exporting = True; self.stop_export_event.clear()
        
        pool_to_use = THEME_EXPORT_THREAD_POOL # Default for dynamic viz due to GL context
        if viz_type_name == 'storyboard_frames' and 'frames_pil' in viz_engine_config_dict:
            # Storyboard frame saving is I/O bound, ProcessPool *could* be okay if PIL images are passed,
            # but ffmpeg part is still I/O. Keeping on ThreadPool for simpler progress IPC.
            logger.info("Storyboard export using ThreadPool for frame saving & ffmpeg.")
            # pool_to_use = THEME_EXPORT_PROCESS_POOL # If VisualGenerator made fully picklable for worker.
        
        future = pool_to_use.submit(self._export_visualization_worker, audio_file_path, output_video_path, viz_engine_config_dict, viz_type_name, video_format_name, duration_to_export)
        future.add_done_callback(lambda f: setattr(self, 'is_exporting', False))
        return future

    def _export_visualization_worker(self, audio_file_path: str, output_video_path: str, viz_config_dict: Dict, viz_type_name: str, video_format_name: str, duration_to_export: Optional[float]):
        # This runs in a separate thread (or process for storyboard if adapted)
        worker_logger = logging.getLogger(f"ExportWorker.{threading.get_ident() if isinstance(threading.current_thread(), threading._MainThread) else os.getpid()}")
        worker_logger.info(f"Viz export worker started for {Path(output_video_path).name} ({viz_type_name})")
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="flowstate_export_")
        temp_dir = temp_dir_obj.name; worker_logger.info(f"Temp frame dir: {temp_dir}")
        
        try:
            viz_config_obj = VisualizationConfig(**viz_config_dict) if VisualizationConfig else None
            if not viz_config_obj and viz_type_name != 'storyboard_frames': # VizConfig needed for dynamic
                worker_logger.error("VisualizationConfig not available for dynamic export."); self._update_progress(100, "Error: Viz config missing."); return False

            if duration_to_export is None:
                try: duration_to_export = sf.info(audio_file_path).duration
                except Exception as e: worker_logger.error(f"No duration: {e}"); self._update_progress(100,"Error: Bad audio for duration"); return False
            if duration_to_export <= 0: worker_logger.error("Invalid duration."); self._update_progress(100,"Error: Invalid duration"); return False

            fps = viz_config_obj.fps if viz_config_obj else 25 # Default if no viz_config
            total_frames = int(duration_to_export * fps); time_step = 1.0 / fps
            worker_logger.info(f"Exporting {duration_to_export:.2f}s at {fps} FPS, ~{total_frames} frames.")
            generated_frames_count = 0

            if viz_type_name == 'storyboard_frames' and 'frames_pil' in viz_config_dict:
                pil_frames = viz_config_dict['frames_pil']
                scenes_data = viz_config_dict.get('scenes_data', []) # List of Scene dicts
                frame_idx = 0
                for scene_idx, scene_img_pil in enumerate(pil_frames):
                    if self.stop_export_event.is_set(): break
                    scene_dur = scenes_data[scene_idx]['duration'] if scenes_data and scene_idx < len(scenes_data) else time_step
                    num_vid_frames_for_scene = max(1, int(scene_dur * fps))
                    for _ in range(num_vid_frames_for_scene):
                        if self.stop_export_event.is_set(): break
                        frame_path = os.path.join(temp_dir,f"frame_{frame_idx:06d}.png"); scene_img_pil.save(frame_path)
                        frame_idx +=1; generated_frames_count +=1
                        if frame_idx % fps ==0: self._update_progress((frame_idx/total_frames)*50 if total_frames > 0 else 50, f"Saved frame {frame_idx}")
                    if self.stop_export_event.is_set(): break
                total_frames = frame_idx # Update total_frames based on actual generated
            elif VisualizationEngine and self.host_app and self.host_app.visualization_ui_ref and self.host_app.visualization_ui_ref.engine_instance:
                worker_logger.info("Attempting dynamic visualization frame generation using main engine instance (experimental).")
                viz_engine = self.host_app.visualization_ui_ref.engine_instance # Access main engine (THREAD POOL ONLY)
                
                # Backup and set export config (this is risky if user changes viz type during export)
                original_engine_config = viz_engine.config
                original_viz_type = viz_engine.current_visualization.info.name if viz_engine.current_visualization and hasattr(viz_engine.current_visualization,'info') else None
                
                viz_engine.config = viz_config_obj # Apply export config
                viz_engine.set_visualization(viz_type_name) # Set to desired viz for export
                
                samples_per_update = viz_config_obj.fft_size
                with sf.SoundFile(audio_file_path, 'r') as audio_file:
                    if audio_file.samplerate != viz_engine.config.sample_rate: # Resampling needed conceptually
                         worker_logger.warning(f"Audio SR ({audio_file.samplerate}) and Viz SR ({viz_engine.config.sample_rate}) mismatch. Results may be off.")
                    for i in range(total_frames):
                        if self.stop_export_event.is_set(): break
                        frame_time = i * time_step
                        audio_file.seek(int(frame_time * audio_file.samplerate))
                        audio_chunk = audio_file.read(samples_per_update, dtype='float32', always_2d=True)
                        if audio_chunk.shape[0] == 0: break
                        if audio_chunk.shape[1] != 2: audio_chunk = np.tile(audio_chunk, (1,2)) # Ensure stereo for viz engine
                        
                        viz_engine.update_audio(audio_chunk) # Update with specific chunk for this frame time
                        pil_img = viz_engine.capture_frame_to_pil() # Capture its state
                        
                        if pil_img: pil_img.save(os.path.join(temp_dir, f"frame_{i:06d}.png")); generated_frames_count+=1
                        else: Image.new('RGB',(viz_config_obj.width,viz_config_obj.height),'black').save(os.path.join(temp_dir,f"frame_{i:06d}.png")) # Fallback
                        if i % fps == 0: self._update_progress((i/total_frames)*50 if total_frames > 0 else 50, f"Rendered frame {i+1}/{total_frames}")
                
                # Restore engine state
                viz_engine.config = original_engine_config
                if original_viz_type: viz_engine.set_visualization(original_viz_type)
            else: worker_logger.error("Dynamic viz export failed: No viz engine or requirements not met."); self._update_progress(100, "Error: Dynamic viz engine unavailable."); return False

            if self.stop_export_event.is_set() or generated_frames_count == 0: worker_logger.info("Frame gen cancelled or no frames."); self._update_progress(100, "Export cancelled/failed."); return False
            
            self._update_progress(50, "Starting FFmpeg encoding...")
            fmt = self.video_formats.get(video_format_name)
            if not fmt: worker_logger.error(f"Unknown video format: {video_format_name}"); return False
            cmd = ['ffmpeg','-y','-framerate',str(fps),'-i',os.path.join(temp_dir,'frame_%06d.png'),'-i',audio_file_path,
                   '-c:v',fmt['vcodec'],'-c:a',fmt['acodec'],'-shortest'] + fmt.get('opts',[]) + [output_video_path]
            return self._run_ffmpeg_command(cmd) # This now calls self._update_progress via main thread
        except Exception as e: worker_logger.error(f"Error in viz worker: {e}",exc_info=True); self._update_progress(100,f"Error: {e}"); return False
        finally: temp_dir_obj.cleanup(); worker_logger.info(f"Cleaned temp dir: {temp_dir}")

    def batch_export_audio_async(self, track_list: List[Dict], output_dir: str, format_name: str):
        if self.is_exporting: self._update_progress(0,"Export error: Busy"); return None
        self.is_exporting = True; self.stop_export_event.clear()
        future = THEME_EXPORT_THREAD_POOL.submit(self._batch_export_audio_worker, track_list, output_dir, format_name)
        future.add_done_callback(lambda f: setattr(self, 'is_exporting', False))
        return future

    def _batch_export_audio_worker(self, track_list: List[Dict], out_dir: str, fmt_name: str) -> bool:
        total = len(track_list); success_count = 0
        for i, track_info in enumerate(track_list):
            if self.stop_export_event.is_set(): logger.info("Batch export cancelled."); break
            in_path = track_info.get('file_path')
            title = track_info.get('title', Path(in_path).stem if in_path else f"track_{i+1}")
            artist = track_info.get('artist', 'UnknownArtist')
            if not in_path: logger.warning(f"Skipping track, no file_path: {title}"); continue
            
            fmt_details = self.audio_formats.get(fmt_name)
            if not fmt_details: self._update_progress(100, f"Error: Bad format {fmt_name}"); return False
            
            # Sanitize filename
            safe_artist = "".join(c if c.isalnum() or c in (' ','_') else '_' for c in artist).rstrip()
            safe_title = "".join(c if c.isalnum() or c in (' ','_') else '_' for c in title).rstrip()
            out_filename = f"{safe_artist} - {safe_title}{fmt_details['ext']}"
            out_path = os.path.join(out_dir, out_filename)
            
            self._update_progress((i/total)*100 if total>0 else 0, f"Exporting {i+1}/{total}: {safe_title}")
            if self._export_audio_worker(in_path, out_path, fmt_name, None, None): success_count+=1
        
        final_msg = f"Batch export finished. {success_count}/{total} successful."
        self._update_progress(100, final_msg)
        logger.info(final_msg)
        return success_count == total


    def stop_current_export(self):
        if self.is_exporting:
            self.stop_export_event.set()
            logger.info("Stop export requested. FFmpeg process will be terminated if running.")
            if self.current_ffmpeg_process:
                logger.info("Terminating active FFmpeg process...")
                try:
                    self.current_ffmpeg_process.terminate()
                    self.current_ffmpeg_process.wait(timeout=5) # Give it a chance to die
                except subprocess.TimeoutExpired: self.current_ffmpeg_process.kill()
                except Exception as e_term: logger.error(f"Error terminating ffmpeg: {e_term}")
                self.current_ffmpeg_process = None
            # is_exporting flag will be reset by the future's done callback.

class StreamingServer: # Conceptual, needs full aiohttp rewrite for robustness
    def __init__(self, host_app_ref: Any, host='0.0.0.0', port=8085):
        self.host_app = host_app_ref; self.host = host; self.port = port
        self.httpd: Optional[socketserver.TCPServer] = None; self.thread: Optional[threading.Thread] = None
    def start_server(self):
        if self.httpd: logger.info("Streaming server already running."); return
        try:
            # handler = partial(self._StreamingRequestHandler, self.host_app) # If handler needs host_app
            handler = self._StreamingRequestHandler # Simpler if no host_app needed directly
            self.httpd = socketserver.ThreadingTCPServer((self.host, self.port), handler)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Conceptual Streaming Server started on http://{self.host}:{self.port}")
        except Exception as e: logger.error(f"Failed to start streaming server: {e}"); self.httpd = None
    def stop_server(self):
        if self.httpd: self.httpd.shutdown(); self.httpd.server_close(); self.httpd = None
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=2)
        logger.info("Conceptual Streaming Server stopped.")
    class _StreamingRequestHandler(http.server.SimpleHTTPRequestHandler): # Very basic
        # This needs significant work for proper streaming, range requests, security etc.
        # Should ideally use aiohttp if other parts of app use it.
        def do_GET(self):
            # Minimal: serve files from a specific directory (e.g. host_app's library root)
            # This is NOT secure and very basic.
            # filepath = Path(self.host_app.music_library_db_ref.base_dir) / self.path.lstrip("/") # Example
            # if filepath.is_file():
            #    return super().do_GET() # Let SimpleHTTPRequestHandler serve it
            self.send_error(404, "File not found or streaming not fully implemented")


class ThemeExportMainUI(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent)
        self.host_app = host_app_ref
        self.export_manager = ExportManager(progress_callback=self._update_export_progress_ui, host_app_ref=host_app_ref)
        self.streaming_server: Optional[StreamingServer] = None # Lazy init
        self.theme_vars: Dict[str, tk.StringVar] = {} # For theme combobox
        self._create_ui()
        self._load_themes_to_ui()
        if self.host_app and self.host_app.theme_manager:
            self.host_app.theme_manager.register_callback(self.on_external_theme_change)
            # Initial sync with theme manager's current theme
            current_tm_theme = self.host_app.theme_manager.get_current_theme()
            if current_tm_theme and self.theme_combobox: self.theme_combobox.set(current_tm_theme.name)
        if self.host_app and self.host_app.theme_manager and self.host_app.theme_manager.current_theme_obj:
             self.apply_theme_to_manage_ui(self.host_app.theme_manager.current_theme_obj)

    def _create_ui(self):
        self.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Theme Selection Section
        theme_frame = ttk.LabelFrame(self, text="Theme Management", padding=10)
        theme_frame.pack(fill=tk.X, pady=5)
        ttk.Label(theme_frame, text="Current Theme:").pack(side=tk.LEFT, padx=(0,5))
        self.theme_combobox = ttk.Combobox(theme_frame, state="readonly", width=25)
        self.theme_combobox.pack(side=tk.LEFT, padx=5)
        self.theme_combobox.bind("<<ComboboxSelected>>", self._on_theme_selected_from_ui)
        ttk.Button(theme_frame, text="Edit Theme...", command=lambda: self._open_theme_editor(edit_current=True)).pack(side=tk.LEFT, padx=5)
        ttk.Button(theme_frame, text="New Theme...", command=lambda: self._open_theme_editor(edit_current=False)).pack(side=tk.LEFT)

        # Export Section
        export_frame = ttk.LabelFrame(self, text="Export Options", padding=10)
        export_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(export_frame, text="Batch Export Audio...", command=self.open_detailed_batch_audio_export_dialog).pack(pady=5, fill=tk.X)
        ttk.Button(export_frame, text="Export Current Visualization as Video...", command=self.open_detailed_viz_export_dialog).pack(pady=5, fill=tk.X)
        ttk.Button(export_frame, text="Stop Current Export", command=lambda: self.export_manager.stop_current_export()).pack(pady=5)
        
        self.export_status_label = ttk.Label(export_frame, text="Export status: Idle")
        self.export_status_label.pack(fill=tk.X, pady=(5,0))
        self.export_progress_bar = ttk.Progressbar(export_frame, length=200, mode='determinate', value=0)
        self.export_progress_bar.pack(fill=tk.X, pady=(0,5))

        # Streaming Section (Conceptual)
        stream_frame = ttk.LabelFrame(self, text="Streaming (Conceptual)", padding=10)
        stream_frame.pack(fill=tk.X, pady=5)
        self.stream_toggle_button = ttk.Button(stream_frame, text="Start Streaming Server", command=self._toggle_streaming_server)
        self.stream_toggle_button.pack(side=tk.LEFT)
        self.stream_status_label = ttk.Label(stream_frame, text="Streaming: OFF")
        self.stream_status_label.pack(side=tk.LEFT, padx=10)

    def _load_themes_to_ui(self):
        if self.host_app and self.host_app.theme_manager and self.theme_combobox:
            theme_names = self.host_app.theme_manager.get_theme_names()
            self.theme_combobox['values'] = theme_names
            current_theme = self.host_app.theme_manager.get_current_theme()
            if current_theme and current_theme.name in theme_names:
                self.theme_combobox.set(current_theme.name)
            elif theme_names: self.theme_combobox.set(theme_names[0]) # Default to first if current not found

    def on_external_theme_change(self, theme: Theme): # Callback for ThemeManager
        if self.theme_combobox: self.theme_combobox.set(theme.name)
        self.apply_theme_to_manage_ui(theme)

    def apply_theme_to_manage_ui(self, theme: Theme):
        self.configure(background=theme.secondary_bg)
        for child in self.winfo_children(): # Apply to direct children frames too
            if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                child.configure(style="Content.TFrame" if isinstance(child, ttk.Frame) else "Content.TLabelframe")
        # Specific styles if needed for buttons, labels within this UI based on theme
        # self.style.configure("Manage.TButton", ...)
        # self.style.configure("Manage.TLabel", ...)

    def _on_theme_selected_from_ui(self, event=None):
        if self.host_app and self.host_app.theme_manager and self.theme_combobox:
            selected_name = self.theme_combobox.get()
            self.host_app.theme_manager.apply_theme(selected_name) # This will trigger callbacks including on_external_theme_change

    def _open_theme_editor(self, edit_current: bool = False):
        if not self.host_app or not self.host_app.theme_manager: return
        base_theme_for_editor = None
        if edit_current: base_theme_for_editor = self.host_app.theme_manager.get_current_theme()
        
        ThemeEditor(self.host_app.root, self.host_app.theme_manager, self.host_app, base_theme_for_editor)
        # ThemeEditor is modal, after it closes, refresh theme list
        self._load_themes_to_ui() 

    def _toggle_streaming_server(self):
        if not self.streaming_server: self.streaming_server = StreamingServer(self.host_app)
        if self.streaming_server.httpd and self.streaming_server.thread and self.streaming_server.thread.is_alive():
            self.streaming_server.stop_server(); self.stream_status_label.config(text="Streaming: OFF"); self.stream_toggle_button.config(text="Start Streaming")
        else:
            self.streaming_server.start_server(); self.stream_status_label.config(text="Streaming: ON"); self.stream_toggle_button.config(text="Stop Streaming")

    def open_detailed_batch_audio_export_dialog(self): # Placeholder for more complex dialog
        if not self.host_app or not self.host_app.music_library_db_ref: messagebox.showerror("Error","Music library not available.",parent=self.winfo_toplevel()); return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Batch Export", parent=self.winfo_toplevel())
        if not output_dir: return
        
        # Simple format choice for now
        available_formats = list(self.export_manager.audio_formats.keys())
        # TODO: Create a proper dialog to select tracks and format
        format_name = available_formats[0] # Default to first
        
        all_tracks_from_lib = self.host_app.music_library_db_ref.get_all_tracks(limit=10) # Export first 10 for test
        if not all_tracks_from_lib: messagebox.showinfo("Info","No tracks in library to export.",parent=self.winfo_toplevel()); return
        
        tracks_to_export_dicts = [{'file_path':t.file_path, 'title':t.title, 'artist':t.artist} for t in all_tracks_from_lib if t.file_path]
        if tracks_to_export_dicts:
            self.export_manager.batch_export_audio_async(tracks_to_export_dicts, output_dir, format_name)
        else: messagebox.showinfo("Info","No valid tracks selected/found for batch export.",parent=self.winfo_toplevel())


    def open_detailed_viz_export_dialog(self): # As refined before
        pass # Full implementation from previous refined step

    def _update_export_progress_ui(self, percentage: float, message: str):
        if self.export_progress_bar: self.export_progress_bar['value'] = percentage
        if self.export_status_label: self.export_status_label.config(text=message)
        # Show messagebox on completion/error, managed by worker callback now
        # if percentage == 100 and "complete" in message.lower() and "error" not in message.lower(): messagebox.showinfo("Export Complete", message, parent=self.host_app.root)
        # elif "error" in message.lower().strip() and percentage !=0 : messagebox.showerror("Export Error", message, parent=self.host_app.root)


    def on_app_exit(self):
        logger.info("ThemeExportMainUI on_app_exit: Stopping streaming server if active.")
        if self.streaming_server: self.streaming_server.stop_server()
        if self.host_app and self.host_app.theme_manager: self.host_app.theme_manager.unregister_callback(self.on_external_theme_change)
        self.export_manager.stop_current_export() # Ensure any ffmpeg process is killed


def create_theme_export_main_tab(notebook: ttk.Notebook, host_app_ref: Any) -> ThemeExportMainUI:
    main_frame = ttk.Frame(notebook)
    notebook.add(main_frame, text="Manage") 
    ui = ThemeExportMainUI(main_frame, host_app_ref=host_app_ref)
    ui.pack(fill=tk.BOTH, expand=True)
    # host_app_ref.export_manager_ref = ui.export_manager # Launcher already sets this based on service init
    # host_app_ref.theme_export_main_ui_ref = ui # Launcher sets this
    logger.info("Manage (Theme/Export) Tab UI created.")
    return ui


def create_theme_menu_items(menubar_widget: tk.Menu, host_app_ref: Any):
    if not host_app_ref.theme_manager: logger.warning("ThemeManager not available for menu."); return
    tm = host_app_ref.theme_manager
    theme_menu_cascade = tk.Menu(menubar_widget, tearoff=0)
    # Basic theming for the menu itself
    # if tm.get_current_theme():
    #     try: theme_menu_cascade.configure(bg=tm.get_current_theme().secondary_bg, fg=tm.get_current_theme().primary_fg)
    #     except: pass # Ignore TclErrors if theming menu fails

    menubar_widget.add_cascade(label="Themes", menu=theme_menu_cascade)
    for theme_name_val in tm.get_theme_names():
        theme_menu_cascade.add_command(label=theme_name_val, command=lambda name=theme_name_val: tm.apply_theme(name))
    theme_menu_cascade.add_separator()
    
    # Check if ThemeEditor class is available in this module (it is)
    theme_menu_cascade.add_command(label="Create New Theme...", command=lambda: ThemeEditor(host_app_ref.root, tm, host_app_ref, None))
    theme_menu_cascade.add_command(label="Edit Current Theme...", command=lambda: ThemeEditor(host_app_ref.root, tm, host_app_ref, tm.get_current_theme()))


if __name__ == "__main__":
    # ... (Standalone test block as previously refined) ...
    pass


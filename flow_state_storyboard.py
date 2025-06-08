

"""
Flow State: Storyboard Generator Module
AI-powered visual storyboard generation from music and lyrics
"""

import tkinter as tk
from tkinter import ttk, Canvas, messagebox, scrolledtext # Added scrolledtext if used for anything
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageColor 
import threading
# import queue # Not directly used
import json 
import re
import os
from dataclasses import dataclass, field, asdict 
from typing import List, Dict, Tuple, Optional, Any
import colorsys
import random
import concurrent.futures
import platform # For default font selection

# For NLP analysis
import nltk
try:
    nltk.data.find('tokenizers/punkt') 
except LookupError:
    try: nltk.download('punkt', quiet=True, raise_on_error=False)
    except Exception as e_nltk_dl: logger.warning(f"NLTK 'punkt' download attempt failed/skipped: {e_nltk_dl}")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try: nltk.download('averaged_perceptron_tagger', quiet=True, raise_on_error=False)
    except Exception as e_nltk_dl: logger.warning(f"NLTK 'averaged_perceptron_tagger' download attempt failed/skipped: {e_nltk_dl}")

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import logging
logger = logging.getLogger("FlowStateStoryboard")

STORYBOARD_PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2))
STORYBOARD_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="StoryboardIO")


@dataclass
class Scene: 
    timestamp: float 
    duration: float  
    text: str        
    mood: str        
    color_palette: List[str] 
    visual_elements: List[Tuple[str, str]] 
    transition_type: str 
    energy_level: float  
    thumbnail_pil: Optional[Image.Image] = field(default=None, repr=False, compare=False)
    thumbnail_tk: Optional[ImageTk.PhotoImage] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict: 
        d = asdict(self)
        d.pop('thumbnail_pil', None); d.pop('thumbnail_tk', None)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'Scene':
        field_names = {f.name for f in cls.__dataclass_fields__ if f.name not in ['thumbnail_pil', 'thumbnail_tk']}
        init_data = {k: data[k] for k in data if k in field_names}
        return cls(**init_data)


class LyricAnalyzer: 
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['joy','happy','smile','laugh','bright','sun','sunshine','celebrate','glee','delight','fun','playful','merry','light','paradise','heaven', 'wonderful', 'fantastic', 'amazing'],
            'sad': ['cry','tear','sad','sorrow','pain','hurt','lonely','grief','gloomy','blue','heartbreak','alone','empty','lost','misery','despair', 'sorrowful', 'weep', 'mourn'],
            'angry': ['anger','rage','fury','mad','hate','fight','scream','storm','ire','war','battle','destroy','revenge','bitter','frustrated', 'explode', 'furious'],
            'calm': ['peace','calm','serene','quiet','gentle','soft','tranquil','still','hush','rest','breeze','flow','smooth','meditate', 'serenity', 'harmony'],
            'energetic': ['run','jump','dance','move','fast','wild','free','power','alive','rush','explode','fly','soar','fire','electric','dynamic','pulse', 'vibrant', 'ignite'],
            'romantic': ['love','heart','kiss','embrace','together','forever','soul','passion','desire','darling','sweetheart','adore','cherish','stars','moonlight', 'romance', 'beauty'],
            'fear': ['fear','scared','terror','horror','afraid','ghost','dark','nightmare','shadow','haunt','panic','danger','monster', 'anxious', 'dread'],
            'hope': ['hope','dream','wish','future','believe','aspire','faith','dawn','morning','new','miracle','star', 'vision', 'inspire'],
            'reflective': ['think','wonder','memory','past','reflect','question','why','if','maybe','search','journey','time','yesterday', 'contemplate', 'ponder', 'reminisce'],
            'strong': ['strong','brave','courage','conquer','unbreakable','stand','rise','strength','mighty','bold', 'valiant', 'heroic', 'resilient']
        }
        self.visual_elements_map = { 
            'nature': ['tree','forest','mountain','ocean','river','sky','cloud','sun','moon','star','flower','field','rain','snow','wind','leaf','grass','sea','wave','stone','earth','desert','island','valley','hill', 'canyon', 'waterfall', 'meadow'],
            'urban': ['city','street','road','building','window','door','car','train','bridge','light','neon','sign','town','alley','rooftop','concrete','steel','traffic', 'skyscraper', 'subway', 'avenue'],
            'abstract': ['color','shape','line','pattern','spiral','circle','blur','glow','void','dream','light','darkness','swirl','fractal','energy','aura', 'texture', 'gradient', 'pulse'],
            'human': ['face','eye','hand','heart','smile','tear','shadow','figure','silhouette','person','body','soul','mind','voice','breath','blood','skin', 'footprint', 'reflection'],
            'object': ['key','book','clock','mirror','letter','phone','ship','road','path','door','gate','sword','ring','cup','fire','time','sound', 'candle', 'photograph', 'map'],
            'celestial': ['sun','moon','star','galaxy','planet','comet','sky','universe','space','nebula', 'cosmos', 'orbit', 'eclipse'],
            'elements': ['fire','water','air','earth','ice','flame','ash','dust','smoke','lightning','thunder', 'lava', 'steam', 'mist']
        }

    def analyze_line(self, text: str) -> Dict[str, Any]:
        if not text: 
            return {'mood':'neutral','visual_elements':[],'imagery_keywords':[],'energy':0.1,'main_nouns':[],'main_verbs':[],'main_adjectives':[]}

        text_lower = text.lower(); tokens = word_tokenize(text_lower); pos_tags = pos_tag(tokens)
        mood_scores = {mood: 0 for mood in self.emotion_keywords}
        for token in tokens:
            for mood, keywords in self.emotion_keywords.items():
                if token in keywords: mood_scores[mood] += 1
        
        primary_mood = 'neutral'
        if any(mood_scores.values()):
            max_score = max(mood_scores.values())
            if max_score > 0: top_moods = [m for m,s in mood_scores.items() if s==max_score]; primary_mood = random.choice(top_moods)
        
        visual_elements_found = []
        for token in tokens:
            for category, elements in self.visual_elements_map.items():
                if token in elements: visual_elements_found.append((category, token))
        visual_elements_found = list(dict.fromkeys(visual_elements_found)) 
        
        main_nouns = [w for w,p in pos_tags if p.startswith('NN') and len(w)>2 and w not in nltk.corpus.stopwords.words('english')]
        main_verbs = [w for w,p in pos_tags if p.startswith('VB') and len(w)>2 and w not in nltk.corpus.stopwords.words('english')]
        main_adjectives = [w for w,p in pos_tags if p.startswith('JJ') and len(w)>2 and w not in nltk.corpus.stopwords.words('english')]
        imagery_keywords = list(set(main_nouns + main_verbs + main_adjectives))
        energy = self._calculate_energy(text_lower, tokens, pos_tags)
        
        return {'mood':primary_mood,'visual_elements':visual_elements_found[:3],'imagery_keywords':imagery_keywords[:5],'energy':np.clip(energy,0.0,1.0),'main_nouns':main_nouns[:3],'main_verbs':main_verbs[:2],'main_adjectives':main_adjectives[:2]}

    def _calculate_energy(self, text_lower: str, tokens: List[str], pos_tags: List[Tuple[str,str]]) -> float:
        energy_score = 0.0
        energetic_hits = sum(1 for t in tokens if t in self.emotion_keywords.get('energetic',[]))
        strong_hits = sum(1 for t in tokens if t in self.emotion_keywords.get('strong',[]))
        energy_score += (energetic_hits * 0.25) + (strong_hits * 0.15)
        if '!' in text_lower: energy_score += 0.2
        num_verbs = len([tag for _,tag in pos_tags if tag.startswith('VB')])
        if tokens: energy_score += min(0.3, (num_verbs/len(tokens))*0.5) # Max 0.3 from verb density
        if tokens and len(tokens) < 7 and len(tokens) > 0: energy_score += 0.1 # Shorter phrases boost
        return np.clip(energy_score, 0.0, 1.0)


class VisualGenerator:
    def __init__(self, width: int = 320, height: int = 180):
        self.width = width; self.height = height
        default_font_family = "Arial"
        if platform.system() == "Windows": default_font_family = "Segoe UI"
        elif platform.system() == "Darwin": default_font_family = "Helvetica Neue"
        self.default_font_size = max(12, int(height / 16)) # Slightly smaller default
        self.small_font_size = max(10, int(height / 20))
        try:
            self.font = ImageFont.truetype(f"{default_font_family.lower().replace(' ','')}.ttf", self.default_font_size)
            self.small_font = ImageFont.truetype(f"{default_font_family.lower().replace(' ','')}.ttf", self.small_font_size)
        except IOError:
            logger.warning(f"{default_font_family} font not found, using default PIL font."); self.font=ImageFont.load_default(); self.small_font=ImageFont.load_default()
        self.palettes = LyricAnalyzer().emotion_keywords.keys() # Get all moods
        self.palettes = { mood: [ImageColor.getrgb(f"hsl({random.randint(0,360)}, {random.randint(40,80)}%, {random.randint(30,70)}%)") for _ in range(5)] for mood in self.palettes}
        # Overwrite with some predefined ones for better aesthetics
        self.palettes.update({
            'happy': ["#FFD700", "#FF8C00", "#FF6347", "#ADFF2F", "#FFFACD"], 
            'sad': ["#191970", "#4682B4", "#B0C4DE", "#778899", "#E6E6FA"], 
            'angry': ["#DC143C", "#FF0000", "#8B0000", "#A52A2A", "#600000"], 
            'calm': ["#ADD8E6", "#AFEEEE", "#E0FFFF", "#F0FFFF", "#B0E0E6"], 
            'energetic': ["#FF4500", "#FF69B4", "#FFFF00", "#7FFF00", "#1E90FF"], 
            'romantic': ["#FFC0CB", "#FF69B4", "#DB7093", "#DA70D6", "#FFB6C1"],
            'neutral': ["#BEBEBE", "#DCDCDC", "#F5F5F5", "#E0E0E0", "#C8C8C8"]
        }) # Ensure all moods in LyricAnalyzer have a fallback
        for mood in LyricAnalyzer().emotion_keywords.keys():
            if mood not in self.palettes: self.palettes[mood] = self.palettes['neutral']


    def generate_scene_image(self, scene_data: Scene) -> Image.Image:
        img = Image.new("RGB", (self.width, self.height), color="#181818")
        draw = ImageDraw.Draw(img)
        palette = self.palettes.get(scene_data.mood, self.palettes['neutral'])
        if not palette: palette = self.palettes['neutral'] # Extra safety
        active_palette = random.sample(palette, min(len(palette), 5)) # Use a subset

        self._draw_gradient_background(draw, active_palette, scene_data.energy)
        
        drawn_specific = False
        if scene_data.visual_elements:
            cat, tok = scene_data.visual_elements[0]
            if cat == 'nature': self._draw_nature_inspired(draw,active_palette,scene_data.energy,tok); drawn_specific=True
            elif cat == 'urban': self._draw_urban_inspired(draw,active_palette,scene_data.energy,tok); drawn_specific=True
            elif cat == 'celestial': self._draw_celestial_inspired(draw,active_palette,scene_data.energy,tok); drawn_specific=True
            elif cat == 'elements': self._draw_elemental_inspired(draw,active_palette,scene_data.energy,tok); drawn_specific=True
        if not drawn_specific: self._draw_abstract_mood_patterns(draw,active_palette,scene_data.mood,scene_data.energy)
        
        self._add_text_overlay_styled(draw, scene_data.text, active_palette, scene_data.mood)
        return img

    def _draw_gradient_background(self, draw: ImageDraw.ImageDraw, palette: List[str], energy: float):
        c1_str, c2_str = random.sample(palette, 2) if len(palette) >= 2 else (palette[0], palette[0])
        c1 = ImageColor.getrgb(c1_str); c2 = ImageColor.getrgb(c2_str)
        angle = random.uniform(0, 360) * (1 + energy * 0.5) # More dynamic angle with energy
        for y in range(self.height):
            # Simple linear gradient based on y, angle could be used for more complex rotation
            ratio = y / self.height
            if energy > 0.6 and (y % (int(max(1, (1-energy)*20)))) < 2 : ratio = 1 - ratio # Add some banding for high energy
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))

    def _draw_nature_inspired(self, draw: ImageDraw.ImageDraw, palette: List[str], energy: float, token: Optional[str]=None):
        # ... (More varied drawing logic based on token, e.g. simple tree, wave, flower shapes)
        color = random.choice(palette)
        if token == 'tree':
            trunk_w = int(self.width * (0.02 + energy*0.03))
            draw.rectangle([(self.width/2 - trunk_w/2, self.height*0.6), (self.width/2 + trunk_w/2, self.height)], fill=ImageColor.getrgb(palette[1]))
            canopy_r = int(self.width * (0.1 + energy*0.2))
            draw.ellipse([(self.width/2 - canopy_r, self.height*0.2), (self.width/2 + canopy_r, self.height*0.6)], fill=ImageColor.getrgb(palette[0]))
        elif token in ['wave', 'ocean', 'river']:
            for i in range(3 + int(energy*5)):
                y_start = self.height * (0.3 + 0.6 * random.random())
                amplitude = self.height * (0.05 + energy * 0.1)
                points = [(x, y_start + math.sin(x * 0.05 + i*0.5) * amplitude) for x in range(self.width)]
                draw.line(points, fill=ImageColor.getrgb(random.choice(palette[:2])), width=1+int(energy*2))
        else: # Generic nature: flowing lines or leaf shapes
            for _ in range(int(5 + energy * 10)):
                x1,y1 = random.randint(0,self.width), random.randint(0,self.height)
                x2,y2 = x1+random.randint(-50,50), y1+random.randint(-50,50)
                draw.line([(x1,y1),(x2,y2)], fill=ImageColor.getrgb(color), width=1+int(energy*2))


    def _draw_urban_inspired(self, draw: ImageDraw.ImageDraw, palette: List[str], energy: float, token: Optional[str]=None):
        # ... (Rectangles for buildings, lines for streets, glowing circles for lights)
        for _ in range(int(3 + energy * 7)): # Buildings
            w = random.randint(self.width//10, self.width//4)
            h = random.randint(self.height//3, int(self.height*0.8 + energy*self.height*0.15) )
            x = random.randint(0, self.width - w)
            y = self.height - h # Align to bottom
            draw.rectangle([(x,y), (x+w, y+h)], fill=ImageColor.getrgb(random.choice(palette[1:])))
        if token == 'light' or energy > 0.5: # Add some lights
            for _ in range(int(5 + energy * 15)):
                 lx,ly = random.randint(0,self.width),random.randint(0, int(self.height*0.7))
                 lr = int(1+energy*3)
                 draw.ellipse([(lx-lr,ly-lr),(lx+lr,ly+lr)], fill=ImageColor.getrgb(palette[0]))


    def _draw_celestial_inspired(self, draw: ImageDraw.ImageDraw, palette: List[str], energy: float, token: Optional[str]=None): # As before
        pass 
    def _draw_elemental_inspired(self, draw: ImageDraw.ImageDraw, palette: List[str], energy: float, token: Optional[str]=None): # As before
        pass
    def _draw_abstract_mood_patterns(self, draw: ImageDraw.ImageDraw, palette: List[str], mood: str, energy: float):
        # ... (More varied abstract patterns: e.g. circles for calm, sharp lines for angry/energetic)
        num_shapes = int(10 + energy * 30)
        for _ in range(num_shapes):
            x1,y1 = random.randint(0,self.width), random.randint(0,self.height)
            size = int(self.width * (0.05 + random.random()*0.15 + energy*0.1))
            color = random.choice(palette)
            if mood in ['happy', 'energetic', 'strong']:
                x2,y2 = x1+random.randint(-size,size), y1+random.randint(-size,size)
                draw.line([(x1,y1),(x2,y2)], fill=ImageColor.getrgb(color), width=1+int(energy*3))
            elif mood in ['calm', 'reflective', 'hope']:
                r = size // 2
                draw.ellipse([(x1-r,y1-r),(x1+r,y1+r)], outline=ImageColor.getrgb(color), width=1+int(energy*2))
            else: # sad, fear, angry - more chaotic or sharp
                points = [(x1,y1)]
                for _p in range(2 + int(energy*3)): points.append((x1+random.randint(-size,size), y1+random.randint(-size,size)))
                draw.polygon(points, fill=ImageColor.getrgb(color) if energy > 0.4 else None, outline=ImageColor.getrgb(color) if energy <=0.4 else None)


    def _add_text_overlay_styled(self, draw: ImageDraw.ImageDraw, text: str, palette: List[str], mood: str):
        if not text: return
        # Choose contrasting text color
        try:
            bg_avg_brightness = sum(ImageColor.getrgb(palette[0])[i] * ImageColor.getrgb(palette[1])[i] for i in range(3)) / (2*3*255) # Rough estimate
        except: bg_avg_brightness = 0.5

        text_color = "#FFFFFF" if bg_avg_brightness < 0.5 else "#000000"
        if mood in ['happy', 'energetic', 'romantic'] and len(palette) > 2 : text_color = palette[2] # Use a palette accent
        
        # Wrap text
        max_width_chars = self.width // (self.small_font.getlength("A") if self.small_font else 8) # Approx chars per line
        wrapped_text = ""
        current_line_len = 0
        for word in text.split():
            if current_line_len + len(word) + 1 > max_width_chars:
                wrapped_text += "\n" + word + " "
                current_line_len = len(word) + 1
            else:
                wrapped_text += word + " "
                current_line_len += len(word) + 1
        wrapped_text = wrapped_text.strip()

        try:
            # Get text bounding box to center it (Pillow 9.2.0+ for anchor, older use getsize/bbox)
            # text_bbox = draw.textbbox((0,0), wrapped_text, font=self.small_font, anchor="mm") # Pillow 9.2+
            # text_width = text_bbox[2] - text_bbox[0]
            # text_height = text_bbox[3] - text_bbox[1]
            # Pillow < 9.2.0 way:
            text_width = self.small_font.getlength(wrapped_text.split('\n')[0]) # width of longest line approx
            text_height = self.small_font_size * wrapped_text.count('\n') + self.small_font_size

            x = (self.width - text_width) / 2
            y = (self.height - text_height) / 2 # Center text block vertically
            
            draw.text((x, y), wrapped_text, font=self.small_font, fill=text_color, align="center")
        except Exception as e_text:
            logger.warning(f"Error drawing text overlay: {e_text}")
            draw.text((10, self.height - 30), text[:50], font=self.small_font, fill=text_color) # Fallback


class StoryboardGenerator(ttk.Frame):
    def __init__(self, parent: ttk.Widget, host_app_ref: Any):
        super().__init__(parent)
        self.root_app_tk = parent.winfo_toplevel()
        self.host_app = host_app_ref
        self.audio_metadata: Optional[Any] = None 
        self.lyrics_data: List[Tuple[float, str]] = []
        self.scenes_data: List[Scene] = []
        # self.pil_images_generated: List[Image.Image] = [] # Stored in Scene.thumbnail_pil now

        self.lyric_analyzer = LyricAnalyzer()
        # Ensure visual_generator uses sizes appropriate for thumbnails displayed in UI
        self.thumb_width, self.thumb_height = 240, 135 # 16:9 aspect ratio
        self.visual_generator = VisualGenerator(width=self.thumb_width, height=self.thumb_height) 

        self._create_ui_widgets()
        if self.host_app:
            self.host_app.subscribe_to_event("track_fully_loaded_with_details", self.on_host_new_track_event)
            current_meta = self.host_app.get_current_track_metadata()
            current_lyrics = self.host_app.get_current_lyrics_data()
            if current_meta: self.on_host_new_track_event(current_meta, current_lyrics)

    def _create_ui_widgets(self):
        self.pack(fill=tk.BOTH, expand=True) # Main frame packs itself

        # Top control frame
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill=tk.X)
        self.generate_button = ttk.Button(control_frame, text="Generate Storyboard", command=self.generate_storyboard_async, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=5)
        self.export_images_button = ttk.Button(control_frame, text="Export Images", command=self.export_images, state=tk.DISABLED)
        self.export_images_button.pack(side=tk.LEFT, padx=5)
        self.export_video_button = ttk.Button(control_frame, text="Export Video", command=self.export_video, state=tk.DISABLED)
        self.export_video_button.pack(side=tk.LEFT, padx=5)
        
        self.progress_bar = ttk.Progressbar(control_frame, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.status_label = ttk.Label(self, text="Load a track with lyrics to begin.", padding=5)
        self.status_label.pack(fill=tk.X)

        # Canvas for scrollable storyboard frames
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, borderwidth=0)
        self.storyboard_frame = ttk.Frame(self.canvas) # This frame holds the scene items
        self.scrollbar_y = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview) # Scrollbar for rows
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X) # Place x scrollbar under canvas
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.storyboard_frame, anchor="nw")

        self.storyboard_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_canvas_resize) # Adjust inner frame width
        # Mouse wheel scrolling for canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel) # Windows/some Linux
        self.canvas.bind_all("<Button-4>", self._on_mousewheel) # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel) # Linux scroll down


    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width) # Make inner frame width of canvas

    def _on_mousewheel(self, event):
        # Determine scroll direction (platform-dependent for delta)
        if event.num == 5 or event.delta < 0: # Scroll down
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta > 0: # Scroll up
            self.canvas.yview_scroll(-1, "units")


    def on_host_new_track_event(self, track_metadata: Any, lyrics_data_for_track: Optional[List[Tuple[float, str]]]):
        self.audio_metadata = track_metadata
        self.lyrics_data = lyrics_data_for_track or []
        self._clear_storyboard_display(); self.scenes_data.clear()
        
        if self.audio_metadata and self.lyrics_data:
            self.status_label.config(text=f"Track: {self.audio_metadata.title}. Ready to generate.")
            self.generate_button.config(state=tk.NORMAL)
        elif self.audio_metadata:
            self.status_label.config(text=f"Track: {self.audio_metadata.title} (No lyrics). Storyboard needs lyrics.")
            self.generate_button.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="No track loaded or no lyrics."); self.generate_button.config(state=tk.DISABLED)
        self.export_images_button.config(state=tk.DISABLED); self.export_video_button.config(state=tk.DISABLED)


    def generate_storyboard_async(self):
        if not self.lyrics_data or not self.audio_metadata:
            messagebox.showerror("Error", "Lyrics or audio metadata missing.", parent=self.root_app_tk); return
        self._clear_storyboard_display(); self.scenes_data.clear()
        self.status_label.config(text="Generating storyboard..."); self.generate_button.config(state=tk.DISABLED)
        self.export_video_button.config(state=tk.DISABLED); self.export_images_button.config(state=tk.DISABLED)
        self.progress_bar.config(value=0, maximum=len(self.lyrics_data))
        STORYBOARD_THREAD_POOL.submit(self._generation_worker)

    def _generation_worker(self):
        temp_scenes: List[Scene] = []
        for i, (timestamp, line_text) in enumerate(self.lyrics_data):
            if self.root_app_tk.winfo_exists() == 0: return 
            analysis = self.lyric_analyzer.analyze_line(line_text)
            next_ts = self.lyrics_data[i+1][0] if i+1 < len(self.lyrics_data) else self.audio_metadata.duration
            duration = max(0.5, next_ts - timestamp) # Min duration
            palette = self.visual_generator.palettes.get(analysis['mood'], self.visual_generator.palettes['neutral'])
            scene = Scene(timestamp,duration,line_text,analysis['mood'],list(palette),analysis['visual_elements'],'fade',analysis['energy'])
            temp_scenes.append(scene)
            if self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, self.progress_bar.config, {'value': i + 1})
        
        if self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, self.status_label.config, {'text': "Rendering scene images..."})
        if self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, self.progress_bar.config, {'value': 0, 'maximum': len(temp_scenes)})

        img_futures = {fut: i for i, fut in enumerate([STORYBOARD_THREAD_POOL.submit(self.visual_generator.generate_scene_image, s_data) for s_data in temp_scenes])}
        
        generated_pil_images = [None] * len(temp_scenes)
        for fut_done_idx, future_done in enumerate(concurrent.futures.as_completed(img_futures.keys())):
            original_idx = img_futures[future_done]
            try:
                img = future_done.result(timeout=15)
                if img: temp_scenes[original_idx].thumbnail_pil = img; generated_pil_images[original_idx] = img
            except Exception as e_img: logger.error(f"Error generating image for scene '{temp_scenes[original_idx].text}': {e_img}", exc_info=True)
            if self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, self.progress_bar.config, {'value': fut_done_idx + 1})
        
        self.scenes_data = temp_scenes # Now scenes have .thumbnail_pil populated
        
        if self.root_app_tk.winfo_exists():
            self.root_app_tk.after(0, self._display_storyboard_from_scenes_data)
            self.root_app_tk.after(0, self.status_label.config, {'text': f"Storyboard generated ({len(self.scenes_data)} scenes)."})
            self.root_app_tk.after(0, self.generate_button.config, {'state': tk.NORMAL})
            self.root_app_tk.after(0, self.export_video_button.config, {'state': tk.NORMAL if self.scenes_data else tk.DISABLED})
            self.root_app_tk.after(0, self.export_images_button.config, {'state': tk.NORMAL if self.scenes_data else tk.DISABLED})

    def _clear_storyboard_display(self):
        for widget in self.storyboard_frame.winfo_children(): widget.destroy()
        self.storyboard_frame.update_idletasks()
        self.canvas.config(scrollregion=(0,0,0,0))

    def _display_storyboard_from_scenes_data(self): # Renamed from _display_storyboard_from_pil
        self._clear_storyboard_display()
        col_count = 0; max_cols = max(1, int(self.canvas.winfo_width() / (self.thumb_width + 15)))
        current_row_frame: Optional[ttk.Frame] = None
        for i, scene_obj in enumerate(self.scenes_data):
            if scene_obj.thumbnail_pil is None: continue
            if col_count % max_cols == 0: current_row_frame = ttk.Frame(self.storyboard_frame); current_row_frame.pack(fill=tk.X)
            
            item_frame = ttk.Frame(current_row_frame, relief=tk.SOLID, borderwidth=1); item_frame.pack(side=tk.LEFT, padx=5, pady=5)
            if scene_obj.thumbnail_tk is None: scene_obj.thumbnail_tk = ImageTk.PhotoImage(scene_obj.thumbnail_pil)
            ttk.Label(item_frame, image=scene_obj.thumbnail_tk).pack()
            text_disp = scene_obj.text[:25]+"..." if len(scene_obj.text)>25 else scene_obj.text
            ttk.Label(item_frame, text=f"{self._format_time(scene_obj.timestamp)}|{scene_obj.mood}\n{text_disp}", justify=tk.LEFT, wraplength=self.thumb_width-10, font=("Arial",8)).pack(fill=tk.X)
            col_count +=1
        self.storyboard_frame.update_idletasks(); self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def export_video(self):
        if not self.scenes_data or not any(s.thumbnail_pil for s in self.scenes_data): messagebox.showerror("Error","No storyboard images generated.",parent=self.root_app_tk); return
        if not self.host_app or not self.host_app.export_manager_ref or not self.audio_metadata or not self.audio_metadata.file_path:
            messagebox.showerror("Error","Export manager or audio track unavailable.",parent=self.root_app_tk); return
        
        output_path = filedialog.asksaveasfilename(title="Save Storyboard Video", defaultextension=".mp4", filetypes=(("MP4 Video","*.mp4"),("All files","*.*")), parent=self.root_app_tk)
        if not output_path: return

        pil_images = [s.thumbnail_pil for s in self.scenes_data if s.thumbnail_pil]
        if not pil_images: messagebox.showerror("Error","No valid images in storyboard.", parent=self.root_app_tk); return

        # Prepare viz_config_dict for storyboard export type
        viz_config_for_export = {
            'type': 'storyboard_frames', # Special type for ExportManager
            'frames_pil': pil_images, 
            'scenes_data': [s.to_dict() for s in self.scenes_data], # Pass durations via scene data
            'width': self.thumb_width, # Dimensions of the frames
            'height': self.thumb_height,
            'fps': 25 # Default FPS for storyboard video, can be made configurable
        }
        self.status_label.config(text=f"Exporting storyboard video to {Path(output_path).name}...")
        self.export_video_button.config(state=tk.DISABLED)
        
        # ExportManager will handle this in its worker thread (using THREAD_POOL is fine for this type)
        future = self.host_app.export_manager_ref.export_visualization_async(
            audio_file_path=self.audio_metadata.file_path,
            output_video_path=output_path,
            viz_engine_config_dict=viz_config_for_export, # Contains frames and scene data
            viz_type_name='storyboard_frames', # Signal to worker
            video_format_name="MP4 (H.264/AAC)", # Default format
            duration_to_export=self.audio_metadata.duration # Use full audio duration
        )
        # Optional: Monitor future if needed, or rely on ExportManager's progress callback
        def _on_export_done(f):
            if self.root_app_tk.winfo_exists(): # Check if UI still around
                self.export_video_button.config(state=tk.NORMAL)
                try:
                    if f.result(): self.status_label.config(text=f"Storyboard video exported successfully.")
                    else: self.status_label.config(text="Storyboard video export failed or cancelled.")
                except Exception as e_exp: self.status_label.config(text=f"Export error: {e_exp}")
        if future: future.add_done_callback(_on_export_done)


    def export_images(self):
        if not self.scenes_data or not any(s.thumbnail_pil for s in self.scenes_data): messagebox.showerror("Error","No storyboard images.",parent=self.root_app_tk); return
        folder_selected = filedialog.askdirectory(title="Select Folder to Save Storyboard Images", parent=self.root_app_tk)
        if not folder_selected: return
        
        self.status_label.config(text="Exporting images...")
        self.export_images_button.config(state=tk.DISABLED)
        
        def _export_worker():
            path_obj = Path(folder_selected)
            base_filename = Path(self.audio_metadata.file_path).stem if self.audio_metadata and self.audio_metadata.file_path else "scene"
            errors = 0
            for i, scene_obj in enumerate(self.scenes_data):
                if scene_obj.thumbnail_pil:
                    try: scene_obj.thumbnail_pil.save(path_obj / f"{base_filename}_{i+1:03d}.png")
                    except Exception as e: logger.error(f"Error saving scene image {i+1}: {e}"); errors+=1
                if self.root_app_tk.winfo_exists() == 0: break # Stop if main window closed
            
            def _ui_update():
                self.export_images_button.config(state=tk.NORMAL)
                if errors == 0: messagebox.showinfo("Success", f"Exported {len(self.scenes_data)-errors} images to:\n{folder_selected}",parent=self.root_app_tk)
                else: messagebox.showwarning("Partial Success", f"Exported {len(self.scenes_data)-errors} images with {errors} errors.",parent=self.root_app_tk)
                self.status_label.config(text=f"Image export finished. {errors} errors." if errors else "Image export complete.")

            if self.root_app_tk.winfo_exists(): self.root_app_tk.after(0, _ui_update)

        STORYBOARD_THREAD_POOL.submit(_export_worker)


    def _format_time(self, seconds_float: float) -> str:
        if seconds_float is None: return "00:00"
        m, s = divmod(int(seconds_float), 60)
        return f"{m:02d}:{s:02d}"

    def on_app_exit(self):
        logger.info("StoryboardGenerator UI on_app_exit called.")
        if self.host_app: self.host_app.unsubscribe_from_event("track_fully_loaded_with_details", self.on_host_new_track_event)
        # Cancel any running generation (worker threads should check a stop event or root_app_tk.winfo_exists())
        # For ProcessPool, it's harder to cancel mid-task, rely on pool shutdown by launcher.


def create_storyboard_tab(notebook: ttk.Notebook, host_app_ref: Any) -> StoryboardGenerator:
    storyboard_frame = ttk.Frame(notebook)
    notebook.add(storyboard_frame, text="Storyboard")
    storyboard_ui_instance = StoryboardGenerator(storyboard_frame, host_app_ref=host_app_ref)
    # StoryboardGenerator packs itself in its __init__
    logger.info("Storyboard Tab UI created.")
    return storyboard_ui_instance

# if __name__ == "__main__": test block as before


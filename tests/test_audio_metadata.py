import os
import sys
import ast
import types

# Dynamically load only the AudioMetadata dataclass from flow_state_main.py
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(repo_root, 'flow_state_main.py')
with open(file_path, 'r') as f:
    source = f.read()

module = types.ModuleType('flow_state_main')

# Parse the source to find the AudioMetadata class definition
tree = ast.parse(source)
lines = source.splitlines()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'AudioMetadata':
        start = node.decorator_list[0].lineno - 1 if node.decorator_list else node.lineno - 1
        snippet = "\n".join(lines[start: node.end_lineno])
        break
else:
    raise RuntimeError('AudioMetadata class not found')

exec('from dataclasses import dataclass, asdict\nfrom typing import Optional, Any, Dict\n' + snippet, module.__dict__)
sys.modules['flow_state_main'] = module

from flow_state_main import AudioMetadata


def test_audio_metadata_to_dict():
    meta = AudioMetadata(
        title="Song",
        artist="Artist",
        album="Album",
        duration=3.5,
        sample_rate=48000,
        channels=1,
        file_path="song.mp3",
        id=1,
        genre="Pop",
        year="2021",
        track_number="1",
        disc_number="1",
        album_artist="Artist",
        bpm_tag="120",
        key_tag="C"
    )

    data = meta.to_dict()
    assert data == {
        'title': "Song",
        'artist': "Artist",
        'album': "Album",
        'duration': 3.5,
        'sample_rate': 48000,
        'channels': 1,
        'file_path': "song.mp3",
        'id': 1,
        'genre': "Pop",
        'year': "2021",
        'track_number': "1",
        'disc_number': "1",
        'album_artist': "Artist",
        'bpm_tag': "120",
        'key_tag': "C"
    }


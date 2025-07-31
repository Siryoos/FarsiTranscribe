"""
FarsiTranscribe - A modular Persian/Farsi audio transcription system.

A clean, efficient, and extensible audio transcription tool optimized for 
Farsi/Persian language using OpenAI's Whisper model.
"""

from .core import FarsiTranscriber
from .config import TranscriptionConfig, ConfigPresets
from .audio import AudioProcessor
from .utils import TranscriptionManager

__version__ = "2.0.0"
__author__ = "FarsiTranscribe Team"
__license__ = "MIT"

__all__ = [
    "FarsiTranscriber",
    "TranscriptionConfig", 
    "ConfigPresets",
    "AudioProcessor",
    "TranscriptionManager"
]
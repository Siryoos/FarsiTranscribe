"""
FarsiTranscribe - A modular Persian/Farsi audio transcription system.

A clean, efficient, and extensible audio transcription tool optimized for 
Farsi/Persian language using OpenAI's Whisper model.

Note: The legacy `farsi_transcribe` package is maintained for backward
compatibility. New development is consolidated under the `src` package.
Consider migrating to `src.core.UnifiedTranscriber` and
`src.core.config.ConfigFactory`.
"""

# Legacy API removed; provide thin re-exports to modern src API
from src.core import UnifiedTranscriber as FarsiTranscriber  # type: ignore
from src.core.config import (  # type: ignore
    TranscriptionConfig,
    ConfigFactory,
)
from src.preprocessing import UnifiedAudioPreprocessor as AudioProcessor  # type: ignore
from src.utils.file_manager import TranscriptionFileManager as TranscriptionManager  # type: ignore

__version__ = "2.0.0"
__author__ = "FarsiTranscribe Team"
__license__ = "MIT"

__all__ = [
    "FarsiTranscriber",
    "TranscriptionConfig", 
    "ConfigPresets",
    "AudioProcessor",
    "TranscriptionManager",
    "ConfigFactory",
]
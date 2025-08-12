"""
Unified public API for the modern `src` package.
"""

__version__ = "2.0.0"
__author__ = "FarsiTranscribe Team"

from .core.transcriber import UnifiedAudioTranscriber
from .core.config import TranscriptionConfig, ConfigFactory
from .utils.file_manager import TranscriptionFileManager
from .utils.repetition_detector import RepetitionDetector
from .utils.sentence_extractor import SentenceExtractor

__all__ = [
    "UnifiedAudioTranscriber",
    "TranscriptionConfig",
    "ConfigFactory",
    "TranscriptionFileManager",
    "RepetitionDetector",
    "SentenceExtractor",
]

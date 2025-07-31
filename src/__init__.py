"""
FarsiTranscribe - A modular audio transcription system with anti-repetition features.
"""

__version__ = "1.0.0"
__author__ = "FarsiTranscribe Team"

from .core.transcriber import UnifiedAudioTranscriber
from .core.config import TranscriptionConfig
from .utils.file_manager import TranscriptionFileManager
from .utils.repetition_detector import RepetitionDetector
from .utils.sentence_extractor import SentenceExtractor

__all__ = [
    "UnifiedAudioTranscriber",
    "TranscriptionConfig", 
    "TranscriptionFileManager",
    "RepetitionDetector",
    "SentenceExtractor"
] 
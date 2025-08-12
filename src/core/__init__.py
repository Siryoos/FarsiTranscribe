"""
Core transcription functionality for FarsiTranscribe.
"""

from .transcriber import UnifiedAudioTranscriber as UnifiedTranscriber
from .config import TranscriptionConfig

__all__ = [
    "UnifiedTranscriber",
    "TranscriptionConfig",
]

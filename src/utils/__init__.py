"""
Utility modules for transcription system.
Consolidated and deduplicated for better maintainability.
"""

# Core utilities
from .repetition_detector import RepetitionDetector
from .sentence_extractor import SentenceExtractor
from .file_manager import TranscriptionFileManager
from .chunk_calculator import ChunkCalculator, create_chunk_calculator

# Unified modules (consolidated functionality)
from ..preprocessing import (
    UnifiedAudioPreprocessor,
    create_unified_preprocessor,
    get_unified_preprocessing_capabilities,
)

from .unified_terminal_display import (
    UnifiedTerminalDisplay,
    EnhancedPreviewDisplay,
    SimplePreviewDisplay,
    create_preview_display,
    get_terminal_capabilities,
)

from .unified_memory_manager import (
    UnifiedMemoryManager,
    create_unified_memory_manager,
    performance_monitor,
)

# Specialized modules (kept separate due to specific functionality)
from .advanced_model_ensemble import (
    AdvancedModelEnsemble,
    create_advanced_ensemble,
)
from .speaker_diarization import SpeakerDiarizer, create_speaker_diarizer
from .quality_assessor import QualityAssessor, create_quality_assessor
from .persian_text_postprocessor import (
    PersianTextPostProcessor,
    create_persian_postprocessor,
)
from .preprocessing_validator import (
    PreprocessingValidator,
    validate_preprocessing,
)

# Backward compatibility aliases
# Audio preprocessing
AudioPreprocessor = UnifiedAudioPreprocessor
create_audio_preprocessor = create_unified_preprocessor
get_preprocessing_capabilities = get_unified_preprocessing_capabilities

# Terminal display
TerminalDisplay = UnifiedTerminalDisplay
create_terminal_display = create_preview_display

# Memory management
EnhancedMemoryManager = UnifiedMemoryManager
create_memory_manager = create_unified_memory_manager

__all__ = [
    # Core utilities
    "RepetitionDetector",
    "SentenceExtractor",
    "TranscriptionFileManager",
    "ChunkCalculator",
    "create_chunk_calculator",
    # Unified modules
    "UnifiedAudioPreprocessor",
    "UnifiedTerminalDisplay",
    "EnhancedPreviewDisplay",
    "SimplePreviewDisplay",
    "UnifiedMemoryManager",
    "create_unified_preprocessor",
    "create_preview_display",
    "create_unified_memory_manager",
    "get_unified_preprocessing_capabilities",
    "get_terminal_capabilities",
    "performance_monitor",
    # Specialized modules
    "AdvancedModelEnsemble",
    "SpeakerDiarizer",
    "QualityAssessor",
    "PersianTextPostProcessor",
    "PreprocessingValidator",
    "create_advanced_ensemble",
    "create_speaker_diarizer",
    "create_quality_assessor",
    "create_persian_postprocessor",
    "validate_preprocessing",
    # Backward compatibility
    "AudioPreprocessor",
    "TerminalDisplay",
    "EnhancedMemoryManager",
    "create_audio_preprocessor",
    "create_terminal_display",
    "create_preview_display",
    "create_memory_manager",
    "get_preprocessing_capabilities",
]

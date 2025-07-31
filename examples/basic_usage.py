#!/usr/bin/env python3
"""
Basic usage example for FarsiTranscribe.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import UnifiedAudioTranscriber
from src.core.config import ConfigFactory


def basic_transcription_example():
    """Basic transcription example."""
    
    # Audio file path (update this to your actual file)
    audio_file = "examples/audio/jalase bi va zirsakht.m4a"
    
    # Create configuration
    config = ConfigFactory.create_optimized_config(
        model_size="large-v3",
        language="fa",
        enable_preview=True,
        output_dir="./output"
    )
    
    # Run transcription
    with UnifiedAudioTranscriber(config) as transcriber:
        transcription = transcriber.transcribe_file(audio_file)
        
        print(f"Transcription completed: {len(transcription)} characters")
        print(f"Preview: {transcription[:200]}...")


def high_quality_transcription_example():
    """High quality transcription example."""
    
    audio_file = "examples/audio/jalase bi va zirsakht.m4a"
    
    # Use high quality preset
    config = ConfigFactory.create_high_quality_config()
    config.output_directory = "./high_quality_output"
    
    with UnifiedAudioTranscriber(config) as transcriber:
        transcription = transcriber.transcribe_file(audio_file)
        print("High quality transcription completed!")


def fast_transcription_example():
    """Fast transcription example."""
    
    audio_file = "examples/audio/jalase bi va zirsakht.m4a"
    
    # Use fast preset
    config = ConfigFactory.create_fast_config()
    config.output_directory = "./fast_output"
    
    with UnifiedAudioTranscriber(config) as transcriber:
        transcription = transcriber.transcribe_file(audio_file)
        print("Fast transcription completed!")


def custom_configuration_example():
    """Custom configuration example."""
    
    audio_file = "examples/audio/jalase bi va zirsakht.m4a"
    
    # Create custom configuration
    config = ConfigFactory.create_optimized_config()
    config.model_name = "large-v3"
    config.chunk_duration_ms = 15000  # 15 seconds
    config.overlap_ms = 500  # 500ms overlap
    config.repetition_threshold = 0.9  # Very strict repetition detection
    config.max_word_repetition = 1  # Allow only 1 repetition
    config.output_directory = "./custom_output"
    
    with UnifiedAudioTranscriber(config) as transcriber:
        transcription = transcriber.transcribe_file(audio_file)
        print("Custom transcription completed!")


if __name__ == "__main__":
    print("FarsiTranscribe Examples")
    print("=" * 30)
    print("The examples use the sample audio file: examples/audio/jalase bi va zirsakht.m4a")
    print("\nAvailable examples:")
    print("1. basic_transcription_example()")
    print("2. high_quality_transcription_example()")
    print("3. fast_transcription_example()")
    print("4. custom_configuration_example()")
    
    # Uncomment the example you want to run:
    # basic_transcription_example()
    # high_quality_transcription_example()
    # fast_transcription_example()
    # custom_configuration_example() 
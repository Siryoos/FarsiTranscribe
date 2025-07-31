#!/usr/bin/env python3
"""
Basic usage examples for FarsiTranscribe.

This script demonstrates various ways to use the FarsiTranscribe library
for Persian/Farsi audio transcription.
"""

from pathlib import Path
import sys

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from farsi_transcribe import FarsiTranscriber, TranscriptionConfig, ConfigPresets
from farsi_transcribe.utils import TranscriptionManager


def example_basic_transcription():
    """Basic transcription with default settings."""
    print("Example 1: Basic Transcription")
    print("-" * 40)
    
    # Create transcriber with balanced preset
    config = ConfigPresets.balanced()
    transcriber = FarsiTranscriber(config)
    
    # Transcribe audio file
    audio_file = Path("examples/audio/sample.mp3")
    if audio_file.exists():
        result = transcriber.transcribe_file(audio_file)
        
        # Print results
        print(f"Duration: {result.duration:.1f}s")
        print(f"Processing time: {result.processing_time:.1f}s")
        print(f"Text preview: {result.text[:200]}...")
        
        # Save results
        result.save_text("output/sample_transcription.txt")
    else:
        print(f"Audio file not found: {audio_file}")


def example_persian_optimized():
    """Persian-optimized transcription with high quality."""
    print("\nExample 2: Persian-Optimized Transcription")
    print("-" * 40)
    
    # Use Persian-optimized preset
    config = ConfigPresets.persian_optimized()
    
    # Customize some settings
    config.output_directory = Path("output/persian")
    config.remove_diacritics = True
    
    with FarsiTranscriber(config) as transcriber:
        audio_file = Path("examples/audio/farsi_speech.mp3")
        if audio_file.exists():
            result = transcriber.transcribe_file(audio_file)
            
            # Use TranscriptionManager for organized output
            manager = TranscriptionManager(config.output_directory)
            saved_files = manager.save_result(
                result,
                "farsi_speech",
                ["txt", "json", "segments"]
            )
            
            print(f"Saved files:")
            for format_type, path in saved_files.items():
                print(f"  - {format_type}: {path}")


def example_memory_efficient():
    """Memory-efficient transcription for large files."""
    print("\nExample 3: Memory-Efficient Transcription")
    print("-" * 40)
    
    # Use memory-efficient preset
    config = ConfigPresets.memory_efficient()
    config.clear_cache_every = 5  # Clear cache every 5 chunks
    
    transcriber = FarsiTranscriber(config)
    
    # Use streaming for large files
    large_audio = Path("examples/audio/long_podcast.mp3")
    if large_audio.exists():
        result = transcriber.transcribe_stream(large_audio)
        print(f"Transcribed {result.duration/60:.1f} minutes of audio")
    else:
        print(f"Large audio file not found: {large_audio}")


def example_custom_configuration():
    """Custom configuration example."""
    print("\nExample 4: Custom Configuration")
    print("-" * 40)
    
    # Create custom configuration
    config = TranscriptionConfig(
        model_name="medium",
        language="fa",
        chunk_duration=45,
        overlap=5,
        device="cpu",  # Force CPU
        persian_normalization=True,
        output_formats=["txt", "json"],
        max_memory_gb=3.0
    )
    
    print(f"Custom config:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Chunk duration: {config.chunk_duration}s")
    
    # Use custom config
    transcriber = FarsiTranscriber(config)
    # ... transcribe files


def example_with_extension():
    """Example using custom extension."""
    print("\nExample 5: Using Extensions")
    print("-" * 40)
    
    # Define a simple extension
    class LoggingExtension:
        """Extension that logs chunk processing."""
        
        def install(self, transcriber):
            transcriber.hooks.add_pre_chunk_hook(self.log_chunk_start)
            transcriber.hooks.add_post_chunk_hook(self.log_chunk_end)
        
        def log_chunk_start(self, chunk):
            print(f"  Processing chunk {chunk.index} ({chunk.duration:.1f}s)")
            return chunk
        
        def log_chunk_end(self, result):
            words = len(result['text'].split())
            print(f"  Chunk {result['index']} completed: {words} words")
            return result
    
    # Create transcriber and add extension
    config = ConfigPresets.fast()
    transcriber = FarsiTranscriber(config)
    transcriber.add_extension(LoggingExtension())
    
    # Now transcription will include logging
    audio_file = Path("examples/audio/sample.mp3")
    if audio_file.exists():
        result = transcriber.transcribe_file(audio_file)
        print(f"Total words: {len(result.text.split())}")


def example_batch_processing():
    """Batch processing multiple files."""
    print("\nExample 6: Batch Processing")
    print("-" * 40)
    
    # Setup for batch processing
    config = ConfigPresets.balanced()
    config.output_directory = Path("output/batch")
    
    audio_dir = Path("examples/audio")
    
    with FarsiTranscriber(config) as transcriber:
        manager = TranscriptionManager(config.output_directory)
        
        # Process all MP3 files
        for audio_file in audio_dir.glob("*.mp3"):
            print(f"\nProcessing: {audio_file.name}")
            
            try:
                result = transcriber.transcribe_file(audio_file)
                
                # Save with same base name
                base_name = audio_file.stem
                saved_files = manager.save_result(
                    result,
                    base_name,
                    ["txt", "json"]
                )
                
                # Save summary
                manager.save_summary(result, base_name)
                
                print(f"  ✓ Completed: {len(result.text)} characters")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")


def main():
    """Run all examples."""
    print("FarsiTranscribe - Usage Examples")
    print("=" * 60)
    
    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)
    
    # Run examples (comment out any you don't want to run)
    example_basic_transcription()
    example_persian_optimized()
    example_memory_efficient()
    example_custom_configuration()
    example_with_extension()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("Examples completed!")


if __name__ == "__main__":
    main() 
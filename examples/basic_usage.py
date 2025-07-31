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
    """
    Performs basic transcription of a sample MP3 audio file using default balanced settings.
    
    If the sample audio file exists, transcribes it, prints summary information, and saves the transcription text to an output file. If the file is missing, prints a notification.
    """
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
    """
    Performs high-quality transcription of a Persian audio file using a Persian-optimized configuration and saves the results in multiple formats.
    
    This example demonstrates customizing output directory and diacritics removal, transcribing a sample Farsi speech file, and saving the transcription as text, JSON, and segment files. Prints the paths of the saved files.
    """
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
    """
    Demonstrates memory-efficient streaming transcription of a large audio file using the FarsiTranscribe library.
    
    Uses a memory-optimized configuration with periodic cache clearing to transcribe a long audio file in streaming mode. Prints the duration of transcribed audio in minutes or notifies if the file is missing.
    """
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
    """
    Demonstrates how to create and use a custom transcription configuration with specific parameters for Farsi audio transcription.
    
    This example sets up a `TranscriptionConfig` with a medium model, Persian language, custom chunking, CPU device, normalization, output formats, and memory limit, then instantiates a transcriber with this configuration.
    """
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
    """
    Demonstrates how to use a custom extension with FarsiTranscriber to log chunk processing during transcription.
    
    Defines a `LoggingExtension` that logs the start and end of each audio chunk processed. Instantiates a transcriber with a fast preset, adds the extension, and transcribes a sample audio file if present, printing the total word count of the transcription.
    """
    print("\nExample 5: Using Extensions")
    print("-" * 40)
    
    # Define a simple extension
    class LoggingExtension:
        """Extension that logs chunk processing."""
        
        def install(self, transcriber):
            """
            Installs logging hooks on the transcriber to log information before and after processing each audio chunk.
            
            Parameters:
                transcriber: The transcriber instance to which the logging hooks will be attached.
            """
            transcriber.hooks.add_pre_chunk_hook(self.log_chunk_start)
            transcriber.hooks.add_post_chunk_hook(self.log_chunk_end)
        
        def log_chunk_start(self, chunk):
            """
            Logs the start of processing for a transcription chunk, displaying its index and duration.
            
            Returns:
                chunk: The same chunk object that was passed in.
            """
            print(f"  Processing chunk {chunk.index} ({chunk.duration:.1f}s)")
            return chunk
        
        def log_chunk_end(self, result):
            """
            Logs the completion of a transcription chunk and its word count.
            
            Parameters:
                result (dict): A dictionary containing the chunk's transcription result, including 'text' and 'index' keys.
            
            Returns:
                dict: The original result dictionary.
            """
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
    """
    Transcribes all MP3 audio files in a specified directory and saves the results in multiple formats.
    
    For each audio file, attempts transcription using a balanced preset configuration, saves the output as text and JSON, and generates a summary. Errors encountered during processing are reported for each file.
    """
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
    """
    Runs all FarsiTranscribe usage example functions sequentially, ensuring the output directory exists and printing progress messages.
    """
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
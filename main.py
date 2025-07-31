#!/usr/bin/env python3
"""
FarsiTranscribe - Main application entry point.
A modular audio transcription system with anti-repetition features.
"""

import argparse
import sys
import os
from pathlib import Path

from src import UnifiedAudioTranscriber
from src.core.config import TranscriptionConfig, ConfigFactory


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="FarsiTranscribe - Enhanced audio transcription with anti-repetition features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a
  
  # High quality transcription
  python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a --quality high
  
  # Fast transcription
  python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a --quality fast
  
  # CPU-optimized transcription (recommended for CPU-only systems)
  python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a --quality cpu-optimized
  
  # Custom configuration
  python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a --model large-v3 --language fa --output-dir ./output
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to transcribe"
    )
    
    parser.add_argument(
        "--quality", "-q",
        choices=["fast", "balanced", "high", "cpu-optimized"],
        default="balanced",
        help="Transcription quality preset (default: balanced)"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model to use (overrides quality preset)"
    )
    
    parser.add_argument(
        "--language", "-l",
        default="fa",
        help="Language code for transcription (default: fa for Persian)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Output directory for transcription files (default: ./output)"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable sentence preview during transcription"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for transcription (default: auto)"
    )
    
    parser.add_argument(
        "--chunk-duration",
        type=int,
        help="Chunk duration in milliseconds (default: based on quality preset)"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        help="Overlap between chunks in milliseconds (default: based on quality preset)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing (default: auto-detected)"
    )
    
    parser.add_argument(
        "--repetition-threshold",
        type=float,
        help="Repetition detection threshold (0.0-1.0, default: based on quality preset)"
    )
    
    parser.add_argument(
        "--max-word-repetition",
        type=int,
        help="Maximum allowed word repetitions (default: based on quality preset)"
    )
    
    return parser


def get_config_from_args(args: argparse.Namespace) -> TranscriptionConfig:
    """Create configuration from command line arguments."""
    
    # Start with quality preset
    if args.quality == "fast":
        config = ConfigFactory.create_fast_config()
    elif args.quality == "high":
        config = ConfigFactory.create_high_quality_config()
    elif args.quality == "cpu-optimized":
        config = ConfigFactory.create_cpu_optimized_config()
    else:  # balanced
        config = ConfigFactory.create_optimized_config()
    
    # Override with command line arguments
    if args.model:
        config.model_name = args.model
    
    if args.language:
        config.language = args.language
    
    if args.output_dir:
        config.output_directory = args.output_dir
    
    if args.no_preview:
        config.enable_sentence_preview = False
    
    if args.device != "auto":
        config.device = args.device
    
    if args.chunk_duration:
        config.chunk_duration_ms = args.chunk_duration
    
    if args.overlap:
        config.overlap_ms = args.overlap
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.repetition_threshold:
        config.repetition_threshold = args.repetition_threshold
    
    if args.max_word_repetition:
        config.max_word_repetition = args.max_word_repetition
    
    return config


def validate_audio_file(audio_file: str) -> bool:
    """Validate that the audio file exists and is accessible."""
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file not found: {audio_file}")
        return False
    
    if not os.path.isfile(audio_file):
        print(f"❌ Error: Path is not a file: {audio_file}")
        return False
    
    # Check file size
    file_size = os.path.getsize(audio_file)
    if file_size == 0:
        print(f"❌ Error: Audio file is empty: {audio_file}")
        return False
    
    # Check if file is readable
    try:
        with open(audio_file, 'rb') as f:
            f.read(1024)  # Read first 1KB to check if file is readable
    except Exception as e:
        print(f"❌ Error: Cannot read audio file: {e}")
        return False
    
    return True


def main():
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate audio file
    if not validate_audio_file(args.audio_file):
        sys.exit(1)
    
    # Create configuration
    try:
        config = get_config_from_args(args)
    except Exception as e:
        print(f"❌ Error creating configuration: {e}")
        sys.exit(1)
    
    # Display configuration
    print("🎙️  FarsiTranscribe Configuration")
    print("=" * 50)
    print(f"📁 Audio File: {args.audio_file}")
    print(f"🔧 Model: {config.model_name}")
    print(f"🌍 Language: {config.language}")
    print(f"💾 Output Directory: {config.output_directory}")
    print(f"⚡ Device: {config.device}")
    print(f"📊 Quality Preset: {args.quality}")
    print(f"🔍 Preview: {'Enabled' if config.enable_sentence_preview else 'Disabled'}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(config.output_directory, exist_ok=True)
    
    # Run transcription
    try:
        with UnifiedAudioTranscriber(config) as transcriber:
            transcription = transcriber.transcribe_file(args.audio_file)
            
            if transcription:
                print("\n📝 Transcription completed successfully!")
                print(f"📄 Output files saved to: {config.output_directory}")
                
                # Show preview of transcription
                preview_length = 200
                if len(transcription) > preview_length:
                    preview = transcription[:preview_length] + "..."
                else:
                    preview = transcription
                
                print(f"\n📖 Preview ({len(transcription)} characters):")
                print("-" * 40)
                print(preview)
                print("-" * 40)
            else:
                print("⚠️  Transcription completed but no text was generated.")
                
    except KeyboardInterrupt:
        print("\n❌ Transcription interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Cleaned up main.py with unified transcriber and simplified options.
"""

import argparse
import sys
import os
from pathlib import Path

from src import UnifiedAudioTranscriber
from src.core.config import TranscriptionConfig, ConfigFactory


def create_parser() -> argparse.ArgumentParser:
    """Create simplified argument parser."""
    parser = argparse.ArgumentParser(
        description="FarsiTranscribe - Optimized audio transcription"
    )
    
    parser.add_argument("audio_file", help="Audio file path")
    parser.add_argument("--quality", "-q", 
                       choices=["fast", "balanced", "high", "memory-optimized"], 
                       default="memory-optimized",
                       help="Quality preset")
    parser.add_argument("--model", "-m", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model (overrides quality preset)")
    parser.add_argument("--output-dir", "-o", default="./output",
                       help="Output directory")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable transcription preview")
    
    return parser


def get_config(args: argparse.Namespace) -> TranscriptionConfig:
    """Get configuration from arguments."""
    config_map = {
        "fast": ConfigFactory.create_fast_config,
        "balanced": ConfigFactory.create_optimized_config,
        "high": ConfigFactory.create_high_quality_config,
        "memory-optimized": ConfigFactory.create_memory_optimized_config
    }
    
    config = config_map[args.quality]()
    
    # Apply overrides
    if args.model:
        config.model_name = args.model
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.no_preview:
        config.enable_sentence_preview = False
        
    return config


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Create configuration
    config = get_config(args)
    os.makedirs(config.output_directory, exist_ok=True)
    
    # Display configuration
    print(f"🎙️  FarsiTranscribe - Unified Mode")
    print("=" * 40)
    print(f"📁 Audio: {args.audio_file}")
    print(f"🔧 Model: {config.model_name}")
    print(f"📊 Quality: {args.quality}")
    print(f"💾 Output: {config.output_directory}")
    print("=" * 40)
    
    # Run transcription
    try:
        with UnifiedAudioTranscriber(config) as transcriber:
            transcription = transcriber.transcribe_file(args.audio_file)
            
            if transcription:
                print(f"\n✅ Success! {len(transcription)} characters transcribed")
            else:
                print("⚠️  No transcription generated")
                
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

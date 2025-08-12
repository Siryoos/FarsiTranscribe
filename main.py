#!/usr/bin/env python3
"""
Cleaned up main.py with unified transcriber and simplified options.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Callable

from src import UnifiedAudioTranscriber
from src.core.config import TranscriptionConfig, ConfigFactory


def create_parser() -> argparse.ArgumentParser:
    """Create simplified argument parser."""
    parser = argparse.ArgumentParser(
        description="FarsiTranscribe - Optimized audio transcription"
    )

    parser.add_argument("audio_file", help="Audio file path")
    parser.add_argument(
        "--quality",
        "-q",
        choices=["fast", "balanced", "high", "memory-optimized", "95-percent"],
        default="memory-optimized",
        help="Quality preset (95-percent for maximum quality)",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=[
            "nezamisafa/whisper-persian-v4",
            "tiny",
            "base",
            "small",
            "medium",
            "large",
        ],
        default="nezamisafa/whisper-persian-v4",
        help="Whisper model (default: Persian fine-tuned model)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="./output", help="Output directory"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable transcription preview",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode (useful when CUDA has compatibility issues)",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization (use standard chunked transcription)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Specify exact number of speakers for diarization",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to detect",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to detect",
    )

    return parser


def get_config(args: argparse.Namespace) -> TranscriptionConfig:
    """Get configuration from arguments."""
    
    config_map: Dict[str, Callable[[], TranscriptionConfig]] = {
        "fast": ConfigFactory.create_fast_config,
        "balanced": ConfigFactory.create_optimized_config,
        "high": ConfigFactory.create_high_quality_config,
        "memory-optimized": ConfigFactory.create_memory_optimized_config,
        "95-percent": ConfigFactory.create_95_percent_quality_config,
    }

    config_func = config_map.get(args.quality)
    if config_func is None:
        raise ValueError(f"Unknown quality preset: {args.quality}")
    
    config = config_func()

    # Apply overrides
    if args.model:
        config.model_name = args.model
        # Set Hugging Face flag for Persian model
        if args.model == "nezamisafa/whisper-persian-v4":
            config.use_huggingface_model = True
        else:
            config.use_huggingface_model = False
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.no_preview:
        config.enable_sentence_preview = False
    if args.force_cpu:
        config.device = "cpu"
        config._apply_cpu_optimizations()
    if args.no_diarization:
        config.enable_speaker_diarization = False
    
    # Store diarization parameters for use in the transcriber
    config.diarization_params = {
        "num_speakers": args.num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
    }

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
    print("🎙️  FarsiTranscribe - Unified Mode")
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
                print(
                    f"\n✅ Success! {len(transcription)} characters transcribed"
                )
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

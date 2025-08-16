"""
Modern CLI for FarsiTranscribe using the consolidated `src` architecture.
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Callable

from .config import TranscriptionConfig, ConfigFactory
from . import UnifiedTranscriber


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("farsi_transcribe.log", mode="w", encoding="utf-8"),
        ],
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="farsi-transcribe",
        description="FarsiTranscribe - Efficient Persian/Farsi Audio Transcription (modern src CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument(
        "--preset",
        "-p",
        choices=[
            "fast",
            "balanced",
            "high-quality",
            "memory-efficient",
            "persian-optimized",
            "gpu-optimized",
        ],
        default="balanced",
        help="Configuration preset (default: balanced)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model name or HF repo id (overrides preset)",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="fa",
        help="Language code (default: fa)",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        help="Chunk duration in seconds (overrides preset)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Overlap between chunks in seconds (overrides preset)",
    )
    parser.add_argument("--gpu", "--cuda", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> TranscriptionConfig:
    preset_map: Dict[str, Callable[[], TranscriptionConfig]] = {
        "fast": ConfigFactory.create_fast_config,
        "balanced": ConfigFactory.create_optimized_config,
        "high-quality": ConfigFactory.create_high_quality_config,
        "memory-efficient": ConfigFactory.create_memory_optimized_config,
        "persian-optimized": ConfigFactory.create_persian_optimized_config,
        "gpu-optimized": lambda: ConfigFactory.create_optimized_config(
            enable_preview=True
        ),
    }

    config = preset_map[args.preset]()

    if args.model:
        config.model_name = args.model
    if args.language:
        config.language = args.language
    if args.chunk_duration is not None:
        # Convert seconds to ms
        config.chunk_duration_ms = int(args.chunk_duration * 1000)
    if args.overlap is not None:
        config.overlap_ms = int(args.overlap * 1000)
    if args.gpu:
        config.device = "cuda"
    elif args.cpu:
        config.device = "cpu"
    if args.output_dir:
        config.output_directory = str(args.output_dir)

    return config


def main(argv: Optional[list] = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if not args.audio_file.exists():
        logging.getLogger(__name__).error(
            f"Audio file not found: {args.audio_file}"
        )
        return 1

    config = _config_from_args(args)

    # Banner (quiet-aware)
    if not args.quiet:
        print("\n🎙️  FarsiTranscribe (src) - Persian Audio Transcription")
        print("=" * 60)
        print(f"📁 Audio File: {args.audio_file.name}")
        size_mb = args.audio_file.stat().st_size / 1024 / 1024
        print(f"📊 File Size: {size_mb:.1f} MB")
        print(f"🔧 Model: {config.model_name}")
        print(f"🖥️  Device: {config.device.upper()}")
        print(
            f"⚡ Chunk Duration: {config.chunk_duration_ms/1000:.0f}s (overlap: {config.overlap_ms/1000:.0f}s)"
        )
        print(f"💾 Output Directory: {config.output_directory}")
        print("=" * 60 + "\n")

    try:
        transcriber = UnifiedTranscriber(config)
        text = transcriber.transcribe_file(str(args.audio_file))

        if not args.quiet:
            preview = text[:500] + ("..." if len(text) > 500 else "")
            print("\n✅ Transcription Complete!\n")
            print(preview)

        return 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("\nTranscription interrupted by user")
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).error(f"Transcription failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())



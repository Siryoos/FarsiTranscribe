"""Command-line interface for FarsiTranscribe (OO-style)."""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Callable

from . import FarsiTranscriber, TranscriptionConfig, ConfigPresets
from .utils import TranscriptionManager


class CLIApplication:
    """Encapsulates CLI behavior to keep the module object oriented."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, verbose: bool = False) -> None:
        """Configure logging based on verbosity level."""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    "farsi_transcribe.log", mode="w", encoding="utf-8"
                ),
            ],
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(
            prog="farsi-transcribe",
            description=(
                "FarsiTranscribe - Efficient Persian/Farsi Audio Transcription"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=(
                "Examples:\n"
                "  %(prog)s audio.mp3                              # Basic transcription\n"
                "  %(prog)s audio.mp3 --preset high-quality        # High quality preset\n"
                "  %(prog)s audio.mp3 --model large-v3 --gpu       # Use large model on GPU\n"
                "  %(prog)s audio.mp3 --output-dir results/        # Custom output directory\n"
                "  %(prog)s audio.mp3 --formats txt json segments  # Multiple output formats\n\n"
                "Presets:\n"
                "  fast              - Quick transcription with smaller model\n"
                "  balanced          - Good balance of speed and quality (default)\n"
                "  high-quality      - Best quality with larger model\n"
                "  memory-efficient  - For systems with limited RAM\n"
                "  persian-optimized - Optimized specifically for Persian audio\n"
                "  gpu-optimized     - Optimized for GPU processing\n"
            ),
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
            choices=[
                "tiny",
                "base",
                "small",
                "medium",
                "large",
                "large-v2",
                "large-v3",
            ],
            help="Whisper model to use (overrides preset)",
        )
        parser.add_argument(
            "--language",
            "-l",
            default="fa",
            help="Language code (default: fa for Farsi/Persian)",
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
        parser.add_argument(
            "--gpu",
            "--cuda",
            action="store_true",
            help="Force GPU usage",
        )
        parser.add_argument(
            "--cpu", action="store_true", help="Force CPU usage"
        )
        parser.add_argument(
            "--output-dir",
            "-o",
            type=Path,
            default=Path("./output"),
            help="Output directory (default: ./output)",
        )
        parser.add_argument(
            "--formats",
            "-f",
            nargs="+",
            choices=["txt", "json", "segments"],
            default=["txt", "json"],
            help="Output formats (default: txt json)",
        )
        parser.add_argument(
            "--no-summary",
            action="store_true",
            help="Do not save transcription summary",
        )
        parser.add_argument(
            "--no-normalize",
            action="store_true",
            help="Disable Persian text normalization",
        )
        parser.add_argument(
            "--remove-diacritics",
            action="store_true",
            help="Remove Arabic diacritical marks",
        )
        parser.add_argument(
            "--stream",
            action="store_true",
            help="Use streaming mode for large files",
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

    def get_config_from_args(self, args: argparse.Namespace) -> TranscriptionConfig:
        """Create configuration from command-line arguments."""
        preset_map: Dict[str, Callable[[], TranscriptionConfig]] = {
            "fast": ConfigPresets.fast,
            "balanced": ConfigPresets.balanced,
            "high-quality": ConfigPresets.high_quality,
            "memory-efficient": ConfigPresets.memory_efficient,
            "persian-optimized": ConfigPresets.persian_optimized,
            "gpu-optimized": ConfigPresets.gpu_optimized,
        }

        config = preset_map[args.preset]()

        if args.model:
            config.model_name = args.model
        if args.language:
            config.language = args.language
        if args.chunk_duration:
            config.chunk_duration = args.chunk_duration
        if args.overlap:
            config.overlap = args.overlap
        if args.gpu:
            config.device = "cuda"
        elif args.cpu:
            config.device = "cpu"
        if args.output_dir:
            config.output_directory = args.output_dir
        if args.formats:
            config.output_formats = args.formats
        if args.no_normalize:
            config.persian_normalization = False
        if args.remove_diacritics:
            config.remove_diacritics = True

        return config

    def print_banner(
        self, config: TranscriptionConfig, audio_file: Path, quiet: bool = False
    ) -> None:
        """Print startup banner with configuration info."""
        if quiet:
            return
        print("\n🎙️  FarsiTranscribe - Persian Audio Transcription")
        print("=" * 60)
        print(f"📁 Audio File: {audio_file.name}")
        size_mb = audio_file.stat().st_size / 1024 / 1024
        print(f"📊 File Size: {size_mb:.1f} MB")
        print(f"🔧 Model: {config.model_name}")
        print(f"🖥️  Device: {config.device.upper()}")
        print(
            f"⚡ Chunk Duration: {config.chunk_duration}s (overlap: {config.overlap}s)"
        )
        print(f"💾 Output Directory: {config.output_directory}")
        print(f"📄 Output Formats: {', '.join(config.output_formats)}")
        print("=" * 60 + "\n")

    def print_results(self, result, saved_files, quiet: bool = False) -> None:
        """Print transcription results and summary."""
        if quiet:
            return
        print("\n✅ Transcription Complete!")
        print("=" * 60)
        print(f"⏱️  Duration: {result.duration:.1f}s")
        print(f"⚡ Processing Time: {result.processing_time:.1f}s")
        print(f"📈 Real-time Factor: {result.real_time_factor:.2f}x")
        print(f"📝 Characters: {len(result.text):,}")
        print(f"📄 Words: {len(result.text.split()):,}")
        print("\n📁 Output Files:")
        for format_name, file_path in saved_files.items():
            print(f"  - {format_name.upper()}: {file_path}")
        print("=" * 60)
        if result.text:
            print("\n📝 Preview (first 500 characters):")
            print("-" * 60)
            preview = (
                result.text[:500] + "..." if len(result.text) > 500 else result.text
            )
            print(preview)
            print("-" * 60)

    def run(self, argv: Optional[list] = None) -> int:
        """Run the CLI application."""
        parser = self.create_parser()
        args = parser.parse_args(argv)
        self.setup_logging(args.verbose)

        if not args.audio_file.exists():
            self.logger.error(f"Audio file not found: {args.audio_file}")
            return 1

        config = self.get_config_from_args(args)
        self.print_banner(config, args.audio_file, args.quiet)

        try:
            with FarsiTranscriber(config) as transcriber:
                if args.stream:
                    result = transcriber.transcribe_stream(args.audio_file)
                else:
                    result = transcriber.transcribe_file(args.audio_file)

                manager = TranscriptionManager(config.output_directory)
                base_name = args.audio_file.stem
                saved_files = manager.save_result(
                    result, base_name, config.output_formats
                )
                if not args.no_summary:
                    summary_path = manager.save_summary(result, base_name)
                    saved_files["summary"] = summary_path
                self.print_results(result, saved_files, args.quiet)
                return 0
        except KeyboardInterrupt:
            self.logger.info("\nTranscription interrupted by user")
            return 1
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Transcription failed: {exc}", exc_info=True)
            return 1


def main(argv: Optional[list] = None) -> None:
    """Module entrypoint that delegates to CLIApplication."""
    app = CLIApplication()
    exit_code = app.run(argv)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
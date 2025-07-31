"""
Command-line interface for FarsiTranscribe.

This module provides a user-friendly CLI for audio transcription with
support for various presets and customization options.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from . import FarsiTranscriber, TranscriptionConfig, ConfigPresets
from .utils import TranscriptionManager


def setup_logging(verbose: bool = False):
    """
    Configures logging to output messages to both stdout and a log file, using DEBUG level if verbose is True, otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('farsi_transcribe.log', mode='w', encoding='utf-8')
        ]
    )


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the FarsiTranscribe CLI.
    
    Returns:
        argparse.ArgumentParser: An argument parser pre-configured with options for audio file input, preset and model selection, language, chunking, device choice, output formats and directory, Persian text normalization, streaming mode, and verbosity controls.
    """
    parser = argparse.ArgumentParser(
        prog='farsi-transcribe',
        description='FarsiTranscribe - Efficient Persian/Farsi Audio Transcription',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                              # Basic transcription
  %(prog)s audio.mp3 --preset high-quality        # High quality preset
  %(prog)s audio.mp3 --model large-v3 --gpu       # Use large model on GPU
  %(prog)s audio.mp3 --output-dir results/        # Custom output directory
  %(prog)s audio.mp3 --formats txt json segments  # Multiple output formats

Presets:
  fast            - Quick transcription with smaller model
  balanced        - Good balance of speed and quality (default)
  high-quality    - Best quality with larger model
  memory-efficient - For systems with limited RAM
  persian-optimized - Optimized specifically for Persian audio
  gpu-optimized   - Optimized for GPU processing
        """
    )
    
    # Positional arguments
    parser.add_argument('audio_file', 
                       type=Path,
                       help='Path to audio file')
    
    # Preset selection
    parser.add_argument('--preset', '-p',
                       choices=['fast', 'balanced', 'high-quality', 
                               'memory-efficient', 'persian-optimized', 'gpu-optimized'],
                       default='balanced',
                       help='Configuration preset (default: balanced)')
    
    # Model settings
    parser.add_argument('--model', '-m',
                       choices=['tiny', 'base', 'small', 'medium', 
                               'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use (overrides preset)')
    
    parser.add_argument('--language', '-l',
                       default='fa',
                       help='Language code (default: fa for Farsi/Persian)')
    
    # Processing options
    parser.add_argument('--chunk-duration',
                       type=int,
                       help='Chunk duration in seconds (overrides preset)')
    
    parser.add_argument('--overlap',
                       type=int,
                       help='Overlap between chunks in seconds (overrides preset)')
    
    parser.add_argument('--gpu', '--cuda',
                       action='store_true',
                       help='Force GPU usage')
    
    parser.add_argument('--cpu',
                       action='store_true',
                       help='Force CPU usage')
    
    # Output options
    parser.add_argument('--output-dir', '-o',
                       type=Path,
                       default=Path('./output'),
                       help='Output directory (default: ./output)')
    
    parser.add_argument('--formats', '-f',
                       nargs='+',
                       choices=['txt', 'json', 'segments'],
                       default=['txt', 'json'],
                       help='Output formats (default: txt json)')
    
    parser.add_argument('--no-summary',
                       action='store_true',
                       help='Do not save transcription summary')
    
    # Persian-specific options
    parser.add_argument('--no-normalize',
                       action='store_true',
                       help='Disable Persian text normalization')
    
    parser.add_argument('--remove-diacritics',
                       action='store_true',
                       help='Remove Arabic diacritical marks')
    
    # Other options
    parser.add_argument('--stream',
                       action='store_true',
                       help='Use streaming mode for large files')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Suppress progress output')
    
    return parser


def get_config_from_args(args) -> TranscriptionConfig:
    """
    Constructs a TranscriptionConfig object based on the selected preset and overrides its attributes with command-line arguments.
    
    Parameters:
        args: Parsed command-line arguments containing preset selection and optional overrides.
    
    Returns:
        TranscriptionConfig: The finalized configuration for the transcription process.
    """
    # Start with preset
    preset_map = {
        'fast': ConfigPresets.fast,
        'balanced': ConfigPresets.balanced,
        'high-quality': ConfigPresets.high_quality,
        'memory-efficient': ConfigPresets.memory_efficient,
        'persian-optimized': ConfigPresets.persian_optimized,
        'gpu-optimized': ConfigPresets.gpu_optimized
    }
    
    config = preset_map[args.preset]()
    
    # Override with command-line options
    if args.model:
        config.model_name = args.model
    
    if args.language:
        config.language = args.language
    
    if args.chunk_duration:
        config.chunk_duration = args.chunk_duration
    
    if args.overlap:
        config.overlap = args.overlap
    
    if args.gpu:
        config.device = 'cuda'
    elif args.cpu:
        config.device = 'cpu'
    
    if args.output_dir:
        config.output_directory = args.output_dir
    
    if args.formats:
        config.output_formats = args.formats
    
    if args.no_normalize:
        config.persian_normalization = False
    
    if args.remove_diacritics:
        config.remove_diacritics = True
    
    return config


def print_banner(config: TranscriptionConfig, audio_file: Path, quiet: bool = False):
    """
    Display a startup banner summarizing the transcription configuration and input audio file details.
    
    Skips output if quiet mode is enabled.
    """
    if quiet:
        return
    
    print("\n🎙️  FarsiTranscribe - Persian Audio Transcription")
    print("=" * 60)
    print(f"📁 Audio File: {audio_file.name}")
    print(f"📊 File Size: {audio_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"🔧 Model: {config.model_name}")
    print(f"🖥️  Device: {config.device.upper()}")
    print(f"⚡ Chunk Duration: {config.chunk_duration}s (overlap: {config.overlap}s)")
    print(f"💾 Output Directory: {config.output_directory}")
    print(f"📄 Output Formats: {', '.join(config.output_formats)}")
    print("=" * 60 + "\n")


def print_results(result, saved_files, quiet: bool = False):
    """
    Display a summary of the transcription results, including duration, processing time, real-time factor, character and word counts, output file locations, and a preview of the transcribed text.
    
    Skips output if quiet mode is enabled.
    
    Parameters:
        result: The transcription result object containing text and metadata.
        saved_files: A dictionary mapping output format names to their file paths.
        quiet (bool): If True, suppresses all output.
    """
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
    
    # Print preview
    if result.text:
        print("\n📝 Preview (first 500 characters):")
        print("-" * 60)
        preview = result.text[:500] + "..." if len(result.text) > 500 else result.text
        print(preview)
        print("-" * 60)


def main(argv: Optional[list] = None):
    """
    Runs the FarsiTranscribe command-line interface, handling argument parsing, configuration, transcription execution, result saving, and user feedback.
    
    Parameters:
        argv (Optional[list]): List of command-line arguments to parse. If None, uses sys.argv.
    
    This function sets up logging, validates the input audio file, constructs the transcription configuration, and manages the transcription process. It supports both streaming and file-based transcription modes, saves results in specified formats, and prints summaries unless quiet mode is enabled. Handles user interruption and errors by logging and exiting with an error status.
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate audio file
    if not args.audio_file.exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Get configuration
    config = get_config_from_args(args)
    
    # Print banner
    print_banner(config, args.audio_file, args.quiet)
    
    try:
        # Initialize transcriber
        with FarsiTranscriber(config) as transcriber:
            # Transcribe
            if args.stream:
                result = transcriber.transcribe_stream(args.audio_file)
            else:
                result = transcriber.transcribe_file(args.audio_file)
            
            # Save results
            manager = TranscriptionManager(config.output_directory)
            base_name = args.audio_file.stem
            
            saved_files = manager.save_result(
                result,
                base_name,
                config.output_formats
            )
            
            # Save summary unless disabled
            if not args.no_summary:
                summary_path = manager.save_summary(result, base_name)
                saved_files['summary'] = summary_path
            
            # Print results
            print_results(result, saved_files, args.quiet)
            
    except KeyboardInterrupt:
        logger.info("\nTranscription interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
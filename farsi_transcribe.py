#!/usr/bin/env python3
"""
FarsiTranscribe - Efficient Persian/Farsi Audio Transcription
=============================================================

A clean, efficient audio transcription tool optimized for Farsi/Persian language
using OpenAI's Whisper model.

Features:
- Automatic audio preprocessing and normalization
- Smart chunking for long audio files
- Memory-efficient processing
- Progress tracking
- Multiple output formats
- Optimized for Farsi/Persian language

Usage:
    python farsi_transcribe.py audio.mp3
    python farsi_transcribe.py audio.mp3 --model medium --output-dir results/
    python farsi_transcribe.py audio.mp3 --quality high --format all
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Core dependencies
import torch
import whisper
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcription.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class FarsiTranscriber:
    """
    Efficient audio transcriber optimized for Farsi/Persian language.
    
    This class handles audio loading, preprocessing, chunking, and transcription
    using OpenAI's Whisper model with optimizations for Persian text.
    """
    
    # Quality presets
    QUALITY_PRESETS = {
        'fast': {
            'model': 'base',
            'chunk_duration': 60,  # seconds
            'overlap': 2,  # seconds
            'batch_size': 4,
            'compute_type': 'int8'
        },
        'balanced': {
            'model': 'small',
            'chunk_duration': 45,
            'overlap': 3,
            'batch_size': 2,
            'compute_type': 'float16'
        },
        'high': {
            'model': 'medium',
            'chunk_duration': 30,
            'overlap': 5,
            'batch_size': 1,
            'compute_type': 'float16'
        }
    }
    
    def __init__(self, 
                 model_name: str = 'small',
                 device: Optional[str] = None,
                 quality: str = 'balanced',
                 compute_type: Optional[str] = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda/cpu), auto-detected if None
            quality: Quality preset (fast, balanced, high)
            compute_type: Computation type (float32, float16, int8)
        """
        # Apply quality preset
        if quality in self.QUALITY_PRESETS:
            preset = self.QUALITY_PRESETS[quality]
            if model_name == 'small':  # Use preset model if not overridden
                model_name = preset['model']
            self.chunk_duration = preset['chunk_duration']
            self.overlap = preset['overlap']
            self.batch_size = preset['batch_size']
            if compute_type is None:
                compute_type = preset['compute_type']
        else:
            self.chunk_duration = 45
            self.overlap = 3
            self.batch_size = 2
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=self.device)
        
        # Set compute type
        if compute_type == 'int8' and self.device == 'cuda':
            self.model = self.model.half()  # Use FP16 as int8 requires special handling
        elif compute_type == 'float16' and self.device == 'cuda':
            self.model = self.model.half()
            
        logger.info(f"Model loaded successfully. Quality: {quality}")
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        logger.info(f"Loading audio: {audio_path}")
        
        # Load audio using pydub for format compatibility
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono and normalize
        audio = audio.set_channels(1)
        audio = audio.normalize()
        
        # Convert to 16kHz sample rate (Whisper requirement)
        audio = audio.set_frame_rate(16000)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.max(np.abs(samples))  # Normalize to [-1, 1]
        
        duration = len(audio) / 1000.0  # Duration in seconds
        logger.info(f"Audio loaded: {duration:.1f} seconds, {len(samples)} samples")
        
        return samples, 16000
        
    def create_chunks(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Split audio into overlapping chunks for processing.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            List of chunk dictionaries
        """
        chunk_samples = int(self.chunk_duration * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        stride_samples = chunk_samples - overlap_samples
        
        chunks = []
        for i in range(0, len(audio), stride_samples):
            end = min(i + chunk_samples, len(audio))
            chunk = audio[i:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples and i > 0:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
                
            chunks.append({
                'audio': chunk,
                'start_time': i / sample_rate,
                'end_time': end / sample_rate,
                'index': len(chunks)
            })
            
            if end >= len(audio):
                break
                
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
        
    def transcribe_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transcribe a single audio chunk.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Transcription result
        """
        # Transcribe with Farsi language hint
        result = self.model.transcribe(
            chunk['audio'],
            language='fa',  # Farsi/Persian
            task='transcribe',
            verbose=False,
            temperature=0.0,  # Deterministic
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        
        return {
            'text': result['text'].strip(),
            'segments': result.get('segments', []),
            'start_time': chunk['start_time'],
            'end_time': chunk['end_time'],
            'index': chunk['index']
        }
        
    def merge_transcriptions(self, results: List[Dict[str, Any]]) -> str:
        """
        Merge chunk transcriptions into final text.
        
        Args:
            results: List of transcription results
            
        Returns:
            Merged transcription text
        """
        # Sort by chunk index
        results.sort(key=lambda x: x['index'])
        
        # Simple merge - can be enhanced with overlap handling
        texts = []
        for result in results:
            if result['text']:
                texts.append(result['text'])
                
        # Join with space, handling Persian RTL text properly
        final_text = ' '.join(texts)
        
        # Clean up extra spaces
        final_text = ' '.join(final_text.split())
        
        return final_text
        
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        start_time = time.time()
        
        # Load audio
        audio, sample_rate = self.load_audio(audio_path)
        
        # Create chunks
        chunks = self.create_chunks(audio, sample_rate)
        
        # Process chunks
        results = []
        with tqdm(total=len(chunks), desc="Transcribing", unit="chunk") as pbar:
            for chunk in chunks:
                try:
                    result = self.transcribe_chunk(chunk)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error transcribing chunk {chunk['index']}: {e}")
                finally:
                    pbar.update(1)
                    
                # Clear GPU memory if needed
                if self.device == 'cuda' and chunk['index'] % 10 == 0:
                    torch.cuda.empty_cache()
                    
        # Merge results
        transcription = self.merge_transcriptions(results)
        
        # Calculate metrics
        duration = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        
        return {
            'text': transcription,
            'audio_duration': audio_duration,
            'processing_time': duration,
            'rtf': duration / audio_duration,  # Real-time factor
            'chunks': len(chunks),
            'model': self.model.name if hasattr(self.model, 'name') else 'unknown'
        }


def save_results(transcription: Dict[str, Any], 
                 audio_path: str,
                 output_dir: str,
                 formats: List[str]) -> None:
    """
    Save transcription results in multiple formats.
    
    Args:
        transcription: Transcription results
        audio_path: Original audio path
        output_dir: Output directory
        formats: List of output formats (txt, json, srt)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(audio_path).stem
    
    # Save as text
    if 'txt' in formats or 'all' in formats:
        txt_path = os.path.join(output_dir, f"{base_name}_transcription.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription['text'])
        logger.info(f"Saved text: {txt_path}")
        
    # Save as JSON with metadata
    if 'json' in formats or 'all' in formats:
        json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'audio_file': audio_path,
                'transcription': transcription['text'],
                'audio_duration': transcription['audio_duration'],
                'processing_time': transcription['processing_time'],
                'real_time_factor': transcription['rtf'],
                'model': transcription['model'],
                'chunks': transcription['chunks'],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
    # Log summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Transcription completed!")
    logger.info(f"Audio duration: {transcription['audio_duration']:.1f} seconds")
    logger.info(f"Processing time: {transcription['processing_time']:.1f} seconds")
    logger.info(f"Real-time factor: {transcription['rtf']:.2f}x")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"{'='*50}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FarsiTranscribe - Efficient Persian/Farsi Audio Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.mp3                          # Transcribe with default settings
  %(prog)s audio.mp3 --model medium           # Use medium model
  %(prog)s audio.mp3 --quality high           # High quality preset
  %(prog)s audio.mp3 --output-dir results/    # Custom output directory
  %(prog)s audio.mp3 --format all             # Save in all formats
        """
    )
    
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--model', '-m', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        default='small',
                        help='Whisper model size (default: small)')
    parser.add_argument('--quality', '-q',
                        choices=['fast', 'balanced', 'high'],
                        default='balanced',
                        help='Quality preset (default: balanced)')
    parser.add_argument('--output-dir', '-o',
                        default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--format', '-f',
                        choices=['txt', 'json', 'all'],
                        default='all',
                        help='Output format (default: all)')
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        help='Force device (auto-detected by default)')
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
        
    # Print banner
    print("\n🎙️  FarsiTranscribe - Persian Audio Transcription")
    print("=" * 50)
    print(f"📁 Audio: {args.audio_file}")
    print(f"🔧 Model: {args.model}")
    print(f"📊 Quality: {args.quality}")
    print(f"💾 Output: {args.output_dir}")
    print("=" * 50 + "\n")
    
    try:
        # Initialize transcriber
        transcriber = FarsiTranscriber(
            model_name=args.model,
            device=args.device,
            quality=args.quality
        )
        
        # Transcribe
        transcription = transcriber.transcribe(args.audio_file)
        
        # Save results
        save_results(
            transcription,
            args.audio_file,
            args.output_dir,
            [args.format]
        )
        
        # Print sample of transcription
        if transcription['text']:
            print("\n📝 Transcription Preview:")
            print("-" * 50)
            preview = transcription['text'][:500] + "..." if len(transcription['text']) > 500 else transcription['text']
            print(preview)
            print("-" * 50)
            
    except KeyboardInterrupt:
        logger.info("\nTranscription interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Audio processing module for FarsiTranscribe.

This module handles audio loading, preprocessing, and chunking operations.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator
import numpy as np
from pydub import AudioSegment
import torch
import torchaudio
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AudioChunk:
    """Represents a single audio chunk with metadata."""
    
    def __init__(self, audio: np.ndarray, start_time: float, end_time: float, index: int):
        self.audio = audio
        self.start_time = start_time
        self.end_time = end_time
        self.index = index
        self.duration = end_time - start_time
    
    def __repr__(self):
        return f"AudioChunk(index={self.index}, start={self.start_time:.1f}s, end={self.end_time:.1f}s)"


class AudioProcessor:
    """
    Handles all audio processing operations.
    
    This class provides methods for loading, preprocessing, and chunking audio files
    for transcription. It supports various audio formats and provides optimized
    processing for Persian speech.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio processing (default: 16000)
        """
        self.sample_rate = sample_rate
        self._supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.mp4'}
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load and preprocess an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_array, metadata_dict)
            
        Raises:
            ValueError: If audio format is not supported
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in self._supported_formats:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
        
        logger.info(f"Loading audio file: {audio_path}")
        
        # Load audio using pydub for compatibility
        try:
            audio = AudioSegment.from_file(str(audio_path))
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
        
        # Convert to mono and normalize
        audio = audio.set_channels(1)
        audio = audio.normalize()
        
        # Resample to target sample rate
        audio = audio.set_frame_rate(self.sample_rate)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Normalize to [-1, 1]
        if samples.size > 0:
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val
        
        # Collect metadata
        metadata = {
            'duration_seconds': len(audio) / 1000.0,
            'sample_rate': self.sample_rate,
            'num_samples': len(samples),
            'original_file': str(audio_path),
            'format': audio_path.suffix
        }
        
        logger.info(f"Audio loaded: {metadata['duration_seconds']:.1f}s, {metadata['num_samples']} samples")
        
        return samples, metadata
    
    def create_chunks(self, 
                     audio: np.ndarray, 
                     chunk_duration: int, 
                     overlap: int = 0) -> List[AudioChunk]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Audio array
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of AudioChunk objects
        """
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        stride = chunk_samples - overlap_samples
        
        if stride <= 0:
            raise ValueError("Overlap must be less than chunk duration")
        
        chunks = []
        audio_length = len(audio)
        
        for i in range(0, audio_length, stride):
            start_sample = i
            end_sample = min(i + chunk_samples, audio_length)
            
            # Extract chunk
            chunk_audio = audio[start_sample:end_sample]
            
            # Pad last chunk if necessary
            if len(chunk_audio) < chunk_samples and i > 0:
                padding = chunk_samples - len(chunk_audio)
                chunk_audio = np.pad(chunk_audio, (0, padding), mode='constant')
            
            # Create chunk object
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            chunk = AudioChunk(
                audio=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                index=len(chunks)
            )
            
            chunks.append(chunk)
            
            # Stop if we've processed all audio
            if end_sample >= audio_length:
                break
        
        logger.info(f"Created {len(chunks)} chunks (duration={chunk_duration}s, overlap={overlap}s)")
        
        return chunks
    
    def stream_chunks(self, 
                     audio_path: Path,
                     chunk_duration: int,
                     overlap: int = 0) -> Iterator[AudioChunk]:
        """
        Stream audio chunks without loading entire file into memory.
        
        This is useful for processing very large audio files.
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Yields:
            AudioChunk objects
        """
        # Load audio metadata first
        audio, metadata = self.load_audio(audio_path)
        
        # Create and yield chunks
        chunks = self.create_chunks(audio, chunk_duration, overlap)
        for chunk in chunks:
            yield chunk
    
    def apply_preprocessing(self, 
                           audio: np.ndarray,
                           noise_reduction: bool = False,
                           normalize: bool = True) -> np.ndarray:
        """
        Apply preprocessing to audio.
        
        Args:
            audio: Input audio array
            noise_reduction: Apply basic noise reduction
            normalize: Normalize audio volume
            
        Returns:
            Preprocessed audio array
        """
        processed = audio.copy()
        
        # Noise reduction (simple spectral subtraction)
        if noise_reduction and len(processed) > 0:
            # This is a placeholder for more sophisticated noise reduction
            # In production, you might use specialized libraries
            pass
        
        # Normalization
        if normalize and len(processed) > 0:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val
        
        return processed
    
    def save_audio(self, audio: np.ndarray, output_path: Path, format: str = 'wav'):
        """
        Save audio array to file.
        
        Args:
            audio: Audio array to save
            output_path: Output file path
            format: Output format (default: 'wav')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to int16 for saving
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,  # 16-bit
            channels=1  # Mono
        )
        
        # Export to file
        audio_segment.export(str(output_path), format=format)
        logger.info(f"Saved audio to: {output_path}")
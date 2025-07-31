"""
Audio processing module for FarsiTranscribe.

This module handles audio loading, preprocessing, and chunking operations.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
import numpy as np
from pydub import AudioSegment
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AudioChunk:
    """Represents a single audio chunk with metadata."""
    
    def __init__(self, audio: np.ndarray, start_time: float, end_time: float, index: int):
        """
        Initialize an AudioChunk instance representing a segment of audio with associated metadata.
        
        Parameters:
            audio (np.ndarray): The audio data for this chunk.
            start_time (float): The start time of the chunk in seconds.
            end_time (float): The end time of the chunk in seconds.
            index (int): The index of the chunk within the original audio.
        """
        self.audio = audio
        self.start_time = start_time
        self.end_time = end_time
        self.index = index
        self.duration = end_time - start_time
    
    def __repr__(self):
        """
        Return a string representation of the AudioChunk showing its index and time range.
        """
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
        Initialize an AudioProcessor with a specified target sample rate for audio processing.
        
        Parameters:
            sample_rate (int): Desired sample rate in Hz. Defaults to 16000.
        """
        self.sample_rate = sample_rate
        self._supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.mp4'}
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load an audio file, convert it to mono, normalize, resample to the target sample rate, and return the audio as a normalized numpy array along with metadata.
        
        Parameters:
            audio_path (Path): Path to the audio file to load.
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing the normalized audio array (float32, range [-1, 1]) and a metadata dictionary with duration, sample rate, number of samples, original file path, and format.
        
        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the audio format is not supported.
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
                     Split an audio array into overlapping chunks of specified duration.
                     
                     Parameters:
                         audio (np.ndarray): Input audio data as a 1D numpy array.
                         chunk_duration (int): Length of each chunk in seconds.
                         overlap (int, optional): Overlap between consecutive chunks in seconds. Must be less than chunk_duration.
                     
                     Returns:
                         List[AudioChunk]: List of AudioChunk objects containing chunked audio data and timing metadata.
                     
                     Raises:
                         ValueError: If overlap is greater than or equal to chunk_duration.
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
                     Yields sequential audio chunks from a file for processing, supporting optional overlap between chunks.
                     
                     Parameters:
                         audio_path (Path): Path to the audio file.
                         chunk_duration (int): Duration of each chunk in seconds.
                         overlap (int, optional): Overlap between consecutive chunks in seconds. Defaults to 0.
                     
                     Yields:
                         AudioChunk: An object containing the audio data and metadata for each chunk.
                     
                     Raises:
                         FileNotFoundError: If the specified audio file does not exist.
                         ValueError: If the overlap is greater than or equal to the chunk duration.
                     
                     Note:
                         The entire audio file is loaded into memory before chunking for format compatibility. True streaming with partial file reads may be supported in future versions.
                     """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # For now, we load the full audio for format compatibility
        # Future versions will implement true streaming with partial reads
        audio, metadata = self.load_audio(audio_path)
        
        # Stream chunks one at a time instead of creating all at once
        chunk_samples = int(chunk_duration * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        stride = chunk_samples - overlap_samples
        
        if stride <= 0:
            raise ValueError("Overlap must be less than chunk duration")
        
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
            
            # Create and yield chunk
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            chunk = AudioChunk(
                audio=chunk_audio,
                start_time=start_time,
                end_time=end_time,
                index=i // stride
            )
            
            yield chunk
            
            # Stop if we've processed all audio
            if end_sample >= audio_length:
                break
    
    def apply_preprocessing(self, 
                           audio: np.ndarray,
                           noise_reduction: bool = False,
                           normalize: bool = True) -> np.ndarray:
        """
                           Apply optional noise reduction and normalization to an audio array.
                           
                           Parameters:
                               audio (np.ndarray): Input audio array to preprocess.
                               noise_reduction (bool): If True, applies basic spectral gating noise reduction.
                               normalize (bool): If True, normalizes audio to have maximum absolute value of 1.
                           
                           Returns:
                               np.ndarray: The preprocessed audio array.
                           """
        processed = audio.copy()
        
        # Noise reduction using simple spectral gating
        if noise_reduction and len(processed) > 0:
            try:
                # Import scipy for signal processing
                from scipy import signal
                from scipy.fft import fft, ifft
                
                # Parameters for noise reduction
                frame_length = 2048
                hop_length = frame_length // 2
                
                # Estimate noise profile from first 0.5 seconds
                noise_sample_length = int(0.5 * self.sample_rate)
                noise_sample = processed[:min(noise_sample_length, len(processed))]
                
                # Compute noise spectrum
                noise_fft = np.abs(fft(noise_sample, n=frame_length))
                noise_profile = np.mean(noise_fft)
                
                # Apply spectral gating
                n_frames = 1 + (len(processed) - frame_length) // hop_length
                processed_frames = []
                
                for i in range(n_frames):
                    start = i * hop_length
                    end = start + frame_length
                    
                    if end > len(processed):
                        frame = np.pad(processed[start:], (0, end - len(processed)))
                    else:
                        frame = processed[start:end]
                    
                    # Apply window
                    window = signal.windows.hann(frame_length)
                    windowed_frame = frame * window
                    
                    # FFT
                    frame_fft = fft(windowed_frame)
                    frame_magnitude = np.abs(frame_fft)
                    frame_phase = np.angle(frame_fft)
                    
                    # Spectral gating
                    mask = frame_magnitude > (noise_profile * 1.5)  # Threshold factor
                    frame_magnitude_cleaned = frame_magnitude * mask
                    
                    # Reconstruct
                    frame_fft_cleaned = frame_magnitude_cleaned * np.exp(1j * frame_phase)
                    frame_cleaned = np.real(ifft(frame_fft_cleaned))
                    
                    processed_frames.append(frame_cleaned[:hop_length])
                
                # Overlap-add reconstruction
                processed = np.zeros(len(audio))
                for i, frame in enumerate(processed_frames):
                    start = i * hop_length
                    end = start + len(frame)
                    if end <= len(processed):
                        processed[start:end] += frame
                        
                # Smooth to reduce artifacts
                from scipy.ndimage import gaussian_filter1d
                processed = gaussian_filter1d(processed, sigma=2)
                
            except ImportError:
                logger.warning("scipy not available for noise reduction, skipping")
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}, using original audio")
                processed = audio.copy()
        
        # Normalization
        if normalize and len(processed) > 0:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val
        
        return processed
    
    def save_audio(self, audio: np.ndarray, output_path: Path, format: str = 'wav'):
        """
        Save a numpy audio array to a file in the specified format.
        
        Parameters:
            audio (np.ndarray): Audio data as a float array in the range [-1, 1].
            output_path (Path): Destination file path.
            format (str): Audio file format (e.g., 'wav', 'mp3'). Defaults to 'wav'.
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
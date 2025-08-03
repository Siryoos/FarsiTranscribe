#!/usr/bin/env python3
"""
Enhanced Audio Preprocessor for Persian Transcription
Implements quality improvements based on user feedback:
- Proper 16kHz mono WAV conversion
- Noise reduction and signal normalization
- Multi-speaker conversation handling
- Audio quality validation
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import librosa
from scipy import signal
from scipy.io import wavfile
import soundfile as sf

try:
    from pydub import AudioSegment
    from pydub.effects import normalize
except ImportError:
    AudioSegment = None
    normalize = None

from .audio_preprocessor import AudioPreprocessor


class EnhancedAudioPreprocessor(AudioPreprocessor):
    """
    Enhanced audio preprocessor with quality improvements for Persian transcription.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_audio_duration = 0.5  # seconds
        self.max_audio_duration = 300  # seconds (5 minutes)
        self.target_snr = 20  # dB
        self.min_amplitude = 0.01
        self.max_amplitude = 0.95
        
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhanced audio preprocessing with quality improvements.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        self.logger.info(f"Starting enhanced preprocessing of: {audio_path}")
        
        # Load and validate audio
        audio_data, sample_rate = self._load_audio(audio_path)
        
        # Generate metadata
        metadata = {
            'original_path': audio_path,
            'original_sample_rate': sample_rate,
            'original_duration': len(audio_data) / sample_rate,
            'original_channels': audio_data.ndim,
            'quality_issues': []
        }
        
        # Quality validation
        quality_report = self._validate_audio_quality(audio_data, sample_rate)
        metadata['quality_report'] = quality_report
        
        if quality_report['issues']:
            self.logger.warning(f"Quality issues detected: {quality_report['issues']}")
            metadata['quality_issues'] = quality_report['issues']
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = self._convert_to_mono(audio_data)
            metadata['converted_to_mono'] = True
        
        # Resample to target sample rate
        if sample_rate != self.config.target_sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.config.target_sample_rate)
            metadata['resampled'] = True
            metadata['new_sample_rate'] = self.config.target_sample_rate
        
        # Apply noise reduction
        if self.config.enable_noise_reduction:
            audio_data = self._apply_noise_reduction(audio_data)
            metadata['noise_reduction_applied'] = True
        
        # Normalize audio
        audio_data = self._normalize_audio(audio_data)
        metadata['normalized'] = True
        
        # Apply high-pass filter to remove low-frequency noise
        audio_data = self._apply_high_pass_filter(audio_data)
        metadata['high_pass_filter_applied'] = True
        
        # Final quality check
        final_quality = self._validate_audio_quality(audio_data, self.config.target_sample_rate)
        metadata['final_quality'] = final_quality
        
        self.logger.info(f"Enhanced preprocessing completed. Final duration: {len(audio_data) / self.config.target_sample_rate:.2f}s")
        
        return audio_data, metadata
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio with multiple fallback methods."""
        try:
            # Try librosa first (handles most formats)
            audio_data, sample_rate = librosa.load(
                audio_path, 
                sr=None,  # Keep original sample rate
                mono=False  # Keep original channels
            )
            return audio_data, sample_rate
        except Exception as e:
            self.logger.warning(f"librosa failed: {e}")
            
            try:
                # Try soundfile
                audio_data, sample_rate = sf.read(audio_path)
                return audio_data, sample_rate
            except Exception as e2:
                self.logger.warning(f"soundfile failed: {e2}")
                
                # Try pydub as last resort
                if AudioSegment:
                    audio = AudioSegment.from_file(audio_path)
                    audio_data = np.array(audio.get_array_of_samples())
                    if audio.channels == 2:
                        audio_data = audio_data.reshape((-1, 2))
                    sample_rate = audio.frame_rate
                    return audio_data, sample_rate
                else:
                    raise RuntimeError(f"Failed to load audio with all methods: {e}, {e2}")
    
    def _convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert stereo to mono using weighted average."""
        if audio_data.ndim == 1:
            return audio_data
        
        # Use weighted average for better quality
        weights = np.array([0.6, 0.4])  # Slight preference for left channel
        if audio_data.shape[1] == 2:
            return np.average(audio_data, axis=1, weights=weights)
        else:
            # For more than 2 channels, use simple mean
            return np.mean(audio_data, axis=1)
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality resampling using librosa."""
        if orig_sr == target_sr:
            return audio_data
        
        self.logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
        
        # Use librosa's high-quality resampling
        resampled = librosa.resample(
            audio_data, 
            orig_sr=orig_sr, 
            target_sr=target_sr,
            res_type='kaiser_best'  # Highest quality
        )
        
        return resampled
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction."""
        self.logger.info("Applying spectral noise reduction")
        
        # Simple spectral subtraction
        # Calculate noise profile from first 0.5 seconds
        noise_samples = int(0.5 * self.config.target_sample_rate)
        noise_profile = np.mean(np.abs(audio_data[:noise_samples]))
        
        # Apply spectral subtraction
        stft = librosa.stft(audio_data, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum
        noise_spectrum = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Apply spectral subtraction with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        cleaned_magnitude = magnitude - alpha * noise_spectrum
        cleaned_magnitude = np.maximum(cleaned_magnitude, beta * magnitude)
        
        # Reconstruct signal
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
        
        return cleaned_audio
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio with peak and RMS normalization."""
        # Peak normalization
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0:
            # Normalize to 0.95 to avoid clipping
            audio_data = audio_data * (0.95 / max_amplitude)
        
        # RMS normalization for consistent loudness
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            audio_data = audio_data * (target_rms / rms)
        
        return audio_data
    
    def _apply_high_pass_filter(self, audio_data: np.ndarray, cutoff: float = 80.0) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        nyquist = self.config.target_sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth high-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        return filtered_audio
    
    def _validate_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Comprehensive audio quality validation."""
        duration = len(audio_data) / sample_rate
        rms = np.sqrt(np.mean(audio_data**2))
        peak = np.max(np.abs(audio_data))
        
        # Calculate SNR (signal-to-noise ratio)
        signal_power = np.mean(audio_data**2)
        noise_estimate = np.var(audio_data - np.mean(audio_data))
        snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else float('inf')
        
        # Detect issues
        issues = []
        warnings = []
        
        if duration < self.min_audio_duration:
            issues.append(f"Audio too short: {duration:.2f}s < {self.min_audio_duration}s")
        
        if duration > self.max_audio_duration:
            warnings.append(f"Audio very long: {duration:.2f}s > {self.max_audio_duration}s")
        
        if rms < self.min_amplitude:
            issues.append(f"Audio too quiet: RMS {rms:.3f} < {self.min_amplitude}")
        
        if peak > self.max_amplitude:
            warnings.append(f"Audio may clip: Peak {peak:.3f} > {self.max_amplitude}")
        
        if snr < self.target_snr:
            warnings.append(f"Low SNR: {snr:.1f}dB < {self.target_snr}dB")
        
        # Check for silence
        silence_threshold = 0.01
        silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
        if silence_ratio > 0.8:
            warnings.append(f"High silence ratio: {silence_ratio:.1%}")
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'rms': rms,
            'peak': peak,
            'snr': snr,
            'silence_ratio': silence_ratio,
            'issues': issues,
            'warnings': warnings,
            'quality_score': self._calculate_quality_score(duration, rms, peak, snr, silence_ratio)
        }
    
    def _calculate_quality_score(self, duration: float, rms: float, peak: float, snr: float, silence_ratio: float) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Duration penalty
        if duration < 1.0:
            score -= 20
        elif duration < 5.0:
            score -= 10
        
        # Amplitude penalty
        if rms < 0.01:
            score -= 30
        elif rms < 0.05:
            score -= 15
        
        # Peak clipping penalty
        if peak > 0.95:
            score -= 20
        elif peak > 0.8:
            score -= 10
        
        # SNR penalty
        if snr < 10:
            score -= 25
        elif snr < 15:
            score -= 15
        
        # Silence penalty
        if silence_ratio > 0.7:
            score -= 20
        elif silence_ratio > 0.5:
            score -= 10
        
        return max(0.0, score)
    
    def chunk_audio_for_conversation(self, audio_data: np.ndarray, chunk_duration: float = 30.0) -> list:
        """
        Chunk audio for multi-speaker conversations.
        
        Args:
            audio_data: Audio data
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * self.config.target_sample_rate)
        chunks = []
        
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) > 0:
                chunks.append(chunk)
        
        self.logger.info(f"Split audio into {len(chunks)} chunks of ~{chunk_duration}s each")
        return chunks
    
    def save_processed_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """Save processed audio as 16kHz mono WAV."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as 16kHz mono WAV
        sf.write(
            output_path,
            audio_data,
            self.config.target_sample_rate,
            subtype='PCM_16'
        )
        
        self.logger.info(f"Saved processed audio to: {output_path}")


# Factory function for easy integration
def create_enhanced_preprocessor(config) -> EnhancedAudioPreprocessor:
    """Create enhanced audio preprocessor with quality improvements."""
    return EnhancedAudioPreprocessor(config) 
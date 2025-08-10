"""
Audio augmenter for Farsi voice fine-tuning.
Provides various audio augmentation techniques to increase training data diversity.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import random
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile


@dataclass
class AudioAugmentationConfig:
    """Configuration for audio augmentation."""
    
    # Basic augmentation settings
    enable_augmentation: bool = True
    augmentation_factor: int = 3  # Number of augmented versions per original
    
    # Pitch and speed augmentation
    enable_pitch_shift: bool = True
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)  # Semitones
    enable_speed_change: bool = True
    speed_change_range: Tuple[float, float] = (0.9, 1.1)
    
    # Noise and effects
    enable_noise_injection: bool = True
    noise_levels: Tuple[float, float] = (0.001, 0.01)
    enable_reverb: bool = True
    reverb_room_scale: Tuple[float, float] = (0.1, 0.3)
    
    # Time-domain augmentation
    enable_time_stretch: bool = True
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    enable_volume_change: bool = True
    volume_change_range: Tuple[float, float] = (0.8, 1.2)
    
    # Farsi-specific settings
    enable_persian_voice_augmentation: bool = True
    preserve_speech_clarity: bool = True
    
    # Quality control
    min_quality_threshold: float = 0.7
    max_augmentations_per_file: int = 5


class AudioAugmenter:
    """Audio augmenter optimized for Farsi voice transcription."""
    
    def __init__(self, config: AudioAugmentationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def augment_audio_file(self, audio_path: str, output_dir: str) -> List[str]:
        """
        Augment a single audio file.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save augmented files
            
        Returns:
            List of paths to augmented audio files
        """
        try:
            self.logger.info(f"Augmenting audio file: {audio_path}")
            
            # Load audio
            audio, sr = self._load_audio(audio_path)
            
            # Generate augmentations
            augmented_files = []
            num_augmentations = min(self.config.augmentation_factor, self.config.max_augmentations_per_file)
            
            for i in range(num_augmentations):
                try:
                    # Apply augmentation pipeline
                    augmented_audio = self._apply_augmentation_pipeline(audio, sr)
                    
                    # Save augmented audio
                    output_filename = f"{Path(audio_path).stem}_aug_{i+1}.wav"
                    output_path = Path(output_dir) / output_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    sf.write(str(output_path), augmented_audio, sr)
                    augmented_files.append(str(output_path))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create augmentation {i+1}: {str(e)}")
                    continue
            
            self.logger.info(f"Created {len(augmented_files)} augmentations")
            return augmented_files
            
        except Exception as e:
            self.logger.error(f"Error augmenting {audio_path}: {str(e)}")
            return []
    
    def augment_directory(self, input_dir: str, output_dir: str) -> Dict[str, List[str]]:
        """
        Augment all audio files in a directory.
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for augmented files
            
        Returns:
            Dictionary mapping original files to their augmentations
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        augmentation_results = {}
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        for audio_file in input_path.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    # Create subdirectory for this file's augmentations
                    file_output_dir = output_path / audio_file.stem
                    
                    augmented_files = self.augment_audio_file(str(audio_file), str(file_output_dir))
                    augmentation_results[str(audio_file)] = augmented_files
                    
                except Exception as e:
                    self.logger.error(f"Failed to augment {audio_file}: {str(e)}")
                    continue
        
        return augmentation_results
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with proper error handling."""
        try:
            # Try librosa first
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            return audio, sr
            
        except Exception as e:
            self.logger.warning(f"Librosa failed, trying torchaudio: {str(e)}")
            try:
                # Fallback to torchaudio
                audio, sr = torchaudio.load(audio_path)
                audio = audio.numpy().flatten()
                return audio, sr
                
            except Exception as e2:
                self.logger.error(f"Both loaders failed: {str(e2)}")
                raise
    
    def _apply_augmentation_pipeline(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply the complete augmentation pipeline."""
        augmented_audio = audio.copy()
        
        # Apply augmentations based on configuration
        if self.config.enable_pitch_shift:
            augmented_audio = self._pitch_shift(augmented_audio, sr)
        
        if self.config.enable_speed_change:
            augmented_audio = self._speed_change(augmented_audio, sr)
        
        if self.config.enable_noise_injection:
            augmented_audio = self._inject_noise(augmented_audio)
        
        if self.config.enable_reverb:
            augmented_audio = self._add_reverb(augmented_audio, sr)
        
        if self.config.enable_time_stretch:
            augmented_audio = self._time_stretch(augmented_audio, sr)
        
        if self.config.enable_volume_change:
            augmented_audio = self._change_volume(augmented_audio)
        
        # Farsi-specific augmentations
        if self.config.enable_persian_voice_augmentation:
            augmented_audio = self._apply_persian_voice_augmentation(augmented_audio, sr)
        
        # Quality control
        if self.config.preserve_speech_clarity:
            augmented_audio = self._preserve_speech_clarity(augmented_audio, sr)
        
        return augmented_audio
    
    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply pitch shifting to audio."""
        if not self.config.enable_pitch_shift:
            return audio
        
        pitch_shift = random.uniform(*self.config.pitch_shift_range)
        
        try:
            # Use librosa for pitch shifting
            shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            return shifted_audio
        except Exception as e:
            self.logger.warning(f"Pitch shift failed: {e}")
            return audio
    
    def _speed_change(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply speed change to audio."""
        if not self.config.enable_speed_change:
            return audio
        
        speed_factor = random.uniform(*self.config.speed_change_range)
        
        try:
            # Use librosa for speed change
            changed_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            return changed_audio
        except Exception as e:
            self.logger.warning(f"Speed change failed: {e}")
            return audio
    
    def _inject_noise(self, audio: np.ndarray) -> np.ndarray:
        """Inject random noise into audio."""
        if not self.config.enable_noise_injection:
            return audio
        
        noise_level = random.uniform(*self.config.noise_levels)
        
        # Generate white noise
        noise = np.random.normal(0, noise_level, len(audio))
        
        # Add noise to audio
        noisy_audio = audio + noise
        
        # Normalize to prevent clipping
        if np.max(np.abs(noisy_audio)) > 1.0:
            noisy_audio = noisy_audio / np.max(np.abs(noisy_audio)) * 0.95
        
        return noisy_audio
    
    def _add_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add reverb effect to audio."""
        if not self.config.enable_reverb:
            return audio
        
        room_scale = random.uniform(*self.config.reverb_room_scale)
        
        try:
            # Simple reverb using convolution with impulse response
            # Create a simple impulse response
            impulse_length = int(sr * room_scale)
            impulse = np.exp(-np.arange(impulse_length) / (sr * room_scale * 0.1))
            impulse = impulse / np.sum(impulse)
            
            # Apply convolution
            reverb_audio = np.convolve(audio, impulse, mode='same')
            
            # Mix original and reverb
            mixed_audio = 0.7 * audio + 0.3 * reverb_audio
            
            return mixed_audio
            
        except Exception as e:
            self.logger.warning(f"Reverb failed: {e}")
            return audio
    
    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply time stretching to audio."""
        if not self.config.enable_time_stretch:
            return audio
        
        stretch_factor = random.uniform(*self.config.time_stretch_range)
        
        try:
            # Use librosa for time stretching
            stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            return stretched_audio
        except Exception as e:
            self.logger.warning(f"Time stretch failed: {e}")
            return audio
    
    def _change_volume(self, audio: np.ndarray) -> np.ndarray:
        """Change volume of audio."""
        if not self.config.enable_volume_change:
            return audio
        
        volume_factor = random.uniform(*self.config.volume_change_range)
        
        # Apply volume change
        volume_changed_audio = audio * volume_factor
        
        # Normalize to prevent clipping
        if np.max(np.abs(volume_changed_audio)) > 1.0:
            volume_changed_audio = volume_changed_audio / np.max(np.abs(volume_changed_audio)) * 0.95
        
        return volume_changed_audio
    
    def _apply_persian_voice_augmentation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply Farsi-specific voice augmentations."""
        if not self.config.enable_persian_voice_augmentation:
            return audio
        
        try:
            # Enhance speech frequencies specific to Persian
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Persian speech enhancement filter (focus on 100Hz - 8kHz)
            speech_mask = np.ones_like(stft)
            speech_mask[(freqs < 100) | (freqs > 8000)] = 0.7
            
            # Apply filter
            stft_enhanced = stft * speech_mask
            enhanced_audio = librosa.istft(stft_enhanced)
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.warning(f"Persian voice augmentation failed: {e}")
            return audio
    
    def _preserve_speech_clarity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Ensure speech clarity is preserved after augmentation."""
        if not self.config.preserve_speech_clarity:
            return audio
        
        try:
            # Apply high-pass filter to remove low-frequency noise
            nyquist = sr / 2
            cutoff = 80  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            filtered_audio = signal.filtfilt(b, a, audio)
            
            # Normalize amplitude
            if np.max(np.abs(filtered_audio)) > 0:
                filtered_audio = filtered_audio / np.max(np.abs(filtered_audio)) * 0.95
            
            return filtered_audio
            
        except Exception as e:
            self.logger.warning(f"Speech clarity preservation failed: {e}")
            return audio
    
    def create_augmentation_report(self, original_files: List[str], augmented_results: Dict[str, List[str]]) -> Dict:
        """Create a report of the augmentation results."""
        report = {
            "total_original_files": len(original_files),
            "successful_augmentations": 0,
            "failed_augmentations": 0,
            "total_augmented_files": 0,
            "augmentation_stats": {},
            "quality_metrics": {}
        }
        
        for original_file, augmented_files in augmented_results.items():
            if augmented_files:
                report["successful_augmentations"] += 1
                report["total_augmented_files"] += len(augmented_files)
            else:
                report["failed_augmentations"] += 1
        
        # Calculate success rate
        if original_files:
            report["success_rate"] = report["successful_augmentations"] / len(original_files)
            report["avg_augmentations_per_file"] = report["total_augmented_files"] / len(original_files)
        
        return report

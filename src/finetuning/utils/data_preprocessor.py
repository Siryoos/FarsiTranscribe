"""
Audio data preprocessor for Farsi voice fine-tuning.
Handles audio cleaning, normalization, and preparation for Whisper models.
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import torch
import torchaudio
from pydub import AudioSegment
from pydub.effects import normalize
import json

logger = logging.getLogger(__name__)


@dataclass
class AudioPreprocessingConfig:
    """Configuration for audio preprocessing."""

    # Audio format settings
    target_sample_rate: int = 16000
    target_channels: int = 1  # Mono for Whisper
    target_format: str = "wav"

    # Quality settings
    min_duration: float = 1.0  # Minimum audio duration in seconds
    max_duration: float = 30.0  # Maximum audio duration in seconds
    min_amplitude: float = 0.01  # Minimum amplitude threshold
    max_amplitude: float = 0.95  # Maximum amplitude threshold

    # Noise reduction settings
    enable_noise_reduction: bool = True
    noise_reduction_strength: float = 0.1
    enable_silence_trimming: bool = True
    silence_threshold: float = 0.01

    # Normalization settings
    enable_normalization: bool = True
    target_db: float = -20.0  # Target loudness in dB
    enable_compression: bool = True
    compression_threshold: float = -16.0

    # Farsi-specific settings
    enable_persian_optimization: bool = True
    remove_background_music: bool = True
    enhance_speech_clarity: bool = True


class AudioDataPreprocessor:
    """Audio data preprocessor optimized for Farsi voice transcription."""

    def __init__(self, config: AudioPreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess_audio_file(
        self, audio_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess a single audio file for fine-tuning.

        Args:
            audio_path: Path to input audio file
            output_path: Path for output file (optional)

        Returns:
            Path to preprocessed audio file
        """
        try:
            self.logger.info(f"Preprocessing audio file: {audio_path}")

            # Load audio
            audio, sr = self._load_audio(audio_path)

            # Apply preprocessing pipeline
            processed_audio = self._apply_preprocessing_pipeline(audio, sr)

            # Save processed audio
            if output_path is None:
                output_path = self._generate_output_path(audio_path)

            self._save_audio(
                processed_audio, output_path, self.config.target_sample_rate
            )

            self.logger.info(f"Successfully preprocessed: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error preprocessing {audio_path}: {str(e)}")
            raise

    def preprocess_directory(
        self, input_dir: str, output_dir: str
    ) -> List[str]:
        """
        Preprocess all audio files in a directory.

        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for processed files

        Returns:
            List of paths to processed audio files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        processed_files = []
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

        for audio_file in input_path.rglob("*"):
            if audio_file.suffix.lower() in audio_extensions:
                try:
                    relative_path = audio_file.relative_to(input_path)
                    output_file = output_path / relative_path.with_suffix(
                        f".{self.config.target_format}"
                    )
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    processed_path = self.preprocess_audio_file(
                        str(audio_file), str(output_file)
                    )
                    processed_files.append(processed_path)

                except Exception as e:
                    self.logger.error(
                        f"Failed to process {audio_file}: {str(e)}"
                    )
                    continue

        return processed_files

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with proper error handling."""
        try:
            # Try librosa first
            audio, sr = librosa.load(audio_path, sr=None, mono=False)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)

            return audio, sr

        except Exception as e:
            self.logger.warning(f"Librosa failed, trying torchaudio: {str(e)}")
            try:
                # Fallback to torchaudio
                audio, sr = torchaudio.load(audio_path)
                audio = audio.numpy()

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=0)

                return audio, sr

            except Exception as e2:
                self.logger.error(f"Both loaders failed: {str(e2)}")
                raise

    def _apply_preprocessing_pipeline(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Apply the complete preprocessing pipeline."""
        # Resample if needed
        if sr != self.config.target_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.config.target_sample_rate
            )

        # Duration filtering
        duration = len(audio) / self.config.target_sample_rate
        if (
            duration < self.config.min_duration
            or duration > self.config.max_duration
        ):
            raise ValueError(
                f"Audio duration {duration:.2f}s outside allowed range [{self.config.min_duration}, {self.config.max_duration}]"
            )

        # Amplitude filtering
        if np.max(np.abs(audio)) < self.config.min_amplitude:
            raise ValueError(
                f"Audio amplitude too low: {np.max(np.abs(audio)):.4f}"
            )

        # Silence trimming
        if self.config.enable_silence_trimming:
            audio = self._trim_silence(audio)

        # Noise reduction
        if self.config.enable_noise_reduction:
            audio = self._reduce_noise(audio)

        # Normalization
        if self.config.enable_normalization:
            audio = self._normalize_audio(audio)

        # Farsi-specific optimizations
        if self.config.enable_persian_optimization:
            audio = self._apply_persian_optimizations(audio)

        return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        # Find non-silent regions
        non_silent = np.where(np.abs(audio) > self.config.silence_threshold)[0]

        if len(non_silent) == 0:
            return audio

        start = non_silent[0]
        end = non_silent[-1]

        return audio[start : end + 1]

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction using spectral gating."""
        # Simple spectral gating noise reduction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)

        # Estimate noise from first few frames
        noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)

        # Apply spectral gating
        gate_threshold = noise_estimate * (
            1 + self.config.noise_reduction_strength
        )
        magnitude_gated = np.where(
            magnitude > gate_threshold, magnitude, gate_threshold
        )

        # Reconstruct audio
        stft_gated = magnitude_gated * np.exp(1j * np.angle(stft))
        audio_gated = librosa.istft(stft_gated)

        return audio_gated

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target loudness."""
        # Peak normalization
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95

        # Loudness normalization (RMS-based)
        target_rms = 10 ** (self.config.target_db / 20)
        current_rms = np.sqrt(np.mean(audio**2))

        if current_rms > 0:
            audio = audio * (target_rms / current_rms)

        return audio

    def _apply_persian_optimizations(self, audio: np.ndarray) -> np.ndarray:
        """Apply Farsi-specific audio optimizations."""
        # Enhance speech frequencies (80Hz - 8kHz)
        stft = librosa.stft(audio)
        freqs = librosa.fft_frequencies(sr=self.config.target_sample_rate)

        # Create speech enhancement filter
        speech_mask = np.ones_like(stft)
        speech_mask[(freqs < 80) | (freqs > 8000)] = 0.5

        # Apply filter
        stft_enhanced = stft * speech_mask
        audio_enhanced = librosa.istft(stft_enhanced)

        return audio_enhanced

    def _generate_output_path(self, input_path: str) -> str:
        """Generate output path for processed audio."""
        input_path_obj = Path(input_path)
        output_dir = input_path_obj.parent / "processed"
        output_dir.mkdir(exist_ok=True)

        return str(
            output_dir
            / f"{input_path_obj.stem}_processed.{self.config.target_format}"
        )

    def _save_audio(
        self, audio: np.ndarray, output_path: str, sample_rate: int
    ):
        """Save processed audio to file."""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, audio, sample_rate)

    def create_preprocessing_report(
        self, input_files: List[str], output_files: List[str]
    ) -> Dict:
        """Create a report of the preprocessing results."""
        report = {
            "total_files": len(input_files),
            "successful_files": len(output_files),
            "failed_files": len(input_files) - len(output_files),
            "success_rate": (
                len(output_files) / len(input_files) if input_files else 0
            ),
            "input_formats": {},
            "output_formats": {},
            "processing_stats": {},
        }

        # Analyze input formats
        for file_path in input_files:
            ext = Path(file_path).suffix.lower()
            report["input_formats"][ext] = (
                report["input_formats"].get(ext, 0) + 1
            )

        # Analyze output formats
        for file_path in output_files:
            ext = Path(file_path).suffix.lower()
            report["output_formats"][ext] = (
                report["output_formats"].get(ext, 0) + 1
            )

        return report

"""
Unified Audio Preprocessor for Persian Transcription
Consolidates functionality from multiple audio preprocessor modules.
Implements comprehensive audio preprocessing with Persian-specific optimizations.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import warnings
from dataclasses import dataclass
from enum import Enum

# Core dependencies
try:
    import librosa
    import scipy.signal
    from scipy.stats import entropy
    from scipy.io import wavfile
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False

# Audio processing
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Optional dependencies with fallbacks
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Facebook Denoiser (optional)
try:
    import torchaudio
    from denoiser import pretrained
    from denoiser.dsp import convert_audio
    FACEBOOK_DENOISER_AVAILABLE = True
except ImportError:
    FACEBOOK_DENOISER_AVAILABLE = False


class AudioQuality(Enum):
    """Audio quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class AudioMetrics:
    """Audio quality metrics."""
    snr_db: float
    spectral_entropy: float
    zero_crossing_rate: float
    rms_energy: float
    quality_level: AudioQuality
    duration: float
    sample_rate: int
    channels: int


class PersianFrequencyOptimizer:
    """Persian-specific frequency optimization."""
    
    # Persian phoneme frequency ranges (Hz)
    PERSIAN_FORMANTS = {
        'vowels': [(300, 1000), (1000, 2500), (2500, 3500)],  # F1, F2, F3
        'consonants': [(2000, 8000), (4000, 12000)],         # Fricatives, stops
        'nasals': [(200, 500), (1000, 2000)],                # م، ن، نگ
        'liquids': [(300, 1500)],                             # ل، ر
    }
    
    @staticmethod
    def create_persian_emphasis_filter(sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create frequency emphasis filter for Persian speech."""
        nyquist = sample_rate // 2
        
        # Multi-band emphasis for Persian phonemes
        frequencies = [200, 300, 800, 1500, 2500, 4000, 6000, nyquist]
        gains = [0.5, 1.0, 1.2, 1.3, 1.1, 0.9, 0.6, 0.3]  # Persian-optimized
        
        # Design filter
        b = scipy.signal.firwin2(101, frequencies, gains, fs=sample_rate)
        a = np.array([1.0])
        
        return b, a
    
    @staticmethod
    def enhance_persian_consonants(audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance Persian consonant clarity."""
        # Emphasis for Persian fricatives (ش، ژ، خ، غ)
        high_freq_b, high_freq_a = scipy.signal.butter(4, [3000, 7000], btype='band', fs=sample_rate)
        consonant_enhanced = scipy.signal.filtfilt(high_freq_b, high_freq_a, audio_samples)
        
        # Blend with original
        return 0.7 * audio_samples + 0.3 * consonant_enhanced


class FacebookDenoiser:
    """Facebook Denoiser integration."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cpu')  # Force CPU for compatibility
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if FACEBOOK_DENOISER_AVAILABLE:
            try:
                self.model = pretrained.dns64().to(self.device)
                self.model.eval()
                self.logger.info("Facebook Denoiser loaded successfully")
            except Exception as e:
                self.logger.warning(f"Facebook Denoiser initialization failed: {e}")
                self.model = None
    
    def denoise(self, audio: AudioSegment) -> AudioSegment:
        """Apply Facebook Denoiser."""
        if not self.model:
            return audio
        
        try:
            # Convert to tensor
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            
            wav = torch.from_numpy(samples).unsqueeze(0).to(self.device)
            
            # Ensure correct sample rate
            if audio.frame_rate != 16000:
                wav = torchaudio.functional.resample(wav, audio.frame_rate, 16000)
            
            # Apply denoising
            with torch.no_grad():
                wav = convert_audio(wav, 16000, self.model.sample_rate, self.model.chin)
                denoised = self.model(wav.unsqueeze(0)).squeeze(0)
            
            # Convert back
            denoised_samples = denoised.cpu().numpy().flatten()
            denoised_samples = np.clip(denoised_samples * 32768, -32768, 32767).astype(np.int16)
            
            return AudioSegment(
                denoised_samples.tobytes(),
                frame_rate=16000,
                sample_width=2,
                channels=1
            )
            
        except Exception as e:
            self.logger.warning(f"Facebook denoising failed: {e}")
            return audio


class AudioQualityAssessor:
    """Adaptive audio quality assessment."""
    
    @staticmethod
    def assess_quality(audio: AudioSegment) -> AudioMetrics:
        """Comprehensive audio quality assessment."""
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            
            samples = samples / 32768.0  # Normalize
            
            # SNR estimation
            signal_power = np.mean(samples ** 2)
            noise_floor = np.percentile(np.abs(samples), 10) ** 2
            snr_db = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
            
            # Spectral entropy
            freqs, psd = scipy.signal.welch(samples, fs=audio.frame_rate)
            psd_norm = psd / np.sum(psd)
            spectral_entropy = entropy(psd_norm)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(samples))
            
            # RMS energy
            rms_energy = np.sqrt(np.mean(samples ** 2))
            
            # Quality classification
            if snr_db > 20 and spectral_entropy < 8:
                quality = AudioQuality.EXCELLENT
            elif snr_db > 15 and spectral_entropy < 10:
                quality = AudioQuality.GOOD
            elif snr_db > 10:
                quality = AudioQuality.FAIR
            else:
                quality = AudioQuality.POOR
            
            return AudioMetrics(
                snr_db=snr_db,
                spectral_entropy=spectral_entropy,
                zero_crossing_rate=zcr,
                rms_energy=rms_energy,
                quality_level=quality,
                duration=len(audio) / 1000.0,
                sample_rate=audio.frame_rate,
                channels=audio.channels
            )
            
        except Exception:
            # Fallback assessment
            return AudioMetrics(15.0, 9.0, 0.1, 0.05, AudioQuality.FAIR, 0.0, 16000, 1)


class UnifiedAudioPreprocessor:
    """
    Unified audio preprocessor that consolidates functionality from multiple modules.
    Implements comprehensive audio preprocessing with Persian-specific optimizations.
    """
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.target_sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize capabilities
        self._setup_capabilities()
        
        # Initialize components
        if self.enable_facebook_denoiser:
            self.facebook_denoiser = FacebookDenoiser()
        
        if self.enable_persian_optimization:
            self.persian_optimizer = PersianFrequencyOptimizer()
        
        self.quality_assessor = AudioQualityAssessor()
        
        # Initialize VAD if available
        if self.enable_vad:
            try:
                self.vad = webrtcvad.Vad(2)
                self.vad_available = True
            except Exception as e:
                self.logger.warning(f"VAD initialization failed: {e}")
                self.vad_available = False
        else:
            self.vad_available = False
        
        self._log_capabilities()
    
    def _setup_capabilities(self):
        """Setup preprocessing capabilities based on available dependencies."""
        self.enable_noise_reduction = self.config.enable_noise_reduction and NOISE_REDUCE_AVAILABLE
        self.enable_vad = self.config.enable_voice_activity_detection and VAD_AVAILABLE
        self.enable_speech_enhancement = self.config.enable_speech_enhancement and ADVANCED_PROCESSING_AVAILABLE
        self.enable_facebook_denoiser = self.config.enable_facebook_denoiser and FACEBOOK_DENOISER_AVAILABLE
        self.enable_persian_optimization = self.config.enable_persian_optimization and ADVANCED_PROCESSING_AVAILABLE
        self.adaptive_processing = self.config.adaptive_processing
    
    def _log_capabilities(self):
        """Log preprocessing capabilities."""
        caps = []
        if self.enable_noise_reduction:
            caps.append("Noise Reduction")
        if self.enable_vad:
            caps.append("Voice Activity Detection")
        if self.enable_speech_enhancement:
            caps.append("Speech Enhancement")
        if self.enable_facebook_denoiser:
            caps.append("Facebook Denoiser")
        if self.enable_persian_optimization:
            caps.append("Persian Optimization")
        if self.adaptive_processing:
            caps.append("Adaptive Processing")
        
        self.logger.info(f"Unified audio preprocessing: {', '.join(caps) if caps else 'Basic mode'}")
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Comprehensive audio preprocessing pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed_audio, metadata)
        """
        self.logger.info(f"Starting unified preprocessing of: {audio_path}")
        
        # Load audio
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(audio_path)
        else:
            raise RuntimeError("Pydub not available for audio loading")
        
        # Generate metadata
        metadata = {
            'original_path': audio_path,
            'original_sample_rate': audio.frame_rate,
            'original_duration': len(audio) / 1000.0,
            'original_channels': audio.channels,
            'quality_issues': []
        }
        
        # Quality assessment
        if self.adaptive_processing:
            metrics = self.quality_assessor.assess_quality(audio)
            self.logger.info(f"Audio quality: {metrics.quality_level.value} (SNR: {metrics.snr_db:.1f}dB)")
            metadata['quality_metrics'] = {
                'snr_db': metrics.snr_db,
                'quality_level': metrics.quality_level.value,
                'spectral_entropy': metrics.spectral_entropy
            }
        
        # Multi-stage preprocessing
        processed_audio = audio
        
        # Stage 1: Format optimization
        processed_audio = self._optimize_format(processed_audio)
        
        # Stage 2: Advanced denoising
        if self.enable_facebook_denoiser and (not metrics or metrics.quality_level in [AudioQuality.FAIR, AudioQuality.POOR]):
            self.logger.info("Applying Facebook Denoiser...")
            processed_audio = self.facebook_denoiser.denoise(processed_audio)
        elif self.enable_noise_reduction:
            processed_audio = self._reduce_noise(processed_audio)
        
        # Stage 3: Persian-specific enhancement
        if self.enable_persian_optimization:
            processed_audio = self._apply_persian_optimization(processed_audio)
        
        # Stage 4: Speech enhancement
        if self.enable_speech_enhancement:
            processed_audio = self._enhance_speech(processed_audio)
        
        # Stage 5: Final normalization
        processed_audio = self._final_normalization(processed_audio)
        
        # Convert to numpy array
        audio_data = self._audio_segment_to_numpy(processed_audio)
        
        self.logger.info(f"Unified preprocessing completed. Final duration: {len(audio_data) / self.sample_rate:.2f}s")
        
        return audio_data, metadata
    
    def _optimize_format(self, audio: AudioSegment) -> AudioSegment:
        """Optimize audio format for speech transcription."""
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Optimize sample rate for Whisper
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
        
        # Remove DC offset
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples - np.mean(samples)
        
        # Convert back to AudioSegment
        samples_int = np.clip(samples, -32768, 32767).astype(np.int16)
        return AudioSegment(
            samples_int.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
    
    def _reduce_noise(self, audio: AudioSegment) -> AudioSegment:
        """Apply noise reduction using noisereduce library."""
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize for processing
            samples = samples / 32768.0
            
            # Apply noise reduction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced_samples = nr.reduce_noise(
                    y=samples,
                    sr=self.sample_rate,
                    stationary=False,  # Better for varying background noise
                    prop_decrease=0.8  # Conservative noise reduction
                )
            
            # Convert back to int16
            reduced_samples = np.clip(reduced_samples * 32768, -32768, 32767).astype(np.int16)
            
            return AudioSegment(
                reduced_samples.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _apply_persian_optimization(self, audio: AudioSegment) -> AudioSegment:
        """Apply Persian-specific optimizations."""
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            # Persian frequency emphasis
            b, a = self.persian_optimizer.create_persian_emphasis_filter(self.sample_rate)
            emphasized_samples = scipy.signal.filtfilt(b, a, samples)
            
            # Persian consonant enhancement
            consonant_enhanced = self.persian_optimizer.enhance_persian_consonants(emphasized_samples, self.sample_rate)
            
            # Blend enhancements
            final_samples = 0.6 * samples + 0.3 * emphasized_samples + 0.1 * consonant_enhanced
            final_samples = np.clip(final_samples * 32768, -32768, 32767).astype(np.int16)
            
            return AudioSegment(final_samples.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=1)
            
        except Exception as e:
            self.logger.warning(f"Persian optimization failed: {e}")
            return audio
    
    def _enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """Enhance speech frequencies for better transcription."""
        if not ADVANCED_PROCESSING_AVAILABLE:
            return audio
        
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0
            
            # Apply speech frequency emphasis (300Hz - 3400Hz)
            nyquist = self.sample_rate // 2
            low_freq = 300 / nyquist
            high_freq = 3400 / nyquist
            
            # Butterworth bandpass filter
            b, a = scipy.signal.butter(4, [low_freq, high_freq], btype='band')
            
            # Apply filter with some of the original signal mixed in
            filtered_samples = scipy.signal.filtfilt(b, a, samples)
            enhanced_samples = 0.7 * samples + 0.3 * filtered_samples
            
            # Gentle compression for dynamic range
            enhanced_samples = np.tanh(enhanced_samples * 1.2)
            
            # Convert back to int16
            enhanced_samples = np.clip(enhanced_samples * 32768, -32768, 32767).astype(np.int16)
            
            return AudioSegment(
                enhanced_samples.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            
        except Exception as e:
            self.logger.warning(f"Speech enhancement failed: {e}")
            return audio
    
    def _final_normalization(self, audio: AudioSegment) -> AudioSegment:
        """Apply final RMS normalization for consistent volume."""
        try:
            # Target RMS level (adjust based on preference)
            target_rms_db = -20.0
            
            # Get current RMS with fallback for older pydub versions
            try:
                current_rms_db = audio.rms_db if hasattr(audio, 'rms_db') else audio.dBFS
            except:
                current_rms_db = -30.0  # Fallback value
            
            # Calculate gain adjustment
            gain_adjustment = target_rms_db - current_rms_db
            
            # Limit gain adjustment to prevent clipping
            gain_adjustment = max(-20, min(20, gain_adjustment))
            
            # Apply gain if significant adjustment needed
            if abs(gain_adjustment) > 1.0:
                audio = audio + gain_adjustment
            
            return audio
            
        except Exception as e:
            self.logger.warning(f"Final normalization failed: {e}")
            return audio
    
    def _audio_segment_to_numpy(self, audio: AudioSegment) -> np.ndarray:
        """Convert AudioSegment to numpy array."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples / 32768.0
    
    def detect_voice_activity(self, audio: AudioSegment, frame_duration_ms: int = 30) -> List[Tuple[int, int]]:
        """Detect voice activity and return speech segments."""
        if not self.vad_available:
            # Return entire audio as single segment if VAD unavailable
            return [(0, len(audio))]
        
        try:
            # Convert audio to format expected by VAD (16kHz, 16-bit, mono)
            vad_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Get raw audio data
            raw_samples = vad_audio.raw_data
            
            # Process in frames
            frame_size = int(16000 * frame_duration_ms / 1000) * 2  # 2 bytes per sample
            speech_segments = []
            current_segment_start = None
            
            for i in range(0, len(raw_samples), frame_size):
                frame = raw_samples[i:i + frame_size]
                
                # Pad frame if necessary
                if len(frame) < frame_size:
                    frame += b'\x00' * (frame_size - len(frame))
                
                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame, 16000)
                    
                    frame_start_ms = int(i * 1000 / (16000 * 2))
                    
                    if is_speech and current_segment_start is None:
                        current_segment_start = frame_start_ms
                    elif not is_speech and current_segment_start is not None:
                        # End of speech segment
                        speech_segments.append((current_segment_start, frame_start_ms))
                        current_segment_start = None
                        
                except Exception:
                    # If VAD fails for this frame, assume it's speech
                    continue
            
            # Handle case where audio ends during speech
            if current_segment_start is not None:
                speech_segments.append((current_segment_start, len(audio)))
            
            # Merge segments that are very close together (< 500ms gap)
            merged_segments = []
            for start, end in speech_segments:
                if merged_segments and start - merged_segments[-1][1] < 500:
                    # Merge with previous segment
                    merged_segments[-1] = (merged_segments[-1][0], end)
                else:
                    merged_segments.append((start, end))
            
            # Filter out very short segments (< 1 second)
            final_segments = [(s, e) for s, e in merged_segments if e - s >= 1000]
            
            if not final_segments:
                # If no speech detected, return entire audio
                return [(0, len(audio))]
            
            self.logger.info(f"Detected {len(final_segments)} speech segments")
            return final_segments
            
        except Exception as e:
            self.logger.warning(f"Voice activity detection failed: {e}")
            return [(0, len(audio))]
    
    def create_smart_chunks(self, audio: AudioSegment, 
                           target_chunk_duration_ms: int = 25000,
                           overlap_ms: int = 200) -> List[Tuple[int, int]]:
        """Create intelligent chunks based on voice activity and target duration."""
        try:
            # Get voice activity segments
            speech_segments = self.detect_voice_activity(audio)
            
            smart_chunks = []
            
            for segment_start, segment_end in speech_segments:
                segment_duration = segment_end - segment_start
                
                if segment_duration <= target_chunk_duration_ms:
                    # Segment fits in one chunk
                    smart_chunks.append((segment_start, segment_end))
                else:
                    # Split long segments into smaller chunks
                    current_pos = segment_start
                    
                    while current_pos < segment_end:
                        chunk_end = min(current_pos + target_chunk_duration_ms, segment_end)
                        smart_chunks.append((current_pos, chunk_end))
                        
                        # Move to next chunk with overlap
                        current_pos = chunk_end - overlap_ms
                        
                        # Avoid very small remaining chunks
                        if segment_end - current_pos < target_chunk_duration_ms * 0.3:
                            break
            
            self.logger.info(f"Created {len(smart_chunks)} intelligent chunks")
            return smart_chunks
            
        except Exception as e:
            self.logger.warning(f"Smart chunking failed: {e}")
            # Fallback to regular chunking
            return self._fallback_chunking(audio, target_chunk_duration_ms, overlap_ms)
    
    def _fallback_chunking(self, audio: AudioSegment, chunk_duration_ms: int, overlap_ms: int) -> List[Tuple[int, int]]:
        """Fallback chunking method."""
        chunks = []
        audio_length = len(audio)
        current_pos = 0
        
        while current_pos < audio_length:
            end_pos = min(current_pos + chunk_duration_ms, audio_length)
            chunks.append((current_pos, end_pos))
            
            if end_pos == audio_length:
                break
                
            current_pos = end_pos - overlap_ms
            
        return chunks
    
    def save_processed_audio(self, audio_data: np.ndarray, output_path: str) -> None:
        """Save processed audio as 16kHz mono WAV."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as 16kHz mono WAV
        if SOUNDFILE_AVAILABLE:
            sf.write(
                output_path,
                audio_data,
                self.sample_rate,
                subtype='PCM_16'
            )
        else:
            # Fallback to scipy
            wavfile.write(output_path, self.sample_rate, audio_data)
        
        self.logger.info(f"Saved processed audio to: {output_path}")


def get_unified_preprocessing_capabilities() -> Dict[str, bool]:
    """Return unified preprocessing capabilities."""
    return {
        'noise_reduction': NOISE_REDUCE_AVAILABLE,
        'voice_activity_detection': VAD_AVAILABLE,
        'speech_enhancement': ADVANCED_PROCESSING_AVAILABLE,
        'facebook_denoiser': FACEBOOK_DENOISER_AVAILABLE,
        'persian_optimization': ADVANCED_PROCESSING_AVAILABLE,
        'pydub': PYDUB_AVAILABLE,
        'soundfile': SOUNDFILE_AVAILABLE
    }


def create_unified_preprocessor(config) -> UnifiedAudioPreprocessor:
    """Create unified audio preprocessor."""
    return UnifiedAudioPreprocessor(config) 
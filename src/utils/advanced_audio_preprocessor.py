"""
Advanced audio preprocessing with Facebook Denoiser and Persian-specific optimizations.
"""

import logging
import numpy as np
import warnings
from typing import Optional, Tuple, List, Dict
from pydub import AudioSegment
from dataclasses import dataclass
from enum import Enum

# Core dependencies
try:
    import librosa
    import scipy.signal
    from scipy.stats import entropy
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False

# Facebook Denoiser (optional)
try:
    import torch
    import torchaudio
    from denoiser import pretrained
    from denoiser.dsp import convert_audio
    FACEBOOK_DENOISER_AVAILABLE = True
except ImportError:
    FACEBOOK_DENOISER_AVAILABLE = False

# Basic preprocessing
try:
    import noisereduce as nr
    import webrtcvad
    BASIC_PREPROCESSING_AVAILABLE = True
except ImportError:
    BASIC_PREPROCESSING_AVAILABLE = False


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
            
            return AudioMetrics(snr_db, spectral_entropy, zcr, rms_energy, quality)
            
        except Exception:
            # Fallback assessment
            return AudioMetrics(15.0, 9.0, 0.1, 0.05, AudioQuality.FAIR)


class AdvancedAudioPreprocessor:
    """Advanced multi-stage preprocessing pipeline with Persian optimization."""
    
    def __init__(self, 
                 enable_facebook_denoiser: bool = True,
                 enable_persian_optimization: bool = True,
                 adaptive_processing: bool = True,
                 sample_rate: int = 16000):
        
        self.sample_rate = sample_rate
        self.enable_facebook_denoiser = enable_facebook_denoiser and FACEBOOK_DENOISER_AVAILABLE
        self.enable_persian_optimization = enable_persian_optimization and ADVANCED_PROCESSING_AVAILABLE
        self.adaptive_processing = adaptive_processing
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        if self.enable_facebook_denoiser:
            self.facebook_denoiser = FacebookDenoiser()
        
        if self.enable_persian_optimization:
            self.persian_optimizer = PersianFrequencyOptimizer()
        
        self.quality_assessor = AudioQualityAssessor()
        
        # Initialize basic preprocessing fallback
        if BASIC_PREPROCESSING_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(2)
                self.vad_available = True
            except:
                self.vad_available = False
        else:
            self.vad_available = False
        
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log preprocessing capabilities."""
        caps = []
        if self.enable_facebook_denoiser:
            caps.append("Facebook Denoiser")
        if self.enable_persian_optimization:
            caps.append("Persian Optimization")
        if self.adaptive_processing:
            caps.append("Adaptive Processing")
        if self.vad_available:
            caps.append("Voice Activity Detection")
        
        self.logger.info(f"Advanced preprocessing: {', '.join(caps) if caps else 'Basic mode'}")
    
    def preprocess_audio(self, audio: AudioSegment) -> AudioSegment:
        """Multi-stage preprocessing pipeline."""
        try:
            processed_audio = audio
            
            # Stage 1: Quality assessment
            if self.adaptive_processing:
                metrics = self.quality_assessor.assess_quality(processed_audio)
                self.logger.info(f"Audio quality: {metrics.quality_level.value} (SNR: {metrics.snr_db:.1f}dB)")
            else:
                metrics = None
            
            # Stage 2: Format optimization
            processed_audio = self._optimize_format(processed_audio)
            
            # Stage 3: Advanced denoising
            if self.enable_facebook_denoiser and (not metrics or metrics.quality_level in [AudioQuality.FAIR, AudioQuality.POOR]):
                self.logger.info("Applying Facebook Denoiser...")
                processed_audio = self.facebook_denoiser.denoise(processed_audio)
            elif BASIC_PREPROCESSING_AVAILABLE:
                # Fallback to basic noise reduction
                processed_audio = self._basic_noise_reduction(processed_audio)
            
            # Stage 4: Persian-specific enhancement
            if self.enable_persian_optimization:
                processed_audio = self._apply_persian_optimization(processed_audio)
            
            # Stage 5: Final normalization and dynamics
            processed_audio = self._final_enhancement(processed_audio, metrics)
            
            self.logger.info("Advanced preprocessing completed")
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Advanced preprocessing failed: {e}")
            return audio
    
    def _optimize_format(self, audio: AudioSegment) -> AudioSegment:
        """Optimize audio format."""
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
        
        # Remove DC offset
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples - np.mean(samples)
        samples_int = np.clip(samples, -32768, 32767).astype(np.int16)
        
        return AudioSegment(samples_int.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=1)
    
    def _basic_noise_reduction(self, audio: AudioSegment) -> AudioSegment:
        """Fallback noise reduction."""
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced_samples = nr.reduce_noise(y=samples, sr=self.sample_rate, stationary=False, prop_decrease=0.7)
            
            reduced_samples = np.clip(reduced_samples * 32768, -32768, 32767).astype(np.int16)
            return AudioSegment(reduced_samples.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=1)
            
        except Exception as e:
            self.logger.warning(f"Basic noise reduction failed: {e}")
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
    
    def _final_enhancement(self, audio: AudioSegment, metrics: Optional[AudioMetrics]) -> AudioSegment:
        """Final enhancement based on quality metrics."""
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            # Adaptive dynamics processing
            if metrics and metrics.quality_level == AudioQuality.POOR:
                # Aggressive processing for poor quality
                samples = np.tanh(samples * 2.0)  # Soft limiting
                samples = self._apply_expander(samples, threshold=0.1)
            elif metrics and metrics.quality_level == AudioQuality.EXCELLENT:
                # Gentle processing for good quality
                samples = samples * 1.1  # Slight boost
            else:
                # Moderate processing
                samples = np.tanh(samples * 1.5)
            
            # Final normalization
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples * (0.95 / peak)
            
            final_samples = np.clip(samples * 32768, -32768, 32767).astype(np.int16)
            return AudioSegment(final_samples.tobytes(), frame_rate=self.sample_rate, sample_width=2, channels=1)
            
        except Exception as e:
            self.logger.warning(f"Final enhancement failed: {e}")
            return audio
    
    def _apply_expander(self, samples: np.ndarray, threshold: float = 0.1, ratio: float = 2.0) -> np.ndarray:
        """Apply upward expansion to reduce noise floor."""
        abs_samples = np.abs(samples)
        mask = abs_samples < threshold
        
        expanded_samples = samples.copy()
        expanded_samples[mask] = samples[mask] * (abs_samples[mask] / threshold) ** (ratio - 1)
        
        return expanded_samples
    
    def create_intelligent_chunks(self, audio: AudioSegment, target_duration_ms: int = 25000, overlap_ms: int = 200) -> List[Tuple[int, int]]:
        """Create chunks with advanced VAD and quality-based segmentation."""
        if not self.vad_available:
            return self._fallback_chunking(audio, target_duration_ms, overlap_ms)
        
        try:
            # Quality-based chunk sizing
            metrics = self.quality_assessor.assess_quality(audio)
            
            if metrics.quality_level == AudioQuality.POOR:
                target_duration_ms = int(target_duration_ms * 0.8)  # Shorter chunks for poor quality
            elif metrics.quality_level == AudioQuality.EXCELLENT:
                target_duration_ms = int(target_duration_ms * 1.2)  # Longer chunks for good quality
            
            # Advanced VAD-based chunking
            speech_segments = self._detect_speech_segments(audio)
            chunks = self._optimize_chunk_boundaries(speech_segments, target_duration_ms, overlap_ms)
            
            self.logger.info(f"Created {len(chunks)} intelligent chunks (quality: {metrics.quality_level.value})")
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Intelligent chunking failed: {e}")
            return self._fallback_chunking(audio, target_duration_ms, overlap_ms)
    
    def _detect_speech_segments(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """Advanced speech detection with quality awareness."""
        vad_audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        raw_samples = vad_audio.raw_data
        
        frame_duration_ms = 30
        frame_size = int(16000 * frame_duration_ms / 1000) * 2
        
        speech_segments = []
        current_segment_start = None
        consecutive_speech = 0
        consecutive_silence = 0
        
        for i in range(0, len(raw_samples), frame_size):
            frame = raw_samples[i:i + frame_size]
            if len(frame) < frame_size:
                frame += b'\x00' * (frame_size - len(frame))
            
            try:
                is_speech = self.vad.is_speech(frame, 16000)
                frame_start_ms = int(i * 1000 / (16000 * 2))
                
                if is_speech:
                    consecutive_speech += 1
                    consecutive_silence = 0
                    
                    if current_segment_start is None and consecutive_speech >= 2:
                        current_segment_start = frame_start_ms
                else:
                    consecutive_silence += 1
                    consecutive_speech = 0
                    
                    if current_segment_start is not None and consecutive_silence >= 5:
                        speech_segments.append((current_segment_start, frame_start_ms))
                        current_segment_start = None
                        
            except Exception:
                continue
        
        if current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio)))
        
        # Merge close segments
        merged_segments = []
        for start, end in speech_segments:
            if merged_segments and start - merged_segments[-1][1] < 800:
                merged_segments[-1] = (merged_segments[-1][0], end)
            else:
                merged_segments.append((start, end))
        
        return [(s, e) for s, e in merged_segments if e - s >= 1500]
    
    def _optimize_chunk_boundaries(self, speech_segments: List[Tuple[int, int]], 
                                 target_duration_ms: int, overlap_ms: int) -> List[Tuple[int, int]]:
        """Optimize chunk boundaries based on speech segments."""
        chunks = []
        
        for segment_start, segment_end in speech_segments:
            segment_duration = segment_end - segment_start
            
            if segment_duration <= target_duration_ms:
                chunks.append((segment_start, segment_end))
            else:
                current_pos = segment_start
                while current_pos < segment_end:
                    chunk_end = min(current_pos + target_duration_ms, segment_end)
                    chunks.append((current_pos, chunk_end))
                    current_pos = chunk_end - overlap_ms
                    
                    if segment_end - current_pos < target_duration_ms * 0.4:
                        break
        
        return chunks
    
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


def get_advanced_preprocessing_capabilities() -> Dict[str, bool]:
    """Return advanced preprocessing capabilities."""
    return {
        'facebook_denoiser': FACEBOOK_DENOISER_AVAILABLE,
        'persian_optimization': ADVANCED_PROCESSING_AVAILABLE,
        'quality_assessment': ADVANCED_PROCESSING_AVAILABLE,
        'basic_preprocessing': BASIC_PREPROCESSING_AVAILABLE
    }

"""
Enhanced audio preprocessing module with noise reduction and voice activity detection.
Optimized for Persian speech transcription quality improvement.
"""

import logging
import numpy as np
from typing import Optional, Tuple, List
from pydub import AudioSegment
import warnings

# Optional imports with fallbacks
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
    import librosa
    import scipy.signal
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False


class AudioPreprocessor:
    """High-performance audio preprocessing with Persian speech optimization."""
    
    def __init__(self, 
                 enable_noise_reduction: bool = True,
                 enable_vad: bool = True,
                 enable_speech_enhancement: bool = True,
                 sample_rate: int = 16000):
        
        self.enable_noise_reduction = enable_noise_reduction and NOISE_REDUCE_AVAILABLE
        self.enable_vad = enable_vad and VAD_AVAILABLE
        self.enable_speech_enhancement = enable_speech_enhancement and ADVANCED_PROCESSING_AVAILABLE
        self.sample_rate = sample_rate
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize VAD if available
        if self.enable_vad:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
            except Exception as e:
                self.logger.warning(f"VAD initialization failed: {e}")
                self.enable_vad = False
        
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log available preprocessing capabilities."""
        capabilities = []
        if self.enable_noise_reduction:
            capabilities.append("Noise Reduction")
        if self.enable_vad:
            capabilities.append("Voice Activity Detection")
        if self.enable_speech_enhancement:
            capabilities.append("Speech Enhancement")
        
        if capabilities:
            self.logger.info(f"Audio preprocessing enabled: {', '.join(capabilities)}")
        else:
            self.logger.info("Audio preprocessing disabled - install dependencies for enhancement")
    
    def preprocess_audio(self, audio: AudioSegment) -> AudioSegment:
        """Apply comprehensive preprocessing pipeline."""
        try:
            processed_audio = audio
            
            # Step 1: Basic normalization and format optimization
            processed_audio = self._optimize_format(processed_audio)
            
            # Step 2: Noise reduction
            if self.enable_noise_reduction:
                processed_audio = self._reduce_noise(processed_audio)
            
            # Step 3: Speech enhancement
            if self.enable_speech_enhancement:
                processed_audio = self._enhance_speech(processed_audio)
            
            # Step 4: Final normalization
            processed_audio = self._final_normalization(processed_audio)
            
            self.logger.info("Audio preprocessing completed successfully")
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return audio  # Return original audio on failure
    
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
    
    def _enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """Enhance speech frequencies for better transcription."""
        if not ADVANCED_PROCESSING_AVAILABLE:
            return audio
        
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0
            
            # Apply speech frequency emphasis (300Hz - 3400Hz)
            # Design a bandpass filter for speech frequencies
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
            current_rms_db = audio.rms_db if audio.rms_db is not None else -30.0
            
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
    
    def detect_voice_activity(self, audio: AudioSegment, frame_duration_ms: int = 30) -> List[Tuple[int, int]]:
        """Detect voice activity and return speech segments."""
        if not self.enable_vad:
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


def get_preprocessing_capabilities() -> dict:
    """Return available preprocessing capabilities."""
    return {
        'noise_reduction': NOISE_REDUCE_AVAILABLE,
        'voice_activity_detection': VAD_AVAILABLE,
        'speech_enhancement': ADVANCED_PROCESSING_AVAILABLE
    }

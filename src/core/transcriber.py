"""
Unified high-performance audio transcriber with true streaming capabilities.
Replaces both UnifiedAudioTranscriber and OptimizedAudioTranscriber with a single optimized implementation.
"""

import os
import logging
import gc
import time
import subprocess
import threading
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import torch
import whisper
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import psutil
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import TranscriptionConfig
from ..preprocessing import UnifiedAudioPreprocessor
from ..utils.preprocessing_validator import validate_preprocessing

# Handle imports with fallback for direct execution
try:
    from ..utils.file_manager import TranscriptionFileManager
    from ..utils.repetition_detector import RepetitionDetector
    from ..utils.sentence_extractor import SentenceExtractor

    # Try to import pyannote diarizer first
    try:
        from ..utils.pyannote_diarizer import PyannoteDiarizer, SpeakerSegment

        PYAANOTE_AVAILABLE = True
    except ImportError:
        from ..utils.speaker_diarization import SpeakerDiarizer, SpeakerSegment

        PYAANOTE_AVAILABLE = False

except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.file_manager import TranscriptionFileManager
    from utils.repetition_detector import RepetitionDetector
    from utils.sentence_extractor import SentenceExtractor
    from utils.speaker_diarization import SpeakerDiarizer, SpeakerSegment


class DeviceManager:
    """Manages device selection and fallback mechanisms."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device_fallback_history = []
        self.current_device = self._detect_best_device()

    def _detect_best_device(self) -> str:
        """Detect the best available device with comprehensive testing."""
        devices = self._get_available_devices()

        for device in devices:
            if self._test_device(device):
                self.logger.info(f"✅ Device {device} is working properly")
                return device
            else:
                self.logger.warning(
                    f"❌ Device {device} failed compatibility test"
                )
                self.device_fallback_history.append(device)

        # If all devices fail, force CPU
        self.logger.warning(
            "⚠️ All devices failed compatibility tests, forcing CPU mode"
        )
        return "cpu"

    def _get_available_devices(self) -> List[str]:
        """Get list of available devices in order of preference."""
        devices = []

        # Check CUDA availability
        if torch.cuda.is_available():
            try:
                cuda_count = torch.cuda.device_count()
                for i in range(cuda_count):
                    devices.append(f"cuda:{i}")
                devices.append("cuda")  # Default CUDA device
            except Exception as e:
                self.logger.warning(f"CUDA device count failed: {e}")

        # Always add CPU as fallback
        devices.append("cpu")

        return devices

    def _test_device(self, device: str) -> bool:
        """Test if a device is working properly."""
        try:
            if device.startswith("cuda"):
                # Test CUDA operations
                test_tensor = torch.randn(2, 2, device=device)
                result = torch.matmul(test_tensor, test_tensor)
                torch.cuda.synchronize(device)
                return True
            elif device == "cpu":
                # CPU is always available
                return True
            else:
                return False
        except Exception as e:
            self.logger.debug(f"Device {device} test failed: {e}")
            return False

    def get_device(self) -> str:
        """Get current working device."""
        return self.current_device

    def force_cpu_fallback(self):
        """Force fallback to CPU mode with full resource utilization."""
        if self.current_device != "cpu":
            total_cores = os.cpu_count() or 4
            self.logger.warning(
                f"🔄 Forcing fallback to CPU mode due to device errors"
            )
            self.logger.info(
                f"🚀 CPU Fallback: Using {total_cores} CPU cores for maximum performance"
            )

            self.current_device = "cpu"
            # Update config with CPU optimizations
            self.config.device = "cpu"
            self.config.batch_size = 1
            self.config.num_workers = total_cores

            # Apply additional CPU optimizations
            self.config._apply_cpu_optimizations()

            self.logger.info(
                f"⚡ CPU Mode Activated: {total_cores} workers, optimized chunk size, parallel processing enabled"
            )

    def is_cuda_available(self) -> bool:
        """Check if CUDA is currently available and working."""
        return self.current_device.startswith("cuda") and self._test_device(
            self.current_device
        )


class UnifiedMemoryManager:
    """Lightweight, efficient memory manager."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.memory_limit_mb = config.memory_threshold_mb
        self.last_check = 0
        self.check_interval = 10  # Reduced from 5 seconds

    def check_and_cleanup(self) -> bool:
        """Check memory and cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return False

        self.last_check = current_time
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

        if memory_mb > self.memory_limit_mb:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False


class StreamingAudioProcessor:
    """True streaming audio processor using ffmpeg."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """Get audio info using ffprobe or fallback to pydub."""
        try:
            # Try ffprobe first (fastest)
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                file_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                format_info = data.get("format", {})
                duration = float(format_info.get("duration", 0))
                return {
                    "duration_seconds": duration,
                    "format": format_info.get("format_name", "unknown"),
                    "bit_rate": format_info.get("bit_rate", "unknown"),
                }
        except Exception as e:
            self.logger.debug(f"ffprobe failed: {e}")

        # Fallback to pydub
        try:
            audio = AudioSegment.from_file(file_path)
            return {
                "duration_seconds": len(audio) / 1000.0,
                "format": "fallback",
                "bit_rate": "unknown",
            }
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            raise

    def stream_chunks(
        self, file_path: str
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Stream audio in chunks with overlap."""
        audio_info = self.get_audio_info(file_path)
        duration_s = audio_info["duration_seconds"]
        chunk_duration_s = self.config.chunk_duration_ms / 1000
        overlap_s = self.config.overlap_ms / 1000

        chunk_index = 0
        current_time = 0

        while current_time < duration_s:
            chunk_start = max(0, current_time - overlap_s)
            chunk_end = min(duration_s, current_time + chunk_duration_s)

            chunk_array = self._extract_with_ffmpeg(
                file_path, chunk_start, chunk_end - chunk_start
            )

            if chunk_array is not None and len(chunk_array) > 0:
                yield chunk_index, chunk_array

            chunk_index += 1
            current_time = chunk_end

            # Memory management
            if chunk_index % 10 == 0:
                gc.collect()

    def _extract_with_ffmpeg(
        self, file_path: str, start_time: float, duration: float
    ) -> Optional[np.ndarray]:
        """Extract audio chunk using ffmpeg."""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                temp_path = temp_file.name

            cmd = [
                "ffmpeg",
                "-i",
                file_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-ar",
                str(self.config.target_sample_rate),
                "-ac",
                "1",
                "-f",
                "wav",
                "-y",
                temp_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                audio = AudioSegment.from_wav(temp_path)
                audio_array = np.array(audio.get_array_of_samples())
                os.unlink(temp_path)
                return audio_array
            else:
                self.logger.warning(
                    f"ffmpeg failed: {result.stderr}, falling back to pydub"
                )
                return self._extract_with_pydub(
                    file_path, start_time, duration
                )

        except Exception as e:
            self.logger.debug(f"ffmpeg extraction failed: {e}")
            return self._extract_with_pydub(file_path, start_time, duration)

    def _extract_with_pydub(
        self, file_path: str, start_time: float, duration: float
    ) -> Optional[np.ndarray]:
        """Extract audio chunk using pydub (fallback)."""
        try:
            audio = AudioSegment.from_file(file_path)
            start_ms = int(start_time * 1000)
            end_ms = int((start_time + duration) * 1000)
            chunk = audio[start_ms:end_ms]

            # Resample if needed
            if chunk.frame_rate != self.config.target_sample_rate:
                chunk = chunk.set_frame_rate(self.config.target_sample_rate)

            # Convert to mono if needed
            if chunk.channels > 1:
                chunk = chunk.set_channels(1)

            # Convert to float32 and normalize to [-1, 1] range
            samples = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            # Normalize based on the bit depth
            if chunk.sample_width == 1:  # 8-bit
                samples = samples / 128.0
            elif chunk.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif chunk.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0

            return samples

        except Exception as e:
            self.logger.error(f"pydub extraction failed: {e}")
            return None

    def stream_smart_chunks(
        self,
        file_path: str,
        segments_ms: List[Tuple[int, int]],
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Stream audio using precomputed smart segments in milliseconds.

        Args:
            file_path: Path to the audio file
            segments_ms: List of (start_ms, end_ms) tuples defining segments

        Yields:
            Tuples of (chunk_index, audio_array)
        """
        try:
            for idx, (start_ms, end_ms) in enumerate(segments_ms):
                start_s = max(0.0, float(start_ms) / 1000.0)
                duration_s = max(0.0, float(end_ms - start_ms) / 1000.0)

                if duration_s <= 0:
                    continue

                chunk_array = self._extract_with_ffmpeg(
                    file_path, start_s, duration_s
                )

                if chunk_array is None or len(chunk_array) == 0:
                    chunk_array = self._extract_with_pydub(
                        file_path, start_s, duration_s
                    )

                if chunk_array is not None and len(chunk_array) > 0:
                    yield idx, chunk_array

                # Periodic cleanup
                if idx % 10 == 0:
                    gc.collect()
        except Exception as e:
            self.logger.error(f"Smart streaming failed: {e}")


class OptimizedWhisperTranscriber:
    """Optimized Whisper transcriber with model caching and device fallback."""

    _model_cache = {}
    _model_lock = threading.Lock()

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device_manager = DeviceManager(config)
        self._model_cache = {}
        self._model_lock = threading.Lock()
        self.cuda_error_count = 0
        self.max_cuda_errors = 3

        # Configure retry strategy for Hugging Face downloads
        self._setup_hf_download_retry()

    def _setup_hf_download_retry(self):
        """Setup retry strategy for Hugging Face model downloads."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        # Configure session for Hugging Face downloads
        self._hf_session = requests.Session()
        self._hf_session.mount("https://", adapter)
        self._hf_session.mount("http://", adapter)

        # Set longer timeout for large model downloads
        self._hf_session.timeout = (30, 300)  # (connect_timeout, read_timeout)

    def _get_cached_model(self):
        """Get cached model or load new one with retry logic and device fallback."""
        current_device = self.device_manager.get_device()
        model_key = f"{self.config.model_name}_{current_device}"

        with self._model_lock:
            if model_key not in self._model_cache:
                self.logger.info(f"Loading model: {self.config.model_name}...")
                start_time = time.time()

                try:
                    if self.config.use_huggingface_model:
                        # Load Hugging Face model with retry logic
                        self.logger.info(
                            "Loading Hugging Face Whisper model..."
                        )

                        # Configure Hugging Face Hub to use our session
                        import huggingface_hub

                        try:
                            huggingface_hub.set_http_backend(self._hf_session)
                        except AttributeError:
                            # Fallback for older versions
                            self.logger.info(
                                "Using default Hugging Face Hub HTTP backend"
                            )

                        # Force consistent data type to avoid mismatches
                        torch_dtype = (
                            torch.float32
                        )  # Use float32 for compatibility
                        device_map = (
                            "auto"
                            if current_device.startswith("cuda")
                            else None
                        )

                        model = WhisperForConditionalGeneration.from_pretrained(
                            self.config.model_name,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            local_files_only=False,  # Allow download
                            resume_download=True,  # Resume interrupted downloads
                            trust_remote_code=True,  # Trust custom model code
                            low_cpu_mem_usage=True,  # Reduce memory usage
                        )
                        processor = WhisperProcessor.from_pretrained(
                            self.config.model_name,
                            local_files_only=False,
                            resume_download=True,
                        )

                        # Move model to device if not using device_map
                        if device_map is None:
                            model = model.to(current_device)

                        self._model_cache[model_key] = {
                            "model": model,
                            "processor": processor,
                        }
                    else:
                        # Load OpenAI Whisper model
                        model = whisper.load_model(
                            self.config.model_name, device=current_device
                        )
                        self._model_cache[model_key] = {
                            "model": model,
                            "processor": None,
                        }

                    self.logger.info(
                        f"Model loaded in {time.time() - start_time:.2f}s"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to load model: {e}")

                    # If CUDA fails, try CPU fallback
                    if current_device.startswith("cuda"):
                        self.logger.info(
                            "🔄 CUDA model loading failed, attempting CPU fallback..."
                        )
                        self.device_manager.force_cpu_fallback()
                        return (
                            self._get_cached_model()
                        )  # Recursive call with CPU

                    if self.config.use_huggingface_model:
                        self.logger.info(
                            "Attempting to load model with increased timeout..."
                        )
                        try:
                            # Try with even longer timeout
                            self._hf_session.timeout = (
                                60,
                                600,
                            )  # 10 minutes read timeout

                            model = WhisperForConditionalGeneration.from_pretrained(
                                self.config.model_name,
                                torch_dtype=torch.float32,  # Force CPU dtype
                                device_map=None,
                                local_files_only=False,
                                resume_download=True,
                                trust_remote_code=True,  # Trust custom model code
                                low_cpu_mem_usage=True,  # Reduce memory usage
                            )
                            processor = WhisperProcessor.from_pretrained(
                                self.config.model_name,
                                local_files_only=False,
                                resume_download=True,
                            )

                            # Force CPU
                            model = model.to("cpu")

                            self._model_cache[model_key] = {
                                "model": model,
                                "processor": processor,
                            }
                            self.logger.info(
                                f"Model loaded successfully with extended timeout in {time.time() - start_time:.2f}s"
                            )
                        except Exception as e2:
                            self.logger.error(
                                f"Failed to load model even with extended timeout: {e2}"
                            )
                            raise e2
                    else:
                        raise e

        return self._model_cache[model_key]

    def transcribe_chunk(self, chunk_array: np.ndarray) -> str:
        """Transcribe audio chunk with optimized parameters and error handling."""
        try:
            if len(chunk_array) == 0:
                return ""

            # Ensure proper format
            if len(chunk_array.shape) > 1:
                chunk_array = chunk_array.mean(axis=1)
            chunk_array = chunk_array.astype(np.float32)

            cached_model = self._get_cached_model()
            model = cached_model["model"]
            processor = cached_model["processor"]
            current_device = self.device_manager.get_device()

            if self.config.use_huggingface_model and processor is not None:
                # Use Hugging Face model
                # Prepare input
                input_features = processor(
                    chunk_array,
                    sampling_rate=self.config.target_sample_rate,
                    return_tensors="pt",
                ).input_features

                # Ensure input features match model data type
                if current_device.startswith("cuda"):
                    input_features = input_features.to(
                        current_device, dtype=torch.float32
                    )
                else:
                    input_features = input_features.to(dtype=torch.float32)

                # Generate transcription with Hugging Face compatible parameters
                generation_kwargs = {
                    "language": self.config.language,
                    "task": "transcribe",
                }

                # Only add temperature if it's greater than 0
                if self.config.temperature > 0:
                    generation_kwargs["temperature"] = self.config.temperature
                    generation_kwargs["do_sample"] = True
                else:
                    generation_kwargs["do_sample"] = False

                # Add optional parameters that are supported by Hugging Face Whisper
                if hasattr(model.config, "no_speech_threshold"):
                    generation_kwargs["no_speech_threshold"] = (
                        self.config.no_speech_threshold
                    )

                predicted_ids = model.generate(
                    input_features, **generation_kwargs
                )

                # Decode transcription
                transcription = processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                text = transcription.strip()
            else:
                # Use OpenAI Whisper model
                result = model.transcribe(
                    chunk_array,
                    language=self.config.language,
                    verbose=False,
                    temperature=self.config.temperature,
                    condition_on_previous_text=self.config.condition_on_previous_text,
                    no_speech_threshold=self.config.no_speech_threshold,
                    logprob_threshold=self.config.logprob_threshold,
                    compression_ratio_threshold=self.config.compression_ratio_threshold,
                )
                text = result["text"].strip()

            return (
                RepetitionDetector.clean_repetitive_text(text, self.config)
                if text
                else ""
            )

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Transcription error: {error_msg}")

            # Handle CUDA-specific errors
            if "CUDA" in error_msg and current_device.startswith("cuda"):
                self.cuda_error_count += 1
                self.logger.warning(
                    f"CUDA error count: {self.cuda_error_count}/{self.max_cuda_errors}"
                )

                if self.cuda_error_count >= self.max_cuda_errors:
                    self.logger.error(
                        "🔄 Maximum CUDA errors reached, forcing CPU fallback"
                    )
                    self.device_manager.force_cpu_fallback()
                    # Clear model cache to force reload on CPU
                    with self._model_lock:
                        self._model_cache.clear()
                    self.cuda_error_count = 0
                    # Retry transcription with CPU
                    return self.transcribe_chunk(chunk_array)

            return ""


class UnifiedAudioTranscriber:
    """Unified high-performance audio transcriber with streaming capabilities."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.memory_manager = UnifiedMemoryManager(config)
        # Optional unified preprocessor (can be used to preprocess full audio once)
        try:
            self.audio_preprocessor = UnifiedAudioPreprocessor(config)
        except Exception:
            self.audio_preprocessor = None
        self.audio_processor = StreamingAudioProcessor(config)
        self.whisper_transcriber = OptimizedWhisperTranscriber(config)
        self.sentence_extractor = SentenceExtractor()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with file output."""
        log_file = os.path.join(
            self.config.output_directory, "transcription.log"
        )
        os.makedirs(self.config.output_directory, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            ],
        )
        return logging.getLogger(__name__)

    def transcribe_file(self, audio_file_path: str) -> str:
        """Main transcription method with optional speaker diarization and streaming processing."""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(
                    f"Audio file not found: {audio_file_path}"
                )

            base_filename = Path(audio_file_path).stem
            file_manager = TranscriptionFileManager(
                base_filename, self.config.output_directory, self.config
            )

            print("🎙️  UNIFIED AUDIO TRANSCRIPTION SYSTEM")
            print("=" * 50)

            # Get audio info
            print("🔄 Analyzing audio...")
            audio_info = self.audio_processor.get_audio_info(audio_file_path)
            duration_s = audio_info["duration_seconds"]
            self.audio_duration = duration_s  # Store for preview display

            # Estimate chunks
            chunk_duration_s = self.config.chunk_duration_ms / 1000
            overlap_s = self.config.overlap_ms / 1000
            estimated_chunks = max(
                1, int(duration_s / (chunk_duration_s - overlap_s))
            )

            current_device = (
                self.whisper_transcriber.device_manager.get_device()
            )
            print(f"✅ Audio: {duration_s:.1f}s, ~{estimated_chunks} chunks")
            print(
                f"🔧 Model: {self.config.model_name}, Device: {current_device}"
            )

            # Capability validation (preprocessing & environment)
            if getattr(self.config, "enable_preprocessing", True):
                try:
                    validation = validate_preprocessing(self.config)
                    caps = validation.capabilities
                    available = [
                        name
                        for name, cap in caps.items()
                        if getattr(cap, "available", False)
                    ]
                    print(
                        "🧩 Preprocessing capabilities: "
                        + (", ".join(sorted(available)) if available else "basic")
                    )
                except Exception:
                    # Non-fatal if validation fails
                    pass

            # Try speaker diarization first
            diarized_segments_output = []
            if getattr(self.config, "enable_speaker_diarization", False):
                try:
                    # Load entire audio once for diarization
                    full_audio = self.audio_processor._extract_with_pydub(
                        audio_file_path, 0.0, duration_s
                    )
                    if full_audio is not None and len(full_audio) > 0:
                        # Use pyannote if available, otherwise fallback to basic diarizer
                        if PYAANOTE_AVAILABLE:
                            diarizer = PyannoteDiarizer(self.config)
                            self.logger.info(
                                "Using pyannote.audio for speaker diarization"
                            )
                        else:
                            diarizer = SpeakerDiarizer(self.config)
                            self.logger.info(
                                "Using basic MFCC diarizer (pyannote not available)"
                            )

                        # Get diarization parameters
                        diarization_params = getattr(
                            self.config, "diarization_params", {}
                        )
                        num_speakers = diarization_params.get("num_speakers")
                        min_speakers = diarization_params.get("min_speakers")
                        max_speakers = diarization_params.get("max_speakers")

                        diarized_segments = diarizer.diarize_audio(
                            full_audio,
                            self.config.target_sample_rate,
                            num_speakers=num_speakers,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                        )

                        # Merge adjacent segments by the same speaker
                        try:
                            diarized_segments = (
                                diarizer.merge_similar_speakers(
                                    diarized_segments
                                )
                            )
                        except Exception:
                            pass

                        # Check if diarization found enough segments
                        total_diarized_time = sum(
                            seg.end_time - seg.start_time
                            for seg in diarized_segments
                        )
                        coverage_percentage = (
                            total_diarized_time / duration_s
                        ) * 100

                        if (
                            coverage_percentage < 20
                        ):  # If less than 20% coverage, fallback to standard
                            self.logger.warning(
                                f"Diarization only covered {coverage_percentage:.1f}% of audio, using standard transcription"
                            )
                            diarized_segments_output = []
                        else:
                            # Build per-speaker transcription
                            for idx, seg in enumerate(diarized_segments):
                                text = (
                                    self.whisper_transcriber.transcribe_chunk(
                                        seg.audio_data
                                    )
                                )
                                diarized_segments_output.append(
                                    {
                                        "speaker_id": int(seg.speaker_id),
                                        "start_time": float(seg.start_time),
                                        "end_time": float(seg.end_time),
                                        "confidence": float(seg.confidence),
                                        "text": text,
                                    }
                                )

                            if diarized_segments_output:
                                # Build unified text from diarized segments (ordered)
                                diarized_segments_output.sort(
                                    key=lambda s: s.get("start_time", 0.0)
                                )
                                final_text = "\n".join(
                                    [
                                        f"[Speaker {s['speaker_id']}] ({s['start_time']:.2f}-{s['end_time']:.2f}): {s['text']}".strip()
                                        for s in diarized_segments_output
                                        if s.get("text", "").strip()
                                    ]
                                )

                                # Save unified diarized and per-segment outputs
                                file_manager.save_unified_transcription(
                                    final_text
                                )
                                file_manager.save_speaker_transcription(
                                    diarized_segments_output
                                )
                                print(
                                    f"✅ Diarized transcription saved to: {file_manager.speaker_file_path}"
                                )
                                return final_text

                except Exception as e:
                    self.logger.warning(
                        f"Speaker diarization failed, continuing without diarization: {e}"
                    )

            # Decide chunking strategy
            use_smart = getattr(self.config, "use_smart_chunking", True)
            print(
                "🔄 Using smart chunking..."
                if use_smart and self.audio_preprocessor
                else "🔄 Using standard chunked transcription..."
            )

            # Initialize enhanced preview display if enabled
            if self.config.enable_sentence_preview:
                try:
                    from ..utils.unified_terminal_display import (
                        create_preview_display,
                    )

                    self.preview_display = create_preview_display(
                        estimated_chunks,
                        estimated_duration=self.audio_duration,
                    )
                    print("✨ Enhanced preview display enabled")
                except ImportError:
                    self.preview_display = None
                    print(
                        "⚠️  Enhanced preview not available, using basic preview"
                    )
            else:
                self.preview_display = None

            transcriptions = []
            # Prepare smart segments if enabled
            smart_segments = None
            if use_smart and self.audio_preprocessor:
                try:
                    # Load minimal header info and full audio once via pydub
                    audio = AudioSegment.from_file(audio_file_path)
                    # Create smart segments with VAD-informed chunking
                    smart_segments = self.audio_preprocessor.create_smart_chunks(
                        audio,
                        target_chunk_duration_ms=int(
                            self.config.chunk_duration_ms
                        ),
                        overlap_ms=int(self.config.overlap_ms),
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Smart chunking unavailable, falling back: {e}"
                    )
                    smart_segments = None

            # Choose streaming iterator
            if smart_segments:
                total_chunks = len(smart_segments)
                chunk_iter: Iterator[Tuple[int, np.ndarray]] = (
                    self.audio_processor.stream_smart_chunks(
                        audio_file_path, smart_segments
                    )
                )
            else:
                total_chunks = estimated_chunks
                chunk_iter = self.audio_processor.stream_chunks(audio_file_path)

            with tqdm(total=total_chunks, desc="🎙️ Transcribing", unit="chunk") as pbar:
                for (chunk_index, chunk_array) in chunk_iter:
                    # Track chunk in preview display
                    if self.preview_display:
                        # Estimate timing based on chunk index
                        total_duration = getattr(self, "audio_duration", 0)
                        chunk_start = (
                            chunk_index / estimated_chunks
                        ) * total_duration
                        chunk_end = (
                            (chunk_index + 1) / estimated_chunks
                        ) * total_duration
                        chunk_duration = chunk_end - chunk_start

                        self.preview_display.add_chunk(
                            chunk_index, chunk_start, chunk_end, chunk_duration
                        )
                        self.preview_display.set_current_chunk(chunk_index)
                        self.preview_display.update_chunk_progress(
                            chunk_index, 0.0
                        )

                    # Update progress to show transcription in progress
                    if self.preview_display:
                        self.preview_display.update_chunk_progress(
                            chunk_index, 50.0
                        )

                    transcription = self.whisper_transcriber.transcribe_chunk(
                        chunk_array
                    )
                    transcriptions.append(transcription)

                    # Show preview
                    if (
                        self.config.enable_sentence_preview
                        and transcription.strip()
                    ):
                        # Use enhanced preview display if available
                        if hasattr(self, "preview_display"):
                            self.preview_display.update_chunk_progress(
                                chunk_index, 100.0, transcription.strip()
                            )
                        else:
                            # Fallback to simple preview
                            preview = transcription.strip()[:80]
                            if len(transcription) > 80:
                                preview += "..."
                            print(f"\n📝 Chunk {chunk_index + 1}: {preview}")

                    pbar.update(1)

                    # Memory management
                    self.memory_manager.check_and_cleanup()

            # Merge transcriptions
            print("\n🔄 Merging transcriptions...")
            final_text = self._merge_transcriptions(transcriptions)

            # Save results
            file_manager.save_unified_transcription(final_text)
            print(
                f"✅ Transcription saved to: {file_manager.unified_file_path}"
            )

            return final_text

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def _merge_transcriptions(self, transcriptions: List[str]) -> str:
        """Merge transcriptions with intelligent deduplication."""
        if not transcriptions:
            return ""

        # Filter out empty transcriptions
        valid_transcriptions = [t.strip() for t in transcriptions if t.strip()]

        if not valid_transcriptions:
            return ""

        # Simple merge for now - can be enhanced with more sophisticated logic
        merged_text = " ".join(valid_transcriptions)

        # Basic cleanup
        merged_text = merged_text.replace("  ", " ")  # Remove double spaces
        merged_text = merged_text.strip()

        return merged_text

    def _cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup preview display
            if hasattr(self, "preview_display") and self.preview_display:
                self.preview_display.stop_display_thread()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            self.logger.debug(f"Cleanup error: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()

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

from .config import TranscriptionConfig

# Handle imports with fallback for direct execution
try:
    from ..utils.file_manager import TranscriptionFileManager
    from ..utils.repetition_detector import RepetitionDetector
    from ..utils.sentence_extractor import SentenceExtractor
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.file_manager import TranscriptionFileManager
    from utils.repetition_detector import RepetitionDetector
    from utils.sentence_extractor import SentenceExtractor


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
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                   '-show_format', '-show_streams', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                audio_stream = next(s for s in data['streams'] if s['codec_type'] == 'audio')
                duration = float(data['format']['duration'])
                
                return {
                    'duration_ms': int(duration * 1000),
                    'duration_seconds': duration,
                    'sample_rate': int(audio_stream['sample_rate']),
                    'channels': int(audio_stream['channels'])
                }
        except Exception as e:
            self.logger.warning(f"ffprobe failed, using pydub: {e}")
        
        # Fallback to pydub
        try:
            sample = AudioSegment.from_file(file_path, duration=1.0)
            full_audio = AudioSegment.from_file(file_path)
            duration_ms = len(full_audio)
            del full_audio
            
            return {
                'duration_ms': duration_ms,
                'duration_seconds': duration_ms / 1000,
                'sample_rate': sample.frame_rate,
                'channels': sample.channels
            }
        except Exception as e:
            self.logger.error(f"Audio info extraction failed: {e}")
            raise
    
    def stream_chunks(self, file_path: str) -> Iterator[Tuple[int, np.ndarray]]:
        """Stream audio chunks using ffmpeg or pydub fallback."""
        chunk_duration_s = self.config.chunk_duration_ms / 1000
        overlap_s = self.config.overlap_ms / 1000
        effective_step_s = chunk_duration_s - overlap_s
        
        audio_info = self.get_audio_info(file_path)
        total_duration_s = audio_info['duration_seconds']
        
        chunk_index = 0
        start_time = 0.0
        
        while start_time < total_duration_s:
            end_time = min(start_time + chunk_duration_s, total_duration_s)
            
            # Try ffmpeg first, fallback to pydub
            chunk_array = self._extract_with_ffmpeg(file_path, start_time, end_time - start_time)
            if chunk_array is None:
                chunk_array = self._extract_with_pydub(file_path, start_time, end_time - start_time)
            
            if chunk_array is not None and len(chunk_array) > 0:
                yield chunk_index, chunk_array
            
            chunk_index += 1
            start_time += effective_step_s
    
    def _extract_with_ffmpeg(self, file_path: str, start_time: float, duration: float) -> Optional[np.ndarray]:
        """Extract audio segment using ffmpeg."""
        try:
            cmd = [
                'ffmpeg', '-i', file_path, '-ss', str(start_time), '-t', str(duration),
                '-ar', str(self.config.target_sample_rate), '-ac', '1', '-f', 'f32le', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, stderr=subprocess.DEVNULL, timeout=60)
            
            if result.returncode == 0 and len(result.stdout) > 0:
                audio_data = np.frombuffer(result.stdout, dtype=np.float32)
                if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                return audio_data
                
        except Exception:
            pass
        return None
    
    def _extract_with_pydub(self, file_path: str, start_time: float, duration: float) -> Optional[np.ndarray]:
        """Extract audio segment using pydub as fallback."""
        try:
            start_ms = int(start_time * 1000)
            duration_ms = int(duration * 1000)
            
            audio = AudioSegment.from_file(file_path)
            chunk = audio[start_ms:start_ms + duration_ms]
            
            if chunk.frame_rate != self.config.target_sample_rate:
                chunk = chunk.set_frame_rate(self.config.target_sample_rate)
            if chunk.channels != 1:
                chunk = chunk.set_channels(1)
            
            chunk_array = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            if len(chunk_array) > 0 and np.max(np.abs(chunk_array)) > 0:
                chunk_array = chunk_array / np.max(np.abs(chunk_array))
            
            return chunk_array
            
        except Exception:
            return None


class OptimizedWhisperTranscriber:
    """Optimized Whisper transcriber with model caching."""
    
    _model_cache = {}
    _model_lock = threading.Lock()
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = self._get_cached_model()
    
    def _get_cached_model(self):
        """Get or create cached Whisper model."""
        model_key = f"{self.config.model_name}_{self.config.device}"
        
        with self._model_lock:
            if model_key not in self._model_cache:
                self.logger.info(f"Loading model: {self.config.model_name}...")
                start_time = time.time()
                
                try:
                    if self.config.use_huggingface_model:
                        # Load Hugging Face model
                        self.logger.info("Loading Hugging Face Whisper model...")
                        model = WhisperForConditionalGeneration.from_pretrained(
                            self.config.model_name,
                            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                            device_map="auto" if self.config.device == "cuda" else None
                        )
                        processor = WhisperProcessor.from_pretrained(self.config.model_name)
                        self._model_cache[model_key] = {"model": model, "processor": processor}
                    else:
                        # Load OpenAI Whisper model
                        model = whisper.load_model(self.config.model_name, device=self.config.device)
                        self._model_cache[model_key] = {"model": model, "processor": None}
                    
                    self.logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"Model loading failed: {e}")
                    raise
            
            return self._model_cache[model_key]
    
    def transcribe_chunk(self, chunk_array: np.ndarray) -> str:
        """Transcribe audio chunk with optimized parameters."""
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
            
            if self.config.use_huggingface_model and processor is not None:
                # Use Hugging Face model
                # Prepare input
                input_features = processor(
                    chunk_array, 
                    sampling_rate=self.config.target_sample_rate, 
                    return_tensors="pt"
                ).input_features
                
                if self.config.device == "cuda":
                    input_features = input_features.to(self.config.device)
                
                # Generate transcription with Hugging Face compatible parameters
                generation_kwargs = {
                    "language": self.config.language,
                    "task": "transcribe"
                }
                
                # Only add temperature if it's greater than 0
                if self.config.temperature > 0:
                    generation_kwargs["temperature"] = self.config.temperature
                    generation_kwargs["do_sample"] = True
                else:
                    generation_kwargs["do_sample"] = False
                
                # Add optional parameters that are supported by Hugging Face Whisper
                if hasattr(model.config, 'no_speech_threshold'):
                    generation_kwargs["no_speech_threshold"] = self.config.no_speech_threshold
                
                predicted_ids = model.generate(input_features, **generation_kwargs)
                
                # Decode transcription
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
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
                    compression_ratio_threshold=self.config.compression_ratio_threshold
                )
                text = result["text"].strip()
            
            return RepetitionDetector.clean_repetitive_text(text, self.config) if text else ""
            
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return ""


class UnifiedAudioTranscriber:
    """Unified high-performance audio transcriber with streaming capabilities."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.memory_manager = UnifiedMemoryManager(config)
        self.audio_processor = StreamingAudioProcessor(config)
        self.whisper_transcriber = OptimizedWhisperTranscriber(config)
        self.sentence_extractor = SentenceExtractor()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with file output."""
        log_file = os.path.join(self.config.output_directory, 'transcription.log')
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='a', encoding='utf-8')
            ]
        )
        return logging.getLogger(__name__)
    
    def transcribe_file(self, audio_file_path: str) -> str:
        """Main transcription method with streaming processing."""
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            base_filename = Path(audio_file_path).stem
            file_manager = TranscriptionFileManager(
                base_filename, self.config.output_directory, self.config
            )
            
            print("🎙️  UNIFIED AUDIO TRANSCRIPTION SYSTEM")
            print("=" * 50)
            
            # Get audio info
            print("🔄 Analyzing audio...")
            audio_info = self.audio_processor.get_audio_info(audio_file_path)
            duration_s = audio_info['duration_seconds']
            
            # Estimate chunks
            chunk_duration_s = self.config.chunk_duration_ms / 1000
            overlap_s = self.config.overlap_ms / 1000
            estimated_chunks = max(1, int(duration_s / (chunk_duration_s - overlap_s)))
            
            print(f"✅ Audio: {duration_s:.1f}s, ~{estimated_chunks} chunks")
            print(f"🔧 Model: {self.config.model_name}, Device: {self.config.device}")
            
            # Process with streaming
            transcriptions = []
            with tqdm(total=estimated_chunks, desc="🎙️ Transcribing", unit="chunk") as pbar:
                for chunk_index, chunk_array in self.audio_processor.stream_chunks(audio_file_path):
                    transcription = self.whisper_transcriber.transcribe_chunk(chunk_array)
                    transcriptions.append(transcription)
                    
                    # Show preview
                    if self.config.enable_sentence_preview and transcription.strip():
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
            print("🔄 Saving results...")
            file_manager.save_unified_transcription(final_text)
            
            print(f"✅ Transcription completed!")
            print(f"📝 Length: {len(final_text)} characters")
            print(f"📄 Saved to: {self.config.output_directory}")
            
            return final_text
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _merge_transcriptions(self, transcriptions: List[str]) -> str:
        """Merge transcriptions with deduplication."""
        valid_transcriptions = [t.strip() for t in transcriptions if t.strip()]
        
        if not valid_transcriptions:
            return ""
        
        # Simple but effective merging
        merged_text = " ".join(valid_transcriptions)
        
        # Clean up whitespace and apply final repetition cleaning
        import re
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()
        merged_text = RepetitionDetector.clean_repetitive_text(merged_text, self.config)
        
        return merged_text
    
    def _cleanup(self):
        """Clean up resources."""
        gc.collect()
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()

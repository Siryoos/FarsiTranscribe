"""
Enhanced transcription system with advanced repetition handling and modular design.
RAM-optimized version with streaming audio processing.
"""

import os
import logging
import gc
import time
import weakref
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterator
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

import torch
import whisper
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

from .config import TranscriptionConfig
from ..utils.file_manager import TranscriptionFileManager
from ..utils.repetition_detector import RepetitionDetector
from ..utils.sentence_extractor import SentenceExtractor
from ..utils.performance_monitor import performance_monitor
from ..utils.audio_preprocessor import AudioPreprocessor, get_preprocessing_capabilities
from ..utils.chunk_calculator import create_chunk_calculator
from ..utils.preprocessing_validator import validate_preprocessing
from ..utils.enhanced_memory_manager import create_memory_manager

# Global worker function for multiprocessing (must be outside class)
def transcribe_chunk_worker(chunk_data_and_config):
    """Global worker function for multiprocessing with memory optimization."""
    chunk_index, chunk_array, config_dict = chunk_data_and_config
    
    try:
        # Recreate config from dict
        config = TranscriptionConfig.from_dict(config_dict)
        
        # Create fresh transcriber instance
        temp_transcriber = StandardWhisperTranscriber(config, shared_model=None)
        result = temp_transcriber.transcribe_chunk(chunk_array)
        
        # Clean up immediately
        del temp_transcriber
        gc.collect()
        
        return chunk_index, result
    except Exception as e:
        return chunk_index, ""

# Global model instance for sharing across processes with weak references
_global_model = None
_model_lock = threading.Lock()
_model_refs = weakref.WeakSet()

def get_shared_model(config: TranscriptionConfig):
    """Get or create shared Whisper model instance with memory management."""
    global _global_model
    
    with _model_lock:
        if _global_model is None:
            logging.info(f"Loading shared Whisper {config.model_name} model...")
            start_time = time.time()
            
            # Enhanced SSL handling for model download
            import ssl
            import urllib.request
            import certifi
            
            # Create SSL context with proper certificate verification
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Monkey patch urllib to use our SSL context
            original_urlopen = urllib.request.urlopen
            def urlopen_with_ssl(*args, **kwargs):
                kwargs['context'] = ssl_context
                return original_urlopen(*args, **kwargs)
            urllib.request.urlopen = urlopen_with_ssl
            
            try:
                _global_model = whisper.load_model(config.model_name, device=config.device)
                load_time = time.time() - start_time
                logging.info(f"Shared model loaded in {load_time:.2f} seconds")
                
                # Track model references for cleanup
                _model_refs.add(_global_model)
                
            except Exception as e:
                # Fallback: try without SSL verification
                logging.warning(f"SSL verification failed, trying without verification: {e}")
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                def urlopen_without_ssl(*args, **kwargs):
                    kwargs['context'] = ssl_context
                    return original_urlopen(*args, **kwargs)
                urllib.request.urlopen = urlopen_without_ssl
                
                _global_model = whisper.load_model(config.model_name, device=config.device)
                load_time = time.time() - start_time
                logging.info(f"Shared model loaded (no SSL) in {load_time:.2f} seconds")
                
                # Track model references for cleanup
                _model_refs.add(_global_model)
            finally:
                # Restore original urlopen
                urllib.request.urlopen = original_urlopen
    
    return _global_model

def cleanup_shared_model():
    """Clean up shared model to free memory."""
    global _global_model
    
    with _model_lock:
        if _global_model is not None:
            del _global_model
            _global_model = None
            _model_refs.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# MemoryManager is now replaced by EnhancedMemoryManager from enhanced_memory_manager.py
# This class is kept for backward compatibility but delegates to the enhanced version
class MemoryManager:
    """Memory management utility for transcription system (legacy wrapper)."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self._enhanced_manager = create_memory_manager(config)
        
    def check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold."""
        return self._enhanced_manager.check_memory_usage()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed based on time interval."""
        return self._enhanced_manager.should_cleanup()
    
    def cleanup(self, force: bool = False):
        """Perform memory cleanup."""
        self._enhanced_manager.cleanup(force=force)


class StreamingAudioProcessor:
    """Streaming audio processor to reduce memory usage."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.chunk_calculator = create_chunk_calculator(config)
    
    def calculate_total_chunks(self, audio_duration_ms: int) -> int:
        """Calculate the exact number of chunks that will be generated."""
        return self.chunk_calculator.calculate_total_chunks(audio_duration_ms)
        
    def process_audio_stream(self, audio_file_path: str) -> Iterator[Tuple[int, np.ndarray]]:
        """Process audio in streaming fashion to reduce memory usage."""
        from pydub import AudioSegment
        
        # Load audio in chunks instead of full file
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono and normalize
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Set sample rate
        if audio.frame_rate != self.config.target_sample_rate:
            audio = audio.set_frame_rate(self.config.target_sample_rate)
        
        # Get chunk boundaries using unified calculator
        chunk_boundaries = self.chunk_calculator.generate_chunk_boundaries(len(audio))
        
        # Process in streaming chunks
        for chunk_index, (start_ms, end_ms) in enumerate(chunk_boundaries):
            # Extract chunk
            chunk = audio[start_ms:end_ms]
            chunk_array = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            
            # Normalize
            if chunk_array.size > 0:
                chunk_array = chunk_array / np.max(np.abs(chunk_array)) if np.max(np.abs(chunk_array)) > 0 else chunk_array
            
            yield chunk_index, chunk_array
            
            # Clean up chunk to free memory
            del chunk
            gc.collect()


class AudioProcessor(ABC):
    """Abstract base class for audio processing."""
    
    @abstractmethod
    def process_audio(self, audio_file_path: str) -> AudioSegment:
        """Process audio file and return AudioSegment."""
        pass
    
    @abstractmethod
    def create_chunks(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """Create chunks from audio."""
        pass


class WhisperTranscriber(ABC):
    """Abstract base class for Whisper transcription."""
    
    @abstractmethod
    def transcribe_chunk(self, chunk: np.ndarray) -> str:
        """Transcribe a single chunk."""
        pass


class StandardAudioProcessor(AudioProcessor):
    """Standard audio processing implementation with preprocessing support."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize preprocessor if enabled
        if config.enable_preprocessing:
            if config.enable_advanced_preprocessing:
                from ..utils.advanced_audio_preprocessor import AdvancedAudioPreprocessor
                self.preprocessor = AdvancedAudioPreprocessor(
                    enable_facebook_denoiser=config.enable_facebook_denoiser,
                    enable_persian_optimization=config.enable_persian_optimization,
                    adaptive_processing=config.adaptive_processing,
                    sample_rate=config.target_sample_rate
                )
            else:
                self.preprocessor = AudioPreprocessor(
                    enable_noise_reduction=config.enable_noise_reduction,
                    enable_vad=config.enable_voice_activity_detection,
                    enable_speech_enhancement=config.enable_speech_enhancement,
                    sample_rate=config.target_sample_rate
                )
        else:
            self.preprocessor = None
    
    def process_audio(self, audio_file_path: str) -> AudioSegment:
        """Load and preprocess audio with validation."""
        try:
            self.logger.info(f"Loading audio: {audio_file_path}")
            
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            audio = AudioSegment.from_file(audio_file_path)
            
            if len(audio) == 0:
                raise ValueError("Audio file is empty")
            
            # Optimize audio for transcription
            audio = audio.set_frame_rate(self.config.target_sample_rate)
            audio = audio.set_channels(1)
            
            if audio.max_possible_amplitude > 0:
                audio = audio.normalize()
            
            # Apply preprocessing if enabled
            if self.preprocessor:
                self.logger.info("Applying audio preprocessing...")
                audio = self.preprocessor.preprocess_audio(audio)
            
            self.logger.info(f"Audio processed - Duration: {len(audio)/1000:.1f}s")
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            raise
    
    def create_chunks(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """Create optimized chunks with preprocessing-aware chunking."""
        # Use advanced chunking if available
        if (self.preprocessor and self.config.use_smart_chunking and 
            hasattr(self.preprocessor, 'create_intelligent_chunks')):
            chunks = self.preprocessor.create_intelligent_chunks(
                audio, 
                self.config.chunk_duration_ms, 
                self.config.overlap_ms
            )
            return chunks
        # Use smart chunking if preprocessing is enabled
        elif self.preprocessor and self.config.use_smart_chunking:
            chunks = self.preprocessor.create_smart_chunks(
                audio, 
                self.config.chunk_duration_ms, 
                self.config.overlap_ms
            )
            return chunks
        
        # Fallback to standard chunking
        chunks = []
        audio_length = len(audio)
        
        if audio_length <= self.config.chunk_duration_ms:
            return [(0, audio_length)]
        
        current_pos = 0
        chunk_size = self.config.chunk_duration_ms
        overlap = self.config.overlap_ms
        
        while current_pos < audio_length:
            end_pos = min(current_pos + chunk_size, audio_length)
            chunks.append((current_pos, end_pos))
            
            if end_pos == audio_length:
                break
                
            current_pos = end_pos - overlap
            
        self.logger.info(f"Created {len(chunks)} chunks with {overlap}ms overlap")
        return chunks


class StandardWhisperTranscriber(WhisperTranscriber):
    """Standard Whisper transcription implementation with shared model support."""
    
    def __init__(self, config: TranscriptionConfig, shared_model: Optional[whisper.Whisper] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = shared_model
        if self.model is None:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model with enhanced configuration."""
        try:
            self.logger.info(f"Loading Whisper {self.config.model_name} model...")
            
            # Disable SSL verification for model download
            import ssl
            import urllib.request
            
            # Create SSL context that doesn't verify certificates
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Monkey patch urllib to use our SSL context
            original_urlopen = urllib.request.urlopen
            def urlopen_without_ssl(*args, **kwargs):
                kwargs['context'] = ssl_context
                return original_urlopen(*args, **kwargs)
            urllib.request.urlopen = urlopen_without_ssl
            
            start_time = time.time()
            self.model = whisper.load_model(self.config.model_name, device=self.config.device)
            load_time = time.time() - start_time
            
            # Restore original urlopen
            urllib.request.urlopen = original_urlopen
            
            if self.config.device == "cuda":
                self.model = self.model.to(self.config.device)
            
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def transcribe_chunk(self, chunk: np.ndarray) -> str:
        """Transcribe chunk with anti-repetition measures."""
        try:
            if len(chunk.shape) > 1:
                chunk = chunk.mean(axis=1)
            
            chunk = chunk.astype(np.float32)
            if len(chunk) == 0:
                return ""
            
            if np.max(np.abs(chunk)) > 0:
                chunk = chunk / np.max(np.abs(chunk))
            
            # Enhanced transcription parameters to reduce repetition
            result = self.model.transcribe(
                chunk,
                language=self.config.language,
                verbose=False,
                temperature=self.config.temperature,
                condition_on_previous_text=self.config.condition_on_previous_text,
                no_speech_threshold=self.config.no_speech_threshold,
                logprob_threshold=self.config.logprob_threshold,
                compression_ratio_threshold=self.config.compression_ratio_threshold
            )
            
            transcribed_text = result["text"].strip()
            
            # Immediate repetition cleaning
            cleaned_text = RepetitionDetector.clean_repetitive_text(transcribed_text, self.config)
            
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return ""


class TranscriptionMerger:
    """Advanced transcription merging with deduplication."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def merge_transcriptions(self, transcriptions: List[str]) -> str:
        """Advanced merging with overlap detection and deduplication."""
        if not transcriptions:
            return ""
        
        # Filter valid transcriptions
        valid_transcriptions = [t.strip() for t in transcriptions if t.strip()]
        if not valid_transcriptions:
            return ""
        
        if len(valid_transcriptions) == 1:
            return RepetitionDetector.clean_repetitive_text(valid_transcriptions[0], self.config)
        
        # Advanced merging with overlap detection
        merged_text = valid_transcriptions[0]
        
        for current_text in valid_transcriptions[1:]:
            # Find potential overlap between texts
            merged_words = merged_text.split()
            current_words = current_text.split()
            
            best_overlap = 0
            overlap_start = len(merged_words)
            
            # Look for overlapping sequences
            min_overlap_length = min(10, len(merged_words), len(current_words))
            
            for i in range(min_overlap_length, 0, -1):
                if len(merged_words) >= i:
                    merged_suffix = merged_words[-i:]
                    if len(current_words) >= i and current_words[:i] == merged_suffix:
                        # Check similarity to avoid false positives
                        suffix_text = ' '.join(merged_suffix)
                        prefix_text = ' '.join(current_words[:i])
                        if RepetitionDetector.similarity_ratio(suffix_text, prefix_text) > self.config.repetition_threshold:
                            best_overlap = i
                            overlap_start = len(merged_words) - i
                            break
            
            # Merge with overlap removal
            if best_overlap > 0:
                merged_text = ' '.join(merged_words[:overlap_start] + current_words)
            else:
                merged_text += ' ' + current_text
        
        # Final comprehensive cleaning
        final_text = RepetitionDetector.clean_repetitive_text(merged_text, self.config)
        
        # Additional cleanup
        import re
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        self.logger.info(f"Merged {len(valid_transcriptions)} segments with deduplication")
        return final_text


class UnifiedAudioTranscriber:
    """Enhanced transcription system with advanced repetition handling and modular design."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Validate preprocessing capabilities before starting
        self.logger.info("Validating preprocessing capabilities...")
        validation_result = validate_preprocessing(config)
        
        if not validation_result.success:
            self.logger.error("Preprocessing validation failed:")
            for error in validation_result.errors:
                self.logger.error(f"  - {error}")
            raise RuntimeError("Preprocessing validation failed. Please check the error messages above.")
        
        if validation_result.warnings:
            self.logger.warning("Preprocessing validation warnings:")
            for warning in validation_result.warnings:
                self.logger.warning(f"  - {warning}")
        
        # Initialize enhanced memory manager
        self.memory_manager = create_memory_manager(config)
        
        # Log preprocessing capabilities
        capabilities = get_preprocessing_capabilities()
        self.logger.info(f"Preprocessing capabilities: {capabilities}")
        
        # Setup GPU
        self._setup_gpu()
        
        # Set up multiprocessing with optimized worker count
        self.num_workers = min(self.config.num_workers, cpu_count())
        self.logger.info(f"Using {self.num_workers} CPU cores for parallel processing")
        
        # Initialize shared model
        if self.config.device == "cpu":
            self.shared_model = get_shared_model(config)
        else:
            self.shared_model = None
        
        # Initialize components
        self.audio_processor = StandardAudioProcessor(config)
        self.whisper_transcriber = StandardWhisperTranscriber(config, shared_model=self.shared_model)
        self.transcription_merger = TranscriptionMerger(config)
        self.sentence_extractor = SentenceExtractor()
        self.repetition_detector = RepetitionDetector()
        self.streaming_processor = StreamingAudioProcessor(config)
        
        # Create unified chunk calculator
        self.chunk_calculator = create_chunk_calculator(config)
        
        # Store validation result for later use
        self.validation_result = validation_result
    

    def _prepare_audio_chunks_parallel(self, audio: AudioSegment, chunks: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Prepare audio chunks in parallel for better performance."""
        if not self.config.use_parallel_audio_prep:
            return self._prepare_audio_chunks_sequential(audio, chunks)
        
        def prepare_single_chunk(chunk_data):
            chunk_index, (start_ms, end_ms) = chunk_data
            try:
                chunk_audio = audio[start_ms:end_ms]
                chunk_array = np.array(chunk_audio.get_array_of_samples())
                
                if chunk_audio.channels == 2:
                    chunk_array = chunk_array.reshape((-1, 2)).mean(axis=1)
                    
                return chunk_index, chunk_array
            except Exception as e:
                self.logger.error(f"Error preparing chunk {chunk_index}: {e}")
                return chunk_index, np.array([])
        
        # Prepare chunk data
        chunk_data = [(i, chunk) for i, chunk in enumerate(chunks)]
        
        # Use ThreadPoolExecutor for I/O-bound audio preparation
        audio_arrays = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=min(4, self.num_workers)) as executor:
            futures = {executor.submit(prepare_single_chunk, data): data for data in chunk_data}
            
            with tqdm(total=len(chunks), desc="📦 Preparing chunks", leave=False) as prep_bar:
                for future in as_completed(futures):
                    chunk_index, chunk_array = future.result()
                    audio_arrays[chunk_index] = chunk_array
                    prep_bar.update(1)
        
        return audio_arrays
    
    def _prepare_audio_chunks_sequential(self, audio: AudioSegment, chunks: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Sequential audio chunk preparation (fallback)."""
        audio_arrays = []
        
        with tqdm(total=len(chunks), desc="📦 Preparing", leave=False) as prep_bar:
            for start_ms, end_ms in chunks:
                try:
                    chunk_audio = audio[start_ms:end_ms]
                    chunk_array = np.array(chunk_audio.get_array_of_samples())
                    
                    if chunk_audio.channels == 2:
                        chunk_array = chunk_array.reshape((-1, 2)).mean(axis=1)
                        
                    audio_arrays.append(chunk_array)
                    prep_bar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error preparing chunk: {e}")
                    audio_arrays.append(np.array([]))
                    prep_bar.update(1)
        
        return audio_arrays

    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging for performance monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(self.config.output_directory, 'transcription.log'),
                    mode='a',
                    encoding='utf-8'
                )
            ]
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _setup_gpu(self) -> None:
        """Configure GPU settings with validation."""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            self.config.device = "cpu"
            return
            
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
            
            device_props = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA Memory: {device_props.total_memory / 1e9:.1f} GB")
            
        except Exception as e:
            self.logger.error(f"GPU setup error: {e}")
            self.config.device = "cpu"
    
    def _process_chunks_with_preview(self, audio: AudioSegment, chunks: List[Tuple[int, int]], 
                                   file_manager: TranscriptionFileManager) -> List[str]:
        """Process chunks with enhanced preview and monitoring - OPTIMIZED VERSION."""
        all_transcriptions = []
        
        # Start performance monitoring
        with performance_monitor(enable_monitoring=True) as monitor:
            monitor.start_monitoring(len(chunks), len(audio) / 1000)
            
            # Parallel audio preparation
            print("🔄 Preparing audio chunks...")
            audio_arrays = self._prepare_audio_chunks_parallel(audio, chunks)
            
            # Prepare chunk data for parallel processing
            chunk_data = [(i, chunk_array) for i, chunk_array in enumerate(audio_arrays)]
            
            print(f"🚀 Processing {len(chunks)} chunks using {self.num_workers} CPU cores")
            
            # Use multiprocessing for parallel transcription with optimized settings
            with Pool(processes=self.num_workers, maxtasksperchild=10) as pool:
                with tqdm(total=len(chunks), desc="🎵 Transcribing", unit="chunk") as main_pbar:
                    
                    # Process chunks in parallel with chunking for memory efficiency
                    chunk_size = max(1, len(chunks) // (self.num_workers * 2))
                    results = []
                    
                    for i in range(0, len(chunk_data), chunk_size):
                        batch = chunk_data[i:i + chunk_size]
                        
                        # Prepare data for global worker function
                        config_dict = self.config.to_dict()
                        worker_data = [(chunk_index, chunk_array, config_dict) for chunk_index, chunk_array in batch]
                        
                        # Process batch
                        batch_results = []
                        for result in pool.imap_unordered(transcribe_chunk_worker, worker_data):
                            chunk_index, transcription = result
                            batch_results.append((chunk_index, transcription))
                            
                            # Update performance monitor
                            monitor.update_progress(len(results) + len(batch_results))
                            
                            # Enhanced preview with repetition detection
                            if self.config.enable_sentence_preview and transcription.strip():
                                sentences = self.sentence_extractor.extract_sentences(
                                    transcription, self.config.preview_sentence_count
                                )
                                if sentences:
                                    preview = self.sentence_extractor.format_sentence_preview(
                                        sentences, chunk_index + 1
                                    )
                                    print(f"\n{preview}")
                            
                            main_pbar.update(1)
                            main_pbar.set_postfix({
                                'Cores': f'{self.num_workers}',
                                'Processed': f"{len(results) + len(batch_results)}/{len(chunks)}",
                                'Non-empty': f"{sum(1 for _, r in results + batch_results if r.strip())}"
                            })
                        
                        results.extend(batch_results)
                        
                        # Memory cleanup after each batch
                        if self.config.memory_efficient_mode:
                            gc.collect()
                    
                    # Sort results by chunk index to maintain order
                    results.sort(key=lambda x: x[0])
                    all_transcriptions = [transcription for _, transcription in results]
        
        return all_transcriptions
    
    def transcribe_file(self, audio_file_path: str) -> str:
        """Main transcription method with comprehensive anti-repetition pipeline and enhanced error handling."""
        try:
            # Validate input file
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            base_filename = Path(audio_file_path).stem
            file_manager = TranscriptionFileManager(base_filename, self.config.output_directory, self.config)
            
            print("=" * 80)
            print("🎙️  ENHANCED AUDIO TRANSCRIPTION SYSTEM")
            print("🔧 Anti-Repetition Features Enabled")
            print("💾 RAM-Optimized Processing")
            print("🔍 Preprocessing Validation: ✅")
            print("=" * 80)
            
            # Get audio duration without loading entire file
            print("🔄 Getting audio duration...")
            from pydub.utils import which
            import subprocess
            
            # Use ffprobe to get duration quickly
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', audio_file_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    duration_seconds = float(result.stdout.strip())
                    duration_ms = int(duration_seconds * 1000)
                    print(f"✅ Audio duration: {duration_seconds:.1f}s ({duration_ms}ms)")
                else:
                    # Fallback: load just metadata
                    print("🔄 Loading audio metadata...")
                    audio = AudioSegment.from_file(audio_file_path)
                    duration_ms = len(audio)
                    print(f"✅ Audio duration: {duration_ms/1000:.1f}s")
                    del audio  # Free memory immediately
            except Exception as e:
                print(f"⚠️  Could not get duration quickly: {e}")
                print("🔄 Loading audio file...")
                audio = AudioSegment.from_file(audio_file_path)
                duration_ms = len(audio)
                print(f"✅ Audio loaded: {duration_ms/1000:.1f}s duration")
                del audio  # Free memory immediately
            
            # Get fast chunk analysis for initial display
            print("🔄 Analyzing chunks...")
            chunk_analysis = self.chunk_calculator.get_fast_analysis(duration_ms)
            print(f"✅ Chunk analysis complete: {chunk_analysis['total_chunks']} chunks")
            
            # Estimate memory usage
            memory_estimate = self.chunk_calculator.estimate_memory_usage(
                duration_ms, 16000, 1  # Assume standard sample rate and mono
            )
            print(f"✅ Memory estimate: {memory_estimate['peak_memory_mb']:.1f}MB peak")
            
            # Log analysis information
            self.logger.info(f"Audio analysis: {duration_ms/1000:.1f}s, {chunk_analysis['total_chunks']} chunks")
            self.logger.info(f"Memory estimate: {memory_estimate['peak_memory_mb']:.1f}MB peak")
            
            # Optimize memory for this operation
            original_threshold = self.memory_manager.optimize_for_operation(
                "transcription", memory_estimate['peak_memory_mb']
            )
            
            try:
                # Choose transcription mode based on file size and memory efficiency setting
                if self.config.memory_efficient_mode or chunk_analysis['total_chunks'] > 10:
                    self.logger.info("Using streaming mode for memory efficiency")
                    result = self._transcribe_file_streaming(audio_file_path, file_manager)
                else:
                    self.logger.info("Using standard mode for optimal performance")
                    result = self._transcribe_file_standard(audio_file_path, file_manager)
                
                return result
                
            finally:
                # Restore original memory thresholds
                self.memory_manager.restore_thresholds(original_threshold)
                
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            # Log memory report for debugging
            try:
                memory_report = self.memory_manager.get_memory_report()
                self.logger.error(f"Memory report: {memory_report}")
            except:
                pass
            raise
    
    def _transcribe_file_streaming(self, audio_file_path: str, file_manager: TranscriptionFileManager) -> str:
        """Streaming transcription method for memory efficiency."""
        print("🔄 Using streaming mode for memory efficiency...")
        
        try:
            # Get audio duration for streaming (reuse the duration we already calculated)
            print("🔄 Preparing for streaming...")
            
            # Get duration from the main method or calculate it quickly
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                    '-of', 'csv=p=0', audio_file_path
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    duration_seconds = float(result.stdout.strip())
                    duration_ms = int(duration_seconds * 1000)
                else:
                    # Fallback: load just metadata
                    audio = AudioSegment.from_file(audio_file_path)
                    duration_ms = len(audio)
                    del audio
            except Exception as e:
                print(f"⚠️  Could not get duration quickly: {e}")
                audio = AudioSegment.from_file(audio_file_path)
                duration_ms = len(audio)
                del audio
            
            chunk_analysis = self.chunk_calculator.get_fast_analysis(duration_ms)
            total_chunks = chunk_analysis['total_chunks']
            print(f"✅ Streaming chunks calculated: {total_chunks} chunks")
            
            print(f"📊 Audio Analysis:")
            print(f"   Duration: {duration_ms/1000:.1f} seconds")
            print(f"   Total Chunks: {total_chunks}")
            print(f"   Chunk Duration: {chunk_analysis['chunk_duration_ms']}ms ({chunk_analysis['chunk_duration_ms']/1000:.1f}s)")
            print(f"   Overlap: {chunk_analysis['overlap_ms']}ms ({chunk_analysis['overlap_ms']/1000:.1f}s)")
            print(f"   Effective Chunk Duration: {chunk_analysis['effective_chunk_duration_ms']}ms ({chunk_analysis['effective_chunk_duration_ms']/1000:.1f}s)")
            print(f"   Average Chunk Duration: {chunk_analysis['avg_chunk_duration_ms']:.0f}ms")
            print()
            
            transcriptions = []
            chunk_count = 0
            
            print("🔄 Starting transcription with progress tracking...")
            print("=" * 60)
            
            # Process audio in streaming fashion with progress bar
            from tqdm import tqdm
            
            print("🔄 Initializing progress bar...")
            with tqdm(total=total_chunks, desc="🎙️ Transcribing", unit="chunk", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                print("🔄 Starting audio stream processing...")
                for chunk_index, chunk_array in self.streaming_processor.process_audio_stream(audio_file_path):
                    try:
                        # Transcribe chunk
                        transcription = self.whisper_transcriber.transcribe_chunk(chunk_array)
                        transcriptions.append(transcription)
                        chunk_count += 1
                        
                        # Memory cleanup after each chunk
                        del chunk_array
                        self.memory_manager.cleanup()
                        
                        # Update progress bar
                        pbar.update(1)
                        
                        # Show transcription preview for each chunk
                        if transcription.strip():
                            # Get first line of transcription
                            first_line = transcription.strip().split('\n')[0][:100]  # First 100 chars
                            if len(first_line) == 100:
                                first_line += "..."
                            
                            # Update progress bar description with preview
                            pbar.set_description(f"🎙️ Chunk {chunk_count}: {first_line}")
                            
                            # Print detailed preview for first few chunks
                            if chunk_count <= 5:
                                print(f"\n📝 Chunk {chunk_count} Preview: {first_line}")
                        
                        # Show memory usage every 5 chunks
                        if chunk_count % 5 == 0:
                            memory_report = self.memory_manager.get_memory_report()
                            print(f"\n💾 Memory: {memory_report['current_usage_mb']:.0f}MB")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_index}: {e}")
                        transcriptions.append("")  # Add empty transcription for failed chunk
                        chunk_count += 1
                        pbar.update(1)
                        continue
            
            # Merge transcriptions
            print("\n🔄 Merging transcriptions...")
            final_transcription = self.transcription_merger.merge_transcriptions(transcriptions)
            
            # Save results
            print("🔄 Saving results...")
            success = file_manager.save_unified_transcription(final_transcription)
            
            if not success:
                raise RuntimeError("Failed to save transcription")
            
            # Display results
            print(f"\n✅ Streaming transcription completed: {chunk_count}/{total_chunks} chunks processed")
            print(f"📄 Output saved to: {self.config.output_directory}")
            
            return final_transcription
            
        except Exception as e:
            self.logger.error(f"Streaming transcription failed: {e}")
            raise
        finally:
            self._cleanup_memory()
    
    def _transcribe_file_standard(self, audio_file_path: str, file_manager: TranscriptionFileManager) -> str:
        """Standard transcription method for smaller files."""
        try:
            # Phase 1: Audio preparation
            print("🔄 Phase 1: Loading audio...")
            audio = self.audio_processor.process_audio(audio_file_path)
            
            # Phase 2: Intelligent chunking
            print("🔄 Phase 2: Creating optimized segments...")
            chunks = self.audio_processor.create_chunks(audio)
            
            # Display chunk information
            duration_ms = len(audio)
            chunk_duration = self.config.chunk_duration_ms
            overlap = self.config.overlap_ms
            effective_chunk_duration = chunk_duration - overlap
            
            print(f"📊 Audio Analysis:")
            print(f"   Duration: {duration_ms/1000:.1f} seconds")
            print(f"   Total Chunks: {len(chunks)}")
            print(f"   Chunk Duration: {chunk_duration}ms ({chunk_duration/1000:.1f}s)")
            print(f"   Overlap: {overlap}ms ({overlap/1000:.1f}s)")
            print(f"   Effective Chunk Duration: {effective_chunk_duration}ms ({effective_chunk_duration/1000:.1f}s)")
            print()
            
            print(f"📊 Processing {len(chunks)} segments • Duration: {len(audio)/1000:.1f}s")
            
            # Phase 3: Enhanced transcription
            print("🔄 Phase 3: Transcribing with repetition control...")
            transcriptions = self._process_chunks_with_preview(audio, chunks, file_manager)
            
            # Phase 4: Advanced merging with deduplication
            print("\n🔄 Phase 4: Merging with deduplication...")
            final_transcription = self.transcription_merger.merge_transcriptions(transcriptions)
            
            # Phase 5: Save results
            print("🔄 Phase 5: Saving cleaned results...")
            success = file_manager.save_unified_transcription(final_transcription)
            
            if not success:
                raise RuntimeError("Failed to save transcription")
            
            # Display results
            self._display_completion_summary(file_manager, audio, len(transcriptions), final_transcription)
            
            return final_transcription
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
        finally:
            self._cleanup_memory()
    
    def _display_completion_summary(self, file_manager: TranscriptionFileManager, 
                                  audio: AudioSegment, segment_count: int, transcription: str) -> None:
        """Display comprehensive results including cleaning statistics."""
        print("\n" + "=" * 80)
        print("✅ TRANSCRIPTION COMPLETED WITH CLEANING")
        print("=" * 80)
        
        info = file_manager.get_transcription_info()
        audio_duration = len(audio) / 1000
        
        print(f"📊 Audio Metrics:")
        print(f"   • Duration: {audio_duration:.1f} seconds")
        print(f"   • Segments: {segment_count}")
        print(f"   • Device: {self.config.device.upper()}")
        
        print(f"📄 Output Files:")
        print(f"   • Original: {info['unified_file_path']}")
        print(f"   • Cleaned: {info['cleaned_file_path']}")
        
        if info['original_exists'] and info['cleaned_exists']:
            reduction_percent = ((info['original_characters'] - info['cleaned_characters']) / 
                               info['original_characters'] * 100) if info['original_characters'] > 0 else 0
            
            print(f"📈 Cleaning Results:")
            print(f"   • Original: {info['original_characters']:,} chars, {info['original_words']:,} words")
            print(f"   • Cleaned: {info['cleaned_characters']:,} chars, {info['cleaned_words']:,} words")
            print(f"   • Reduction: {reduction_percent:.1f}%")
        
        print("=" * 80)
    
    def _cleanup_memory(self) -> None:
        """Enhanced memory cleanup with comprehensive resource management."""
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean up shared model if not needed
        if hasattr(self, 'shared_model') and self.shared_model is not None:
            cleanup_shared_model()
        
        # Force memory cleanup
        self.memory_manager.cleanup(force=True)
        
        # Log memory usage
        if hasattr(self, 'logger'):
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage after cleanup: {memory_mb:.1f} MB")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_memory()
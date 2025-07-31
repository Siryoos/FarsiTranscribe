 """
Enhanced transcription system with advanced repetition handling and modular design.
"""

import os
import logging
import gc
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

import torch
import whisper
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

from .config import TranscriptionConfig
from ..utils.file_manager import TranscriptionFileManager
from ..utils.repetition_detector import RepetitionDetector
from ..utils.sentence_extractor import SentenceExtractor


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
    """Standard audio processing implementation."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
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
            
            self.logger.info(f"Audio processed - Duration: {len(audio)/1000:.1f}s")
            return audio
            
        except Exception as e:
            self.logger.error(f"Error loading audio: {e}")
            raise
    
    def create_chunks(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        """Create optimized chunks with reduced overlap."""
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
    """Standard Whisper transcription implementation."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Whisper model with enhanced configuration."""
        try:
            self.logger.info(f"Loading Whisper {self.config.model_name} model...")
            
            start_time = time.time()
            self.model = whisper.load_model(self.config.model_name, device=self.config.device)
            load_time = time.time() - start_time
            
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
        
        # Initialize components
        self.audio_processor = StandardAudioProcessor(config)
        self.whisper_transcriber = StandardWhisperTranscriber(config)
        self.transcription_merger = TranscriptionMerger(config)
        self.sentence_extractor = SentenceExtractor()
        self.repetition_detector = RepetitionDetector()
        
        # Setup GPU
        self._setup_gpu()
    
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
        """Process chunks with enhanced preview and monitoring."""
        all_transcriptions = []
        
        print("🔄 Preparing audio chunks...")
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
        
        # Process in batches
        batch_size = self.config.batch_size
        total_batches = (len(audio_arrays) + batch_size - 1) // batch_size
        
        print(f"🚀 Processing {len(chunks)} chunks in {total_batches} batches")
        
        with tqdm(total=len(chunks), desc="🎵 Transcribing", unit="chunk") as main_pbar:
            
            for batch_num, i in enumerate(range(0, len(audio_arrays), batch_size), 1):
                batch = audio_arrays[i:i + batch_size]
                
                main_pbar.set_description(f"🎵 Batch {batch_num}/{total_batches}")
                
                try:
                    # Process batch
                    for j, chunk_array in enumerate(batch):
                        chunk_index = i + j
                        result = self.whisper_transcriber.transcribe_chunk(chunk_array)
                        all_transcriptions.append(result)
                        
                        # Enhanced preview with repetition detection
                        if self.config.enable_sentence_preview and result.strip():
                            sentences = self.sentence_extractor.extract_sentences(
                                result, self.config.preview_sentence_count
                            )
                            if sentences:
                                preview = self.sentence_extractor.format_sentence_preview(
                                    sentences, chunk_index + 1
                                )
                                print(f"\n{preview}")
                        
                        main_pbar.update(1)
                        main_pbar.set_postfix({
                            'GPU': '✓' if self.config.device == "cuda" else '✗',
                            'Processed': f"{len(all_transcriptions)}/{len(chunks)}",
                            'Non-empty': f"{sum(1 for r in all_transcriptions if r.strip())}"
                        })
                
                except Exception as e:
                    self.logger.error(f"Batch {batch_num} error: {e}")
                    batch_results = [""] * len(batch)
                    all_transcriptions.extend(batch_results)
                    main_pbar.update(len(batch))
                
                # Memory cleanup
                if batch_num % 4 == 0:
                    gc.collect()
                    if self.config.device == "cuda":
                        torch.cuda.empty_cache()
        
        return all_transcriptions
    
    def transcribe_file(self, audio_file_path: str) -> str:
        """Main transcription method with comprehensive anti-repetition pipeline."""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        base_filename = Path(audio_file_path).stem
        file_manager = TranscriptionFileManager(base_filename, self.config.output_directory, self.config)
        
        print("=" * 80)
        print("🎙️  ENHANCED AUDIO TRANSCRIPTION SYSTEM")
        print("🔧 Anti-Repetition Features Enabled")
        print("=" * 80)
        
        try:
            # Phase 1: Audio preparation
            print("🔄 Phase 1: Loading audio...")
            audio = self.audio_processor.process_audio(audio_file_path)
            
            # Phase 2: Intelligent chunking
            print("🔄 Phase 2: Creating optimized segments...")
            chunks = self.audio_processor.create_chunks(audio)
            
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
        """Enhanced memory cleanup."""
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_memory()
import os
import logging
import gc
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import tempfile
from contextlib import contextmanager
from difflib import SequenceMatcher

import torch
import whisper
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


@dataclass
class TranscriptionConfig:
    """Enhanced configuration class for transcription parameters."""
    model_name: str = "large-v3"
    language: str = "fa"
    chunk_duration_ms: int = 20000  # Shorter chunks for better accuracy
    overlap_ms: int = 200  # Minimal overlap to reduce repetition
    device: str = "cuda"
    batch_size: int = 2  # Smaller batches for better processing
    num_workers: int = 16
    target_sample_rate: int = 16000
    audio_format: str = "wav"
    output_directory: str = field(default_factory=lambda: os.getcwd())
    enable_sentence_preview: bool = True
    preview_sentence_count: int = 2  # Reduced for cleaner output
    save_individual_parts: bool = False
    unified_filename_suffix: str = "_unified_transcription.txt"
    # Enhanced deduplication and quality parameters
    repetition_threshold: float = 0.85  # Stricter similarity threshold
    max_word_repetition: int = 2  # Stricter repetition limit
    min_chunk_confidence: float = 0.7  # Minimum confidence threshold
    noise_threshold: float = 0.4  # Skip low-confidence segments


class RepetitionDetector:
    """Advanced repetition detection and removal utility."""
    
    @staticmethod
    def detect_word_repetition(text: str, max_repetitions: int = 3) -> str:
        """Remove excessive word repetitions from text."""
        if not text:
            return text
            
        words = text.split()
        if len(words) <= 1:
            return text
            
        cleaned_words = []
        i = 0
        
        while i < len(words):
            current_word = words[i]
            cleaned_words.append(current_word)
            
            # Count consecutive repetitions
            repetition_count = 1
            j = i + 1
            
            while j < len(words) and words[j] == current_word:
                repetition_count += 1
                j += 1
            
            # Skip excessive repetitions
            if repetition_count > max_repetitions:
                i = j  # Skip all repetitions beyond the limit
            else:
                # Add remaining valid repetitions
                for _ in range(min(repetition_count - 1, max_repetitions - 1)):
                    if j > i + 1:
                        cleaned_words.append(current_word)
                i = j
        
        return ' '.join(cleaned_words)
    
    @staticmethod
    def detect_phrase_repetition(text: str, min_phrase_length: int = 3) -> str:
        """Remove repetitive phrases from text."""
        if not text:
            return text
            
        words = text.split()
        if len(words) < min_phrase_length * 2:
            return text
            
        cleaned_words = []
        i = 0
        
        while i < len(words):
            # Try different phrase lengths
            found_repetition = False
            
            for phrase_len in range(min_phrase_length, min(10, len(words) - i + 1)):
                if i + phrase_len * 2 > len(words):
                    break
                    
                phrase1 = words[i:i + phrase_len]
                phrase2 = words[i + phrase_len:i + phrase_len * 2]
                
                if phrase1 == phrase2:
                    # Found repetition, add only one instance
                    cleaned_words.extend(phrase1)
                    
                    # Skip all subsequent repetitions of this phrase
                    j = i + phrase_len * 2
                    while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase1:
                        j += phrase_len
                    
                    i = j
                    found_repetition = True
                    break
            
            if not found_repetition:
                cleaned_words.append(words[i])
                i += 1
        
        return ' '.join(cleaned_words)
    
    @staticmethod
    def similarity_ratio(text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @classmethod
    def clean_repetitive_text(cls, text: str, config: TranscriptionConfig) -> str:
        """Comprehensive repetition cleaning."""
        if not text:
            return text
            
        # Step 1: Remove excessive word repetitions
        cleaned = cls.detect_word_repetition(text, config.max_word_repetition)
        
        # Step 2: Remove phrase repetitions
        cleaned = cls.detect_phrase_repetition(cleaned)
        
        # Step 3: Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned


class SentenceExtractor:
    """Enhanced utility class for extracting and processing sentences."""
    
    @staticmethod
    def extract_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """Extract and clean sentences from transcription text."""
        if not text or not text.strip():
            return []
        
        # Clean the text first
        cleaned_text = RepetitionDetector.clean_repetitive_text(
            text.strip(), 
            TranscriptionConfig()  # Use default config for cleaning
        )
        
        # Enhanced sentence splitting for Persian text
        sentence_pattern = r'[.!?؟]+(?:\s+|$)'
        sentences = re.split(sentence_pattern, cleaned_text)
        
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Minimum meaningful length
                valid_sentences.append(sentence)
        
        return valid_sentences[:max_sentences]
    
    @staticmethod
    def format_sentence_preview(sentences: List[str], part_number: int) -> str:
        """Format sentences for preview display."""
        if not sentences:
            return f"Part {part_number}: [No meaningful content]"
        
        preview_lines = [f"Part {part_number} Preview:"]
        for i, sentence in enumerate(sentences, 1):
            # Truncate very long sentences for preview
            display_sentence = sentence[:100] + "..." if len(sentence) > 100 else sentence
            preview_lines.append(f"  {i}. {display_sentence}")
        
        return "\n".join(preview_lines)


class TranscriptionFileManager:
    """Enhanced file management with text post-processing capabilities."""
    
    def __init__(self, base_filename: str, output_directory: str, config: TranscriptionConfig):
        self.base_filename = base_filename
        self.output_directory = Path(output_directory)
        self.config = config
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        self.unified_file_path = self.output_directory / f"{base_filename}{config.unified_filename_suffix}"
        self.cleaned_file_path = self.output_directory / f"{base_filename}_cleaned_transcription.txt"
        
    def save_unified_transcription(self, content: str) -> bool:
        """Save unified transcription with cleaning."""
        try:
            # Save original version
            with open(self.unified_file_path, "w", encoding="utf-8") as file:
                file.write(content)
            
            # Create and save cleaned version
            cleaned_content = RepetitionDetector.clean_repetitive_text(content, self.config)
            with open(self.cleaned_file_path, "w", encoding="utf-8") as file:
                file.write(cleaned_content)
            
            return True
        except Exception as e:
            logging.error(f"Failed to save transcription: {e}")
            return False
    
    def get_transcription_info(self) -> Dict[str, Any]:
        """Get comprehensive information about transcription files."""
        info = {
            "base_filename": self.base_filename,
            "unified_file_path": str(self.unified_file_path),
            "cleaned_file_path": str(self.cleaned_file_path),
            "output_directory": str(self.output_directory),
        }
        
        # Check both files
        for file_type, file_path in [("original", self.unified_file_path), ("cleaned", self.cleaned_file_path)]:
            info[f"{file_type}_exists"] = file_path.exists()
            info[f"{file_type}_size"] = 0
            info[f"{file_type}_characters"] = 0
            info[f"{file_type}_words"] = 0
            
            if info[f"{file_type}_exists"]:
                try:
                    info[f"{file_type}_size"] = file_path.stat().st_size
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        info[f"{file_type}_characters"] = len(content)
                        info[f"{file_type}_words"] = len(content.split())
                except Exception as e:
                    logging.error(f"Error reading {file_type} file info: {e}")
        
        return info


class UnifiedAudioTranscriber:
    """Enhanced transcription system with advanced repetition handling."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.model = None
        self.sentence_extractor = SentenceExtractor()
        self.repetition_detector = RepetitionDetector()
        
        # Initialize GPU and model
        self._setup_gpu()
        self._load_model()
    
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
    
    def _prepare_audio(self, audio_file_path: str) -> AudioSegment:
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
    
    def _create_intelligent_chunks(self, audio: AudioSegment) -> List[Tuple[int, int]]:
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
    
    def _transcribe_chunk_with_retry(self, chunk: np.ndarray, max_retries: int = 2) -> str:
        """Transcribe chunk with anti-repetition measures."""
        for attempt in range(max_retries + 1):
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
                    temperature=0.0,  # Deterministic output to reduce randomness
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    logprob_threshold=-1.0,
                    compression_ratio_threshold=2.0  # Stricter threshold
                )
                
                transcribed_text = result["text"].strip()
                
                # Immediate repetition cleaning
                cleaned_text = RepetitionDetector.clean_repetitive_text(transcribed_text, self.config)
                
                return cleaned_text
                
            except Exception as e:
                if attempt < max_retries:
                    self.logger.warning(f"Transcription attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(0.1 * (attempt + 1))
                else:
                    self.logger.error(f"Transcription failed after {max_retries + 1} attempts: {e}")
                    return ""
        
        return ""
    
    def _merge_transcriptions_with_deduplication(self, transcriptions: List[str]) -> str:
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
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        self.logger.info(f"Merged {len(valid_transcriptions)} segments with deduplication")
        return final_text
    
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
                        result = self._transcribe_chunk_with_retry(chunk_array)
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
            audio = self._prepare_audio(audio_file_path)
            
            # Phase 2: Intelligent chunking
            print("🔄 Phase 2: Creating optimized segments...")
            chunks = self._create_intelligent_chunks(audio)
            
            print(f"📊 Processing {len(chunks)} segments • Duration: {len(audio)/1000:.1f}s")
            
            # Phase 3: Enhanced transcription
            print("🔄 Phase 3: Transcribing with repetition control...")
            transcriptions = self._process_chunks_with_preview(audio, chunks, file_manager)
            
            # Phase 4: Advanced merging with deduplication
            print("\n🔄 Phase 4: Merging with deduplication...")
            final_transcription = self._merge_transcriptions_with_deduplication(transcriptions)
            
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


def create_optimized_config(model_size: str = "large", language: str = "fa", 
                          enable_preview: bool = True) -> TranscriptionConfig:
    """Create optimized configuration with anti-repetition settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optimize settings based on device
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        batch_size = max(2, int(gpu_memory // 4)) if model_size.startswith("large") else max(4, int(gpu_memory // 2))
    else:
        batch_size = 1
    
    return TranscriptionConfig(
        model_name=model_size,
        language=language,
        device=device,
        batch_size=min(batch_size, 6),
        overlap_ms=500,  # Reduced overlap
        enable_sentence_preview=enable_preview,
        save_individual_parts=False,
        repetition_threshold=0.8,
        max_word_repetition=3,
        output_directory=os.getcwd()
    )


def main():
    """Enhanced transcription with repetition control."""
    try:
        # Update this path to your audio file
        # audio_file_path = "your_audio_file.wav"
        
        if not os.path.exists(audio_file_path):
            print("⚠️  Update 'audio_file_path' variable with your actual file path")
            return
        
        # Create anti-repetition configuration
        config = create_optimized_config(
            model_size="large",
            language="fa",
            enable_preview=True
        )
        
        print("🚀 Starting Enhanced Transcription System")
        print(f"📁 Output: {config.output_directory}")
        print(f"🔧 Config: {config.model_name} model, {config.device} device")
        
        with UnifiedAudioTranscriber(config) as transcriber:
            transcription = transcriber.transcribe_file(audio_file_path)
            
            if transcription:
                print("\n📝 Final Transcription Preview:")
                print("-" * 60)
                preview_text = transcription[:300] + "..." if len(transcription) > 300 else transcription
                print(preview_text)
                print("-" * 60)
            
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"❌ Failed: {e}")
        logging.error(f"Execution error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
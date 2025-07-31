"""
Core transcription module for FarsiTranscribe.

This module contains the main transcription engine that coordinates audio processing,
model inference, and result generation.
"""

import logging
import time
import gc
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Callable
from tqdm import tqdm
import torch
import whisper

from .config import TranscriptionConfig
from .audio import AudioProcessor, AudioChunk
from .utils import TranscriptionResult, TextProcessor

logger = logging.getLogger(__name__)


class TranscriptionHooks:
    """
    Hooks for extending transcription functionality.
    
    This class allows users to register callbacks at various stages of transcription
    for custom processing, monitoring, or modification.
    """
    
    def __init__(self):
        self.pre_chunk_hooks: List[Callable] = []
        self.post_chunk_hooks: List[Callable] = []
        self.pre_merge_hooks: List[Callable] = []
        self.post_merge_hooks: List[Callable] = []
    
    def add_pre_chunk_hook(self, hook: Callable):
        """Add a hook to run before processing each chunk."""
        self.pre_chunk_hooks.append(hook)
    
    def add_post_chunk_hook(self, hook: Callable):
        """Add a hook to run after processing each chunk."""
        self.post_chunk_hooks.append(hook)
    
    def add_pre_merge_hook(self, hook: Callable):
        """Add a hook to run before merging results."""
        self.pre_merge_hooks.append(hook)
    
    def add_post_merge_hook(self, hook: Callable):
        """Add a hook to run after merging results."""
        self.post_merge_hooks.append(hook)


class FarsiTranscriber:
    """
    Main transcription engine for Persian/Farsi audio.
    
    This class orchestrates the entire transcription process, from loading audio
    to generating final transcripts. It's designed to be extensible through hooks
    and configurable through the TranscriptionConfig class.
    """
    
    def __init__(self, config: TranscriptionConfig):
        """
        Initialize the transcriber with given configuration.
        
        Args:
            config: TranscriptionConfig object containing all settings
        """
        self.config = config
        self.hooks = TranscriptionHooks()
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=config.sample_rate)
        self.text_processor = TextProcessor(language=config.language)
        
        # Initialize model
        self._init_model()
        
        # Memory management
        self._chunks_processed = 0
        
        logger.info(f"FarsiTranscriber initialized with model: {config.model_name}")
    
    def _init_model(self):
        """Initialize the Whisper model based on configuration."""
        logger.info(f"Loading Whisper model: {self.config.model_name}")
        
        # Load model
        self.model = whisper.load_model(
            self.config.model_name,
            device=self.config.device
        )
        
        # Apply optimizations
        if self.config.device == "cuda" and self.config.use_fp16:
            self.model = self.model.half()
        
        logger.info(f"Model loaded on {self.config.device}")
    
    def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult object containing transcription and metadata
        """
        start_time = time.time()
        audio_path = Path(audio_path)
        
        logger.info(f"Starting transcription of: {audio_path}")
        
        # Load audio
        audio_data, audio_metadata = self.audio_processor.load_audio(audio_path)
        
        # Create chunks
        chunks = self.audio_processor.create_chunks(
            audio_data,
            self.config.chunk_duration,
            self.config.overlap
        )
        
        # Process chunks
        chunk_results = self._process_chunks(chunks)
        
        # Merge results
        merged_text = self._merge_results(chunk_results)
        
        # Create final result
        result = TranscriptionResult(
            text=merged_text,
            chunks=chunk_results,
            metadata={
                **audio_metadata,
                'processing_time': time.time() - start_time,
                'model': self.config.model_name,
                'config': self.config.to_dict()
            }
        )
        
        # Apply post-processing
        if self.config.persian_normalization:
            result.text = self.text_processor.normalize_persian(result.text)
        
        logger.info(f"Transcription completed in {result.metadata['processing_time']:.1f}s")
        
        return result
    
    def transcribe_stream(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe audio using streaming approach for large files.
        
        Note: Currently loads the entire file to extract metadata. True streaming
        with metadata extraction from file headers is planned for future versions.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            TranscriptionResult object
        """
        start_time = time.time()
        chunk_results = []
        audio_path = Path(audio_path)
        
        # Extract basic metadata without loading full audio
        # For now, we get basic info from file
        audio_metadata = {
            'original_file': str(audio_path),
            'format': audio_path.suffix
        }
        
        # Try to get duration without loading full file
        try:
            from pydub.utils import mediainfo
            info = mediainfo(str(audio_path))
            audio_metadata['duration_seconds'] = float(info.get('duration', 0))
        except (ImportError, IOError, OSError, subprocess.CalledProcessError, Exception) as e:
            # Fallback - we'll update duration after processing
            logger.debug(f"Could not get media info: {e}")
            audio_metadata['duration_seconds'] = 0
        
        # Stream and process chunks
        for chunk in self.audio_processor.stream_chunks(
            audio_path,
            self.config.chunk_duration,
            self.config.overlap
        ):
            result = self._process_single_chunk(chunk)
            chunk_results.append(result)
            
            # Memory management
            self._manage_memory()
        
        # Update duration if we couldn't get it earlier
        if audio_metadata['duration_seconds'] == 0 and chunk_results:
            audio_metadata['duration_seconds'] = chunk_results[-1]['end_time']
        
        # Merge results
        merged_text = self._merge_results(chunk_results)
        
        # Create final result
        result = TranscriptionResult(
            text=merged_text,
            chunks=chunk_results,
            metadata={
                **audio_metadata,
                'processing_time': time.time() - start_time,
                'model': self.config.model_name,
                'config': self.config.to_dict()
            }
        )
        
        return result
    
    def _process_chunks(self, chunks: List[AudioChunk]) -> List[Dict[str, Any]]:
        """Process a list of audio chunks."""
        results = []
        
        # Progress bar
        with tqdm(total=len(chunks), desc="Transcribing", unit="chunk") as pbar:
            for chunk in chunks:
                result = self._process_single_chunk(chunk)
                results.append(result)
                pbar.update(1)
                
                # Memory management
                self._manage_memory()
        
        return results
    
    def _process_single_chunk(self, chunk: AudioChunk) -> Dict[str, Any]:
        """Process a single audio chunk."""
        # Run pre-chunk hooks
        for hook in self.hooks.pre_chunk_hooks:
            chunk = hook(chunk)
        
        # Transcribe
        result = self.model.transcribe(
            chunk.audio,
            language=self.config.language,
            task="transcribe",
            temperature=self.config.temperature,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
            logprob_threshold=self.config.logprob_threshold,
            no_speech_threshold=self.config.no_speech_threshold,
            condition_on_previous_text=self.config.condition_on_previous_text,
            verbose=False
        )
        
        # Create chunk result
        chunk_result = {
            'text': result['text'].strip(),
            'start_time': chunk.start_time,
            'end_time': chunk.end_time,
            'index': chunk.index,
            'segments': result.get('segments', []),
            'language': result.get('language', self.config.language)
        }
        
        # Run post-chunk hooks
        for hook in self.hooks.post_chunk_hooks:
            chunk_result = hook(chunk_result)
        
        self._chunks_processed += 1
        
        return chunk_result
    
    def _merge_results(self, chunk_results: List[Dict[str, Any]]) -> str:
        """Merge chunk results into final text."""
        # Run pre-merge hooks
        for hook in self.hooks.pre_merge_hooks:
            chunk_results = hook(chunk_results)
        
        # Sort by chunk index
        chunk_results.sort(key=lambda x: x['index'])
        
        # Extract texts
        texts = []
        for result in chunk_results:
            text = result['text'].strip()
            if text:
                texts.append(text)
        
        # Merge with space
        merged_text = ' '.join(texts)
        
        # Clean up extra spaces
        merged_text = ' '.join(merged_text.split())
        
        # Run post-merge hooks
        for hook in self.hooks.post_merge_hooks:
            merged_text = hook(merged_text)
        
        return merged_text
    
    def _manage_memory(self):
        """Manage memory usage during processing."""
        if self._chunks_processed % self.config.clear_cache_every == 0:
            gc.collect()
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            logger.debug(f"Memory cleanup performed after {self._chunks_processed} chunks")
    
    def add_extension(self, extension):
        """
        Add an extension to the transcriber.
        
        Extensions can add hooks and modify behavior without changing core code.
        
        Args:
            extension: Extension object with install() method
        """
        extension.install(self)
        logger.info(f"Extension installed: {extension.__class__.__name__}")
    
    def set_model(self, model_name: str):
        """
        Change the model being used.
        
        Args:
            model_name: Name of the new model to load
        """
        self.config.model_name = model_name
        self._init_model()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
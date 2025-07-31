"""
Core transcription module for FarsiTranscribe.

This module contains the main transcription engine that coordinates audio processing,
model inference, and result generation.
"""

import logging
import time
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import torch
import whisper
import numpy as np

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
        """
        Initialize hook lists for pre/post chunk and pre/post merge stages in the transcription process.
        """
        self.pre_chunk_hooks: List[Callable] = []
        self.post_chunk_hooks: List[Callable] = []
        self.pre_merge_hooks: List[Callable] = []
        self.post_merge_hooks: List[Callable] = []
    
    def add_pre_chunk_hook(self, hook: Callable):
        """
        Register a callback to be executed before processing each audio chunk.
        
        Parameters:
            hook (Callable): A function to be called with the audio chunk before processing.
        """
        self.pre_chunk_hooks.append(hook)
    
    def add_post_chunk_hook(self, hook: Callable):
        """
        Register a callback to be executed after processing each audio chunk.
        
        Parameters:
            hook (Callable): A function to be called with the chunk result after each chunk is processed.
        """
        self.post_chunk_hooks.append(hook)
    
    def add_pre_merge_hook(self, hook: Callable):
        """
        Register a callback to be executed before merging transcription chunk results.
        
        Parameters:
            hook (Callable): A function to be called with the list of chunk results before merging.
        """
        self.pre_merge_hooks.append(hook)
    
    def add_post_merge_hook(self, hook: Callable):
        """
        Register a callback to be executed after merging transcription results.
        
        Parameters:
            hook (Callable): A function to be called with the merged transcription text after merging is complete.
        """
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
        Initialize the FarsiTranscriber with the specified configuration, setting up audio and text processors, model, and hooks for extensibility.
        
        Parameters:
            config (TranscriptionConfig): Configuration object specifying model, language, sample rate, and other transcription settings.
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
        """
        Loads the Whisper model specified in the configuration onto the configured device, applying half-precision if enabled for CUDA.
        """
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
        Transcribes a complete audio file and returns the transcription result with metadata.
        
        Processes the audio by loading, chunking, transcribing each chunk, merging results, and applying optional Persian text normalization. Returns a `TranscriptionResult` containing the final transcript, chunk details, and processing metadata.
        
        Parameters:
            audio_path (Path): Path to the audio file to be transcribed.
        
        Returns:
            TranscriptionResult: The transcription output, including text, chunk-level results, and metadata.
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
        Transcribes an audio file using a streaming approach, processing and transcribing each chunk sequentially to handle large files efficiently.
        
        Parameters:
            audio_path (Path): Path to the audio file to be transcribed.
        
        Returns:
            TranscriptionResult: The final transcription result, including merged text, chunk details, and metadata.
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
        except:
            # Fallback - we'll update duration after processing
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
        """
        Processes a list of audio chunks sequentially and returns their transcription results.
        
        Parameters:
            chunks (List[AudioChunk]): List of audio chunks to be transcribed.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing transcription results for each chunk.
        """
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
        """
        Processes a single audio chunk through the transcription model and applies registered pre- and post-processing hooks.
        
        Parameters:
            chunk (AudioChunk): The audio chunk to be transcribed.
        
        Returns:
            Dict[str, Any]: A dictionary containing the transcribed text, timing, chunk index, segments, and detected language.
        """
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
        """
        Merge the transcribed texts from all audio chunks into a single final transcription string.
        
        Runs pre-merge and post-merge hooks for extensibility. Chunks are sorted by their original order, and only non-empty texts are included in the merged result. Extra spaces are removed from the final output.
        
        Parameters:
            chunk_results (List[Dict[str, Any]]): List of dictionaries containing transcription results for each audio chunk.
        
        Returns:
            str: The merged transcription text.
        """
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
        """
        Performs periodic memory cleanup by triggering garbage collection and clearing the CUDA cache after a configured number of processed chunks.
        """
        if self._chunks_processed % self.config.clear_cache_every == 0:
            gc.collect()
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            logger.debug(f"Memory cleanup performed after {self._chunks_processed} chunks")
    
    def add_extension(self, extension):
        """
        Installs an extension that can add hooks or modify the transcriber's behavior.
        
        The extension must provide an `install()` method, which will be called with the transcriber instance.
        """
        extension.install(self)
        logger.info(f"Extension installed: {extension.__class__.__name__}")
    
    def set_model(self, model_name: str):
        """
        Switches the Whisper model to the specified model name and reinitializes the model accordingly.
        
        Parameters:
            model_name (str): The name of the new Whisper model to load.
        """
        self.config.model_name = model_name
        self._init_model()
    
    def __enter__(self):
        """
        Enter the context manager for the transcriber, enabling resource management with a `with` statement.
        
        Returns:
            FarsiTranscriber: The current instance for use within the context.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Performs resource cleanup when exiting the context manager, including garbage collection and clearing the CUDA cache if applicable.
        """
        gc.collect()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
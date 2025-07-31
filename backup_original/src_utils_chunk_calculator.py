"""
Unified chunk calculation system for consistent audio processing.
This module provides a single source of truth for chunk calculations
across the entire transcription system.
"""

import math
from typing import List, Tuple, Iterator
import numpy as np
from dataclasses import dataclass


@dataclass
class ChunkInfo:
    """Information about a single audio chunk."""
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    is_final: bool


class UnifiedChunkCalculator:
    """
    Unified chunk calculator that provides consistent chunk calculation
    across the entire transcription system.
    """
    
    def __init__(self, chunk_duration_ms: int, overlap_ms: int):
        """
        Initialize the chunk calculator.
        
        Args:
            chunk_duration_ms: Duration of each chunk in milliseconds
            overlap_ms: Overlap between chunks in milliseconds
        """
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.effective_chunk_duration = max(1, chunk_duration_ms - overlap_ms)
        
        # Validate parameters
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        if overlap_ms < 0:
            raise ValueError("overlap_ms cannot be negative")
        if overlap_ms >= chunk_duration_ms:
            raise ValueError("overlap_ms must be less than chunk_duration_ms")
    
    def calculate_total_chunks(self, audio_duration_ms: int) -> int:
        """
        Calculate the exact number of chunks that will be generated.
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            Total number of chunks
        """
        if audio_duration_ms <= 0:
            return 0
        
        if audio_duration_ms <= self.chunk_duration_ms:
            return 1
        
        # Calculate using the unified algorithm
        total_chunks = 0
        start_ms = 0
        
        while start_ms < audio_duration_ms:
            end_ms = min(start_ms + self.chunk_duration_ms, audio_duration_ms)
            total_chunks += 1
            
            # Move to next chunk with overlap
            start_ms = end_ms - self.overlap_ms
            if start_ms >= audio_duration_ms:
                break
        
        return total_chunks
    
    def generate_chunk_boundaries(self, audio_duration_ms: int) -> List[Tuple[int, int]]:
        """
        Generate all chunk boundaries for the audio.
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            List of (start_ms, end_ms) tuples
        """
        if audio_duration_ms <= 0:
            return []
        
        if audio_duration_ms <= self.chunk_duration_ms:
            return [(0, audio_duration_ms)]
        
        boundaries = []
        start_ms = 0
        
        while start_ms < audio_duration_ms:
            end_ms = min(start_ms + self.chunk_duration_ms, audio_duration_ms)
            boundaries.append((start_ms, end_ms))
            
            # Move to next chunk with overlap
            start_ms = end_ms - self.overlap_ms
            if start_ms >= audio_duration_ms:
                break
        
        return boundaries
    
    def generate_chunk_info(self, audio_duration_ms: int) -> List[ChunkInfo]:
        """
        Generate detailed information about each chunk.
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            List of ChunkInfo objects
        """
        boundaries = self.generate_chunk_boundaries(audio_duration_ms)
        chunk_info = []
        
        for i, (start_ms, end_ms) in enumerate(boundaries):
            duration_ms = end_ms - start_ms
            is_final = (i == len(boundaries) - 1)
            
            chunk_info.append(ChunkInfo(
                index=i,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                is_final=is_final
            ))
        
        return chunk_info
    
    def get_chunk_analysis(self, audio_duration_ms: int) -> dict:
        """
        Get comprehensive analysis of chunking for the audio (optimized for speed).
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            Dictionary with chunk analysis information
        """
        total_chunks = self.calculate_total_chunks(audio_duration_ms)
        
        # Optimized: Calculate statistics without generating all chunk info
        if audio_duration_ms <= self.chunk_duration_ms:
            # Single chunk case
            avg_duration = audio_duration_ms
            min_duration = audio_duration_ms
            max_duration = audio_duration_ms
        else:
            # Multiple chunks case - calculate efficiently
            effective_chunk_duration = self.effective_chunk_duration
            
            # Most chunks will be full size
            full_chunks = (audio_duration_ms - self.chunk_duration_ms) // effective_chunk_duration
            remaining_duration = audio_duration_ms - (full_chunks * effective_chunk_duration)
            
            if full_chunks > 0:
                avg_duration = (full_chunks * self.chunk_duration_ms + remaining_duration) / total_chunks
                min_duration = min(self.chunk_duration_ms, remaining_duration) if remaining_duration > 0 else self.chunk_duration_ms
                max_duration = self.chunk_duration_ms
            else:
                avg_duration = audio_duration_ms
                min_duration = audio_duration_ms
                max_duration = audio_duration_ms
        
        return {
            'total_chunks': total_chunks,
            'chunk_duration_ms': self.chunk_duration_ms,
            'overlap_ms': self.overlap_ms,
            'effective_chunk_duration_ms': self.effective_chunk_duration,
            'audio_duration_ms': audio_duration_ms,
            'audio_duration_seconds': audio_duration_ms / 1000,
            'avg_chunk_duration_ms': avg_duration,
            'min_chunk_duration_ms': min_duration,
            'max_chunk_duration_ms': max_duration,
            'chunk_info': None,  # Don't generate detailed info unless needed
            'boundaries': None   # Don't generate boundaries unless needed
        }
    
    def estimate_memory_usage(self, audio_duration_ms: int, sample_rate: int = 16000, channels: int = 1) -> dict:
        """
        Estimate memory usage for processing the audio (optimized).
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            
        Returns:
            Dictionary with memory usage estimates
        """
        total_chunks = self.calculate_total_chunks(audio_duration_ms)
        
        # Calculate memory per chunk (assuming float32 samples)
        bytes_per_sample = 4  # float32
        samples_per_chunk = (self.chunk_duration_ms / 1000) * sample_rate * channels
        memory_per_chunk_mb = (samples_per_chunk * bytes_per_sample) / (1024 * 1024)
        
        # Estimate peak memory usage (assuming max 3 chunks in memory at once)
        max_concurrent_chunks = min(3, total_chunks)
        peak_memory_mb = memory_per_chunk_mb * max_concurrent_chunks
        
        # Estimate total memory for all chunks
        total_memory_mb = memory_per_chunk_mb * total_chunks
        
        return {
            'memory_per_chunk_mb': memory_per_chunk_mb,
            'peak_memory_mb': peak_memory_mb,
            'total_memory_mb': total_memory_mb,
            'max_concurrent_chunks': max_concurrent_chunks,
            'samples_per_chunk': samples_per_chunk,
            'bytes_per_sample': bytes_per_sample
        }
    
    def estimate_processing_time(self, audio_duration_ms: int, model_name: str = "small") -> dict:
        """
        Estimate processing time based on model and chunk count (optimized).
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            model_name: Whisper model name
            
        Returns:
            Dictionary with time estimates
        """
        total_chunks = self.calculate_total_chunks(audio_duration_ms)
        
        # Time estimates per chunk based on model size (seconds)
        time_per_chunk = {
            "tiny": 2,
            "base": 4,
            "small": 8,
            "medium": 15,
            "large": 25,
            "large-v2": 30,
            "large-v3": 35
        }.get(model_name, 8)
        
        total_time_seconds = total_chunks * time_per_chunk
        total_time_minutes = total_time_seconds / 60
        
        return {
            'time_per_chunk_seconds': time_per_chunk,
            'total_time_seconds': total_time_seconds,
            'total_time_minutes': total_time_minutes,
            'model_name': model_name,
            'chunks_per_minute': 60 / time_per_chunk if time_per_chunk > 0 else 0
        }
    
    def get_fast_analysis(self, audio_duration_ms: int) -> dict:
        """
        Get fast analysis for initial display (optimized for speed).
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            Dictionary with basic chunk analysis information
        """
        total_chunks = self.calculate_total_chunks(audio_duration_ms)
        
        return {
            'total_chunks': total_chunks,
            'chunk_duration_ms': self.chunk_duration_ms,
            'overlap_ms': self.overlap_ms,
            'effective_chunk_duration_ms': self.effective_chunk_duration,
            'audio_duration_ms': audio_duration_ms,
            'audio_duration_seconds': audio_duration_ms / 1000,
            'avg_chunk_duration_ms': self.chunk_duration_ms,  # Approximate
            'min_chunk_duration_ms': self.chunk_duration_ms,  # Approximate
            'max_chunk_duration_ms': self.chunk_duration_ms   # Approximate
        }


def create_chunk_calculator(config) -> UnifiedChunkCalculator:
    """
    Factory function to create a chunk calculator from configuration.
    
    Args:
        config: TranscriptionConfig object
        
    Returns:
        UnifiedChunkCalculator instance
    """
    return UnifiedChunkCalculator(
        chunk_duration_ms=config.chunk_duration_ms,
        overlap_ms=config.overlap_ms
    ) 
"""
Simplified and optimized chunk calculation system.
Replaces the complex original chunk_calculator.py with efficient algorithms.
"""

import math
from typing import List, Tuple


class ChunkCalculator:
    """Efficient chunk calculator with minimal overhead."""
    
    def __init__(self, chunk_duration_ms: int, overlap_ms: int):
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
        self.step_size_ms = chunk_duration_ms - overlap_ms
        
        if chunk_duration_ms <= 0 or overlap_ms < 0 or overlap_ms >= chunk_duration_ms:
            raise ValueError("Invalid chunk parameters")
    
    def calculate_total_chunks(self, audio_duration_ms: int) -> int:
        """Calculate total chunks using efficient formula."""
        if audio_duration_ms <= 0:
            return 0
        if audio_duration_ms <= self.chunk_duration_ms:
            return 1
        
        remaining = audio_duration_ms - self.chunk_duration_ms
        return math.ceil(remaining / self.step_size_ms) + 1
    
    def generate_chunk_boundaries(self, audio_duration_ms: int) -> List[Tuple[int, int]]:
        """Generate chunk boundaries efficiently."""
        if audio_duration_ms <= 0:
            return []
        if audio_duration_ms <= self.chunk_duration_ms:
            return [(0, audio_duration_ms)]
        
        boundaries = []
        start_ms = 0
        
        while start_ms < audio_duration_ms:
            end_ms = min(start_ms + self.chunk_duration_ms, audio_duration_ms)
            boundaries.append((start_ms, end_ms))
            start_ms += self.step_size_ms
            
            if start_ms >= audio_duration_ms:
                break
        
        return boundaries


def create_chunk_calculator(config) -> ChunkCalculator:
    """Factory function for chunk calculator."""
    return ChunkCalculator(config.chunk_duration_ms, config.overlap_ms)

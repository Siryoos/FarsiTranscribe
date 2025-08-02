"""
Configuration management for transcription system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TranscriptionConfig:
    """Enhanced configuration class for transcription parameters."""
    
    # Model settings
    model_name: str = "large-v3"
    language: str = "fa"
    
    # Audio processing
    chunk_duration_ms: int = 20000
    overlap_ms: int = 200
    target_sample_rate: int = 16000
    audio_format: str = "wav"
    
    # Processing settings - OPTIMIZED FOR CPU
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = field(default_factory=lambda: TranscriptionConfig._get_optimal_batch_size())
    num_workers: int = field(default_factory=lambda: min(6, os.cpu_count() or 4))  # Increased from 2 to 6
    
    # Output settings
    output_directory: str = field(default_factory=lambda: os.getcwd())
    save_individual_parts: bool = False
    unified_filename_suffix: str = "_unified_transcription.txt"
    
    # Preview settings
    enable_sentence_preview: bool = True
    preview_sentence_count: int = 2
    
    # Quality and deduplication settings
    repetition_threshold: float = 0.85
    max_word_repetition: int = 2
    min_chunk_confidence: float = 0.7
    noise_threshold: float = 0.4
    
    # Advanced settings
    temperature: float = 0.0
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.0
    
    # CPU Optimization settings
    use_parallel_audio_prep: bool = True
    chunk_prefetch_count: int = 4
    memory_efficient_mode: bool = True
    
    # Memory Management settings
    memory_threshold_mb: int = 1024  # 1GB threshold for cleanup
    cleanup_interval_seconds: int = 30  # Cleanup every 30 seconds
    streaming_chunk_size_mb: int = 50  # 50MB chunks for streaming
    enable_memory_monitoring: bool = True
    
    # Audio Preprocessing settings (Quick Wins)
    enable_preprocessing: bool = True
    enable_noise_reduction: bool = True
    enable_voice_activity_detection: bool = True
    enable_speech_enhancement: bool = True
    use_smart_chunking: bool = True
    
    # Advanced Preprocessing settings
    enable_advanced_preprocessing: bool = False
    enable_facebook_denoiser: bool = False
    enable_persian_optimization: bool = True
    adaptive_processing: bool = True

    def __post_init__(self):
        """Validate and optimize configuration after initialization."""
        self._validate_config()
        self._optimize_for_device()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        
        if self.overlap_ms < 0:
            raise ValueError("overlap_ms cannot be negative")
        
        if self.overlap_ms >= self.chunk_duration_ms:
            raise ValueError("overlap_ms must be less than chunk_duration_ms")
        
        if self.repetition_threshold < 0 or self.repetition_threshold > 1:
            raise ValueError("repetition_threshold must be between 0 and 1")
        
        if self.max_word_repetition < 1:
            raise ValueError("max_word_repetition must be at least 1")
    
    def _optimize_for_device(self):
        """Optimize settings based on available device."""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.batch_size = max(2, int(gpu_memory // 4)) if self.model_name.startswith("large") else max(4, int(gpu_memory // 2))
            self.batch_size = min(self.batch_size, 6)
        else:
            self.device = "cpu"
            # CPU optimizations
            self.batch_size = 1  # Sequential processing for CPU
            self.num_workers = min(6, os.cpu_count() or 4)  # Use more CPU cores
            self.use_parallel_audio_prep = True
            self.memory_efficient_mode = True
    
    @staticmethod
    def _get_optimal_batch_size() -> int:
        """Get optimal batch size based on available resources."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return max(2, int(gpu_memory // 4))
        return 1
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model_name': self.model_name,
            'language': self.language,
            'chunk_duration_ms': self.chunk_duration_ms,
            'overlap_ms': self.overlap_ms,
            'device': self.device,
            'batch_size': self.batch_size,
            'output_directory': self.output_directory,
            'repetition_threshold': self.repetition_threshold,
            'max_word_repetition': self.max_word_repetition
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TranscriptionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def clone(self) -> 'TranscriptionConfig':
        """Create a copy of the configuration."""
        return TranscriptionConfig(**self.to_dict())


class ConfigFactory:
    """Factory class for creating optimized configurations."""
    
    @staticmethod
    def create_optimized_config(
        model_size: str = "large",
        language: str = "fa",
        enable_preview: bool = True,
        output_dir: Optional[str] = None
    ) -> TranscriptionConfig:
        """Create optimized configuration with anti-repetition settings."""
        config = TranscriptionConfig(
            model_name=model_size,
            language=language,
            enable_sentence_preview=enable_preview,
            overlap_ms=500,
            repetition_threshold=0.8,
            max_word_repetition=3,
            num_workers=6  # Increased from 2 to 6
        )
        
        if output_dir:
            config.output_directory = output_dir
            
        return config
    @staticmethod
    def create_advanced_persian_config() -> TranscriptionConfig:
        """Create configuration with advanced Persian preprocessing."""
        return TranscriptionConfig(
            model_name="large-v3",
            language="fa",
            chunk_duration_ms=25000,
            overlap_ms=300,
            num_workers=6,
            repetition_threshold=0.85,
            max_word_repetition=2,
            temperature=0.0,
            condition_on_previous_text=True,
            enable_preprocessing=True,
            enable_advanced_preprocessing=True,
            enable_facebook_denoiser=True,
            enable_persian_optimization=True,
            adaptive_processing=True,
            use_smart_chunking=True
        )
    
    @staticmethod
    def create_facebook_denoiser_config() -> TranscriptionConfig:
        """Create configuration with Facebook Denoiser for noisy audio."""
        return TranscriptionConfig(
            model_name="large-v3",
            language="fa",
            chunk_duration_ms=20000,
            overlap_ms=400,
            enable_preprocessing=True,
            enable_advanced_preprocessing=True,
            enable_facebook_denoiser=True,
            enable_persian_optimization=True,
            adaptive_processing=True
        )
    
    @staticmethod
    def create_cpu_optimized_config() -> TranscriptionConfig:
        """Create configuration specifically optimized for CPU-only systems."""
        return TranscriptionConfig(
            model_name="medium",  # Smaller model for CPU performance
            language="fa",
            chunk_duration_ms=25000,  # Larger chunks = fewer processing steps
            overlap_ms=300,
            num_workers=6,  # Use more CPU cores
            repetition_threshold=0.8,
            max_word_repetition=2,
            min_chunk_confidence=0.6,  # Slightly lower threshold for speed
            temperature=0.0,
            condition_on_previous_text=False,  # Disable for speed
            no_speech_threshold=0.7,  # Higher threshold to skip silence
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.0,
            enable_sentence_preview=True,
            use_parallel_audio_prep=True,
            memory_efficient_mode=True,
            chunk_prefetch_count=6,
            memory_threshold_mb=512,  # Lower threshold for CPU
            cleanup_interval_seconds=20,  # More frequent cleanup
            enable_memory_monitoring=True
        )
    
    @staticmethod
    def create_memory_optimized_config() -> TranscriptionConfig:
        """Create memory-optimized configuration for systems with limited RAM."""
        return TranscriptionConfig(
            model_name="small",  # Smallest model for memory efficiency
            device="cpu",
            chunk_duration_ms=10000,  # Very small chunks
            overlap_ms=50,
            batch_size=1,
            num_workers=min(2, os.cpu_count() or 1),  # Minimal parallelization
            use_parallel_audio_prep=False,  # Disable parallel audio prep
            memory_efficient_mode=True,
            enable_preprocessing=True,  # Enable preprocessing for better quality
            enable_noise_reduction=True,
            enable_voice_activity_detection=True,
            enable_speech_enhancement=True,
            use_smart_chunking=True,  # Enable smart chunking
            adaptive_processing=True,
            memory_threshold_mb=512,  # Moderate threshold
            cleanup_interval_seconds=15,  # Frequent cleanup
            streaming_chunk_size_mb=25,  # Small streaming chunks
            enable_memory_monitoring=True,
            enable_sentence_preview=True  # Enable preview for user feedback
        )
    
    @staticmethod
    def create_fast_config() -> TranscriptionConfig:
        """Create configuration optimized for speed."""
        return TranscriptionConfig(
            model_name="base",  # Smaller model for speed
            chunk_duration_ms=30000,  # Larger chunks = fewer processing steps
            overlap_ms=100,
            batch_size=1,
            num_workers=6,  # Increased from 2 to 6
            enable_sentence_preview=False,
            use_parallel_audio_prep=True,
            memory_efficient_mode=True
        )
    
    @staticmethod
    def create_high_quality_config() -> TranscriptionConfig:
        """Create configuration optimized for quality."""
        return TranscriptionConfig(
            model_name="large-v3",
            chunk_duration_ms=15000,
            overlap_ms=300,
            num_workers=6,  # Increased from 2 to 6
            repetition_threshold=0.9,
            max_word_repetition=1,
            min_chunk_confidence=0.8,
            temperature=0.0,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.0,
            use_parallel_audio_prep=True,
            memory_efficient_mode=False  # Quality over memory efficiency
        )
    
    @staticmethod
    def create_persian_optimized_config() -> TranscriptionConfig:
        """Create configuration specifically optimized for Persian transcription."""
        return TranscriptionConfig(
            model_name="large-v3",
            language="fa",
            chunk_duration_ms=10000,
            overlap_ms=100,
            num_workers=8,  # Increased from 2 to 6
            repetition_threshold=0.85,
            max_word_repetition=2,
            min_chunk_confidence=0.7,
            temperature=0.01,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.0,
            enable_sentence_preview=True,
            use_parallel_audio_prep=True,
            memory_efficient_mode=True,
            enable_advanced_preprocessing=True,
            enable_facebook_denoiser=True,
            enable_persian_optimization=True,
            adaptive_processing=True,
            use_smart_chunking=True,
            enable_preprocessing=True,
            enable_noise_reduction=True,
            enable_voice_activity_detection=True,
            enable_speech_enhancement=True,
        ) 
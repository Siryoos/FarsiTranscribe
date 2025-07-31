"""
Configuration management for FarsiTranscribe.

This module provides configuration classes and presets for different use cases.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import os


@dataclass
class TranscriptionConfig:
    """
    Configuration for audio transcription.
    
    This class holds all configurable parameters for the transcription process,
    including model settings, audio processing parameters, and output options.
    """
    
    # Model settings
    model_name: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: str = "fa"  # Language code (fa for Farsi/Persian)
    device: Optional[str] = None  # cuda, cpu, or None for auto-detect
    
    # Audio processing
    chunk_duration: int = 30  # Duration of each chunk in seconds
    overlap: int = 3  # Overlap between chunks in seconds
    sample_rate: int = 16000  # Sample rate for audio processing
    
    # Performance settings
    batch_size: int = 1  # Number of chunks to process in parallel
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    use_fp16: bool = False  # Use half precision for GPU
    
    # Output settings
    output_directory: Path = field(default_factory=lambda: Path("./output"))
    save_segments: bool = False  # Save individual segment files
    output_formats: list = field(default_factory=lambda: ["txt", "json"])
    
    # Quality settings
    temperature: float = 0.0  # Model temperature (0 for deterministic)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    
    # Persian-specific optimizations
    persian_normalization: bool = True  # Normalize Persian text
    remove_diacritics: bool = False  # Remove Arabic diacritics
    
    # Memory management
    max_memory_gb: float = 4.0  # Maximum memory usage in GB
    clear_cache_every: int = 10  # Clear cache every N chunks
    
    def __post_init__(self):
        """
        Performs post-initialization setup for the configuration, including device selection, FP16 optimization, output directory creation, and parameter validation.
        """
        # Auto-detect device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Enable FP16 on GPU by default
        if self.device == "cuda" and self.use_fp16 is None:
            self.use_fp16 = True
        
        # Create output directory if it doesn't exist
        self.output_directory = Path(self.output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate settings
        self._validate()
    
    def _validate(self):
        """
        Checks configuration parameters for invalid values and raises ValueError if any constraints are violated.
        """
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative") 
        
        if self.overlap >= self.chunk_duration:
            raise ValueError("overlap must be less than chunk_duration")
        
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration instance to a dictionary, converting any Path attributes to strings.
        
        Returns:
            dict: A dictionary representation of the configuration.
        """
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TranscriptionConfig':
        """
        Create a TranscriptionConfig instance from a dictionary, converting string paths to Path objects as needed.
        
        Parameters:
            config_dict (dict): Dictionary of configuration parameters.
        
        Returns:
            TranscriptionConfig: A new configuration instance with values from the dictionary.
        """
        # Convert string paths back to Path objects
        if 'output_directory' in config_dict:
            config_dict['output_directory'] = Path(config_dict['output_directory'])
        return cls(**config_dict)


class ConfigPresets:
    """
    Predefined configuration presets for common use cases.
    
    This class provides factory methods for creating optimized configurations
    for different scenarios like speed, quality, or memory efficiency.
    """
    
    @staticmethod
    def fast() -> TranscriptionConfig:
        """
        Return a TranscriptionConfig preset optimized for fast transcription using the base model, large chunk duration, minimal overlap, and Persian normalization enabled.
        """
        return TranscriptionConfig(
            model_name="base",
            chunk_duration=60,
            overlap=2,
            batch_size=4,
            condition_on_previous_text=False,
            persian_normalization=True
        )
    
    @staticmethod
    def balanced() -> TranscriptionConfig:
        """
        Return a configuration preset optimized for balanced transcription quality and speed.
        
        Uses the "small" model with moderate chunk duration and overlap, batch size of 2, zero temperature, and Persian normalization enabled.
        Returns:
            TranscriptionConfig: Configuration instance with balanced settings.
        """
        return TranscriptionConfig(
            model_name="small",
            chunk_duration=45,
            overlap=3,
            batch_size=2,
            temperature=0.0,
            persian_normalization=True
        )
    
    @staticmethod
    def high_quality() -> TranscriptionConfig:
        """
        Return a TranscriptionConfig preset optimized for high-quality transcription.
        
        Uses the "medium" model with shorter chunk duration, increased overlap, single-batch processing, zero temperature, conditioning on previous text, Persian normalization, and a stricter compression ratio threshold for improved accuracy.
        """
        return TranscriptionConfig(
            model_name="medium",
            chunk_duration=30,
            overlap=5,
            batch_size=1,
            temperature=0.0,
            condition_on_previous_text=True,
            persian_normalization=True,
            compression_ratio_threshold=2.0
        )
    
    @staticmethod
    def memory_efficient() -> TranscriptionConfig:
        """
        Return a TranscriptionConfig preset optimized for minimal memory usage on low-RAM systems.
        
        This configuration uses the "tiny" model, short chunk durations, minimal batch size, limited worker processes, and disables FP16 precision to reduce memory footprint.
        """
        return TranscriptionConfig(
            model_name="tiny",
            chunk_duration=20,
            overlap=2,
            batch_size=1,
            num_workers=2,
            max_memory_gb=2.0,
            clear_cache_every=5,
            use_fp16=False
        )
    
    @staticmethod
    def persian_optimized() -> TranscriptionConfig:
        """
        Return a TranscriptionConfig preset optimized for Persian/Farsi audio transcription.
        
        This configuration uses the "large-v3" model with Farsi language, moderate chunking, zero temperature, conditioning on previous text, Persian normalization enabled, diacritics retained, and adjusted quality thresholds for improved accuracy on Persian speech.
        """
        return TranscriptionConfig(
            model_name="large-v3",
            language="fa",
            chunk_duration=30,
            overlap=5,
            temperature=0.0,
            condition_on_previous_text=True,
            persian_normalization=True,
            remove_diacritics=False,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.5
        )
    
    @staticmethod
    def gpu_optimized() -> TranscriptionConfig:
        """
        Return a TranscriptionConfig preset optimized for GPU usage, with settings adjusted for large models and batch size dynamically set based on available GPU memory.
        """
        config = TranscriptionConfig(
            model_name="large",
            chunk_duration=45,
            overlap=3,
            batch_size=4,
            use_fp16=True,
            device="cuda"
        )
        
        # Adjust batch size based on GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb >= 8:
                config.batch_size = 8
            elif gpu_memory_gb >= 4:
                config.batch_size = 4
            else:
                config.batch_size = 2
        
        return config
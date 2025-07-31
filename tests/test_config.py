"""
Unit tests for configuration module.
"""

import pytest
import os
from unittest.mock import patch
from src.core.config import TranscriptionConfig, ConfigFactory


class TestTranscriptionConfig:
    """Test cases for TranscriptionConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TranscriptionConfig()
        
        assert config.model_name == "large-v3"
        assert config.language == "fa"
        assert config.chunk_duration_ms == 20000
        assert config.overlap_ms == 200
        assert config.device in ["cuda", "cpu"]
        assert config.repetition_threshold == 0.85
        assert config.max_word_repetition == 2
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid chunk duration
        with pytest.raises(ValueError, match="chunk_duration_ms must be positive"):
            TranscriptionConfig(chunk_duration_ms=0)
        
        # Test invalid overlap
        with pytest.raises(ValueError, match="overlap_ms cannot be negative"):
            TranscriptionConfig(overlap_ms=-1)
        
        # Test overlap >= chunk duration
        with pytest.raises(ValueError, match="overlap_ms must be less than chunk_duration_ms"):
            TranscriptionConfig(chunk_duration_ms=1000, overlap_ms=1000)
        
        # Test invalid repetition threshold
        with pytest.raises(ValueError, match="repetition_threshold must be between 0 and 1"):
            TranscriptionConfig(repetition_threshold=1.5)
        
        # Test invalid max word repetition
        with pytest.raises(ValueError, match="max_word_repetition must be at least 1"):
            TranscriptionConfig(max_word_repetition=0)
    
    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = TranscriptionConfig(
            model_name="base",
            language="en",
            chunk_duration_ms=15000,
            overlap_ms=300
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["model_name"] == "base"
        assert config_dict["language"] == "en"
        assert config_dict["chunk_duration_ms"] == 15000
        assert config_dict["overlap_ms"] == 300
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_name": "large",
            "language": "fa",
            "chunk_duration_ms": 25000,
            "overlap_ms": 500
        }
        
        config = TranscriptionConfig.from_dict(config_dict)
        
        assert config.model_name == "large"
        assert config.language == "fa"
        assert config.chunk_duration_ms == 25000
        assert config.overlap_ms == 500
    
    def test_config_clone(self):
        """Test configuration cloning."""
        original = TranscriptionConfig(
            model_name="base",
            language="en",
            chunk_duration_ms=15000
        )
        
        cloned = original.clone()
        
        assert cloned.model_name == original.model_name
        assert cloned.language == original.language
        assert cloned.chunk_duration_ms == original.chunk_duration_ms
        
        # Ensure they are separate objects
        cloned.model_name = "large"
        assert original.model_name == "base"
        assert cloned.model_name == "large"


class TestConfigFactory:
    """Test cases for ConfigFactory class."""
    
    def test_create_optimized_config(self):
        """Test optimized configuration creation."""
        config = ConfigFactory.create_optimized_config(
            model_size="large-v3",
            language="fa",
            enable_preview=True,
            output_dir="./test_output"
        )
        
        assert config.model_name == "large-v3"
        assert config.language == "fa"
        assert config.enable_sentence_preview is True
        assert config.output_directory == "./test_output"
        assert config.overlap_ms == 500
        assert config.repetition_threshold == 0.8
        assert config.max_word_repetition == 3
    
    def test_create_fast_config(self):
        """Test fast configuration creation."""
        config = ConfigFactory.create_fast_config()
        
        assert config.model_name == "base"
        assert config.chunk_duration_ms == 30000
        assert config.overlap_ms == 100
        assert config.batch_size == 4
        assert config.enable_sentence_preview is False
    
    def test_create_high_quality_config(self):
        """Test high quality configuration creation."""
        config = ConfigFactory.create_high_quality_config()
        
        assert config.model_name == "large-v3"
        assert config.chunk_duration_ms == 15000
        assert config.overlap_ms == 300
        assert config.repetition_threshold == 0.9
        assert config.max_word_repetition == 1
        assert config.min_chunk_confidence == 0.8
    
    @patch('torch.cuda.is_available')
    def test_device_optimization_cuda(self, mock_cuda_available):
        """Test device optimization with CUDA available."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.total_memory = 8e9  # 8GB GPU
            
            config = TranscriptionConfig(model_name="large-v3")
            assert config.device == "cuda"
            assert config.batch_size > 1
    
    @patch('torch.cuda.is_available')
    def test_device_optimization_cpu(self, mock_cuda_available):
        """Test device optimization without CUDA."""
        mock_cuda_available.return_value = False
        
        config = TranscriptionConfig()
        assert config.device == "cpu"
        assert config.batch_size == 1


if __name__ == "__main__":
    pytest.main([__file__]) 
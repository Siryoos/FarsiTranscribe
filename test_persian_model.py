#!/usr/bin/env python3
"""
Test script for nezamisafa/whisper-persian-v4 model integration.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import using the proper module structure
from src.core.config import TranscriptionConfig, ConfigFactory
from src.core.transcriber import OptimizedWhisperTranscriber


def test_model_loading():
    """Test if the Persian model loads correctly."""
    print("🧪 Testing nezamisafa/whisper-persian-v4 model loading...")
    
    # Create configuration with Persian model
    config = TranscriptionConfig(
        model_name="nezamisafa/whisper-persian-v4",
        language="fa",
        use_huggingface_model=True,
        device="cpu"  # Use CPU for testing
    )
    
    try:
        # Test model loading
        transcriber = OptimizedWhisperTranscriber(config)
        print("✅ Model loaded successfully!")
        
        # Test with a small dummy audio array
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        print("🧪 Testing transcription with dummy audio...")
        result = transcriber.transcribe_chunk(dummy_audio)
        print(f"✅ Transcription test completed. Result: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_config_factory():
    """Test configuration factory methods."""
    print("\n🧪 Testing configuration factory methods...")
    
    # Test Persian optimized config
    persian_config = ConfigFactory.create_persian_optimized_config()
    print(f"✅ Persian optimized config created: {persian_config.model_name}")
    
    # Test advanced Persian config
    advanced_config = ConfigFactory.create_advanced_persian_config()
    print(f"✅ Advanced Persian config created: {advanced_config.model_name}")
    
    # Test optimized config
    optimized_config = ConfigFactory.create_optimized_config()
    print(f"✅ Optimized config created: {optimized_config.model_name}")
    
    return True


def main():
    """Main test function."""
    print("🚀 Testing nezamisafa/whisper-persian-v4 Integration")
    print("=" * 50)
    
    # Test configuration
    if not test_config_factory():
        print("❌ Configuration tests failed")
        return False
    
    # Test model loading (only if transformers is available)
    try:
        import transformers
        if not test_model_loading():
            print("❌ Model loading tests failed")
            return False
    except ImportError:
        print("⚠️  transformers not available, skipping model loading test")
        print("   Install with: pip install transformers")
    
    print("\n✅ All tests passed! The Persian model is properly configured.")
    print("\n📝 Usage:")
    print("   python main.py your_audio_file.wav")
    print("   python main.py your_audio_file.wav --quality high")
    print("   python main.py your_audio_file.wav --model nezamisafa/whisper-persian-v4")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
Simple test script for FarsiTranscribe functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import UnifiedAudioTranscriber
from src.core.config import ConfigFactory


def test_basic_functionality():
    """Test basic functionality without requiring audio files."""
    print("🧪 Testing FarsiTranscribe Basic Functionality")
    print("=" * 50)
    
    try:
        # Test configuration creation
        print("1. Testing configuration creation...")
        config = ConfigFactory.create_optimized_config()
        print(f"   ✅ Config created: {config.model_name}")
        
        # Test transcriber initialization
        print("2. Testing transcriber initialization...")
        with UnifiedAudioTranscriber(config) as transcriber:
            print(f"   ✅ Transcriber initialized: {transcriber.__class__.__name__}")
            
            # Test configuration access
            print(f"   ✅ Model: {transcriber.config.model_name}")
            print(f"   ✅ Output dir: {transcriber.config.output_directory}")
            print(f"   ✅ Language: {transcriber.config.language}")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_persian_model_config():
    """Test Persian model configuration."""
    print("\n🇮🇷 Testing Persian Model Configuration")
    print("=" * 50)
    
    try:
        # Test Persian model config
        config = ConfigFactory.create_high_quality_config()
        config.model_name = "nezamisafa/whisper-persian-v4"
        config.use_huggingface_model = True
        
        print(f"✅ Persian model config: {config.model_name}")
        print(f"✅ HuggingFace flag: {config.use_huggingface_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Persian model test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting FarsiTranscribe Tests\n")
    
    success = True
    success &= test_basic_functionality()
    success &= test_persian_model_config()
    
    if success:
        print("\n🎉 All tests passed! FarsiTranscribe is ready to use.")
        print("\nNext steps:")
        print("1. Use: python main.py <audio_file>")
        print("2. Try: python main.py examples/audio/jalase bi va zirsakht.m4a")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

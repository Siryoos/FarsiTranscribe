#!/usr/bin/env python3
"""
Quick test for preprocessing functionality without full transcription.
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src'))

def test_preprocessing():
    """Test preprocessing components individually."""
    print("🧪 Testing Audio Preprocessing Components...")
    
    try:
        from src.utils.audio_preprocessor import AudioPreprocessor, get_preprocessing_capabilities
        from pydub import AudioSegment
        
        # Check capabilities
        caps = get_preprocessing_capabilities()
        print(f"📊 Capabilities: {caps}")
        
        # Create a test audio segment (1 second of silence)
        test_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor()
        
        # Test preprocessing
        processed_audio = preprocessor.preprocess_audio(test_audio)
        print(f"✅ Preprocessing: {len(test_audio)}ms → {len(processed_audio)}ms")
        
        # Test smart chunking
        chunks = preprocessor.create_smart_chunks(processed_audio, 5000, 200)
        print(f"✅ Smart chunking: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_worker_fix():
    """Test the worker function fix."""
    print("\n🧪 Testing Worker Function Fix...")
    
    try:
        from src.core.transcriber import UnifiedAudioTranscriber
        from src.core.config import ConfigFactory
        import numpy as np
        
        # Create minimal config
        config = ConfigFactory.create_cpu_optimized_config()
        config.enable_preprocessing = False  # Disable for this test
        
        # Create transcriber
        transcriber = UnifiedAudioTranscriber(config)
        
        # Test worker function with dummy data
        dummy_chunk = np.random.rand(16000).astype(np.float32)  # 1 second
        result = transcriber._transcribe_chunk_worker_optimized((0, dummy_chunk))
        
        print(f"✅ Worker function: chunk_index={result[0]}, text_length={len(result[1])}")
        return True
        
    except Exception as e:
        print(f"❌ Worker test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 FarsiTranscribe Preprocessing Fix Test")
    print("=" * 40)
    
    success_count = 0
    total_tests = 2
    
    if test_preprocessing():
        success_count += 1
    
    if test_worker_fix():
        success_count += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All fixes working! Ready for transcription.")
    else:
        print("⚠️  Some issues remain. Check the error messages above.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

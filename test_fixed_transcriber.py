#!/usr/bin/env python3
"""
Test script to verify the fixed transcriber works correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import TranscriptionConfig
from src.core.transcriber import OptimizedWhisperTranscriber
import numpy as np


def test_fixed_transcriber():
    """Test the fixed transcriber with Persian model."""
    print("🧪 Testing fixed transcriber with Persian model...")
    
    # Create configuration
    config = TranscriptionConfig(
        model_name="nezamisafa/whisper-persian-v4",
        language="fa",
        use_huggingface_model=True,
        device="cpu",
        temperature=0.0  # Deterministic
    )
    
    try:
        # Initialize transcriber
        transcriber = OptimizedWhisperTranscriber(config)
        print("✅ Transcriber initialized successfully!")
        
        # Test with a small audio chunk (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)
        
        print("🧪 Testing transcription...")
        result = transcriber.transcribe_chunk(test_audio)
        print(f"✅ Transcription completed successfully!")
        print(f"📝 Result: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Testing Fixed Transcriber")
    print("=" * 40)
    
    success = test_fixed_transcriber()
    
    if success:
        print("\n✅ All tests passed! The transcriber is working correctly.")
        print("\n📝 You can now use:")
        print("   python3 main.py your_audio_file.wav --model nezamisafa/whisper-persian-v4")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
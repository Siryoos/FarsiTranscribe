#!/usr/bin/env python3
"""
Test script to verify transcription fallback mechanism.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.core.transcriber import UnifiedAudioTranscriber
from src.core.config import TranscriptionConfig


def test_transcription_fallback():
    """Test the transcription fallback mechanism."""
    print("🧪 Testing Transcription Fallback Mechanism")
    print("=" * 50)

    # Create config
    config = TranscriptionConfig()
    print(f"Initial device: {config.device}")
    print(f"Model: {config.model_name}")

    # Create transcriber
    transcriber = UnifiedAudioTranscriber(config)
    print(
        f"Transcriber device: {transcriber.whisper_transcriber.device_manager.get_device()}"
    )

    # Test with a small audio file if available
    test_audio = "examples/audio/jalase bi va zirsakht.m4a"

    if os.path.exists(test_audio):
        print(f"\n🎵 Testing with audio file: {test_audio}")
        print("Note: This will only test the setup, not full transcription")

        try:
            # Just test the device manager
            device_manager = transcriber.whisper_transcriber.device_manager
            print(f"Current device: {device_manager.get_device()}")

            # Test CPU fallback
            print("\n🔄 Testing CPU fallback...")
            device_manager.force_cpu_fallback()
            print(f"Device after fallback: {device_manager.get_device()}")
            print(f"Config device: {config.device}")

            print("\n✅ Fallback test completed!")

        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print(f"⚠️ Test audio file not found: {test_audio}")
        print("Testing device manager only...")

        device_manager = transcriber.whisper_transcriber.device_manager
        print(f"Current device: {device_manager.get_device()}")
        print(f"CUDA available: {device_manager.is_cuda_available()}")


if __name__ == "__main__":
    test_transcription_fallback()

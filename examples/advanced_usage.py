#!/usr/bin/env python3
"""
Advanced usage examples for FarsiTranscribe.

This script demonstrates various advanced features of the FarsiTranscribe library
for Persian/Farsi audio transcription.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import UnifiedAudioTranscriber
from src.core.config import ConfigFactory


def example_persian_optimized_transcription():
    """Persian-optimized transcription with high quality."""
    print("🇮🇷 Example 1: Persian-Optimized Transcription")
    print("=" * 50)
    
    try:
        # Use Persian-optimized preset
        config = ConfigFactory.create_persian_optimized_config()
        
        # Customize some settings
        config.output_directory = Path("output/persian")
        config.enable_sentence_preview = True
        
        print(f"✅ Model: {config.model_name}")
        print(f"✅ Language: {config.language}")
        print(f"✅ Output directory: {config.output_directory}")
        print(f"✅ Preview enabled: {config.enable_sentence_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Persian model test failed: {e}")
        return False


def example_memory_efficient_transcription():
    """Memory-efficient transcription for large files."""
    print("\n💾 Example 2: Memory-Efficient Transcription")
    print("=" * 50)
    
    try:
        # Use memory-efficient preset
        config = ConfigFactory.create_memory_optimized_config()
        config.clear_cache_every = 5  # Clear cache every 5 chunks
        
        print(f"✅ Model: {config.model_name}")
        print(f"✅ Memory efficient mode: {config.memory_efficient_mode}")
        print(f"✅ Memory threshold: {config.memory_threshold_mb}MB")
        print(f"✅ Cleanup interval: {config.cleanup_interval_seconds}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory efficient test failed: {e}")
        return False


def example_high_quality_transcription():
    """High quality transcription with advanced features."""
    print("\n🎯 Example 3: High Quality Transcription")
    print("=" * 50)
    
    try:
        # Use high quality preset
        config = ConfigFactory.create_high_quality_config()
        
        print(f"✅ Model: {config.model_name}")
        print(f"✅ Enhanced preprocessing: {config.enable_enhanced_preprocessing}")
        print(f"✅ Text postprocessing: {config.enable_text_postprocessing}")
        print(f"✅ Persian optimization: {config.enable_persian_optimization}")
        
        return True
        
    except Exception as e:
        print(f"❌ High quality test failed: {e}")
        return False


def example_95_percent_quality():
    """Maximum quality transcription with all features enabled."""
    print("\n🏆 Example 4: 95% Quality Transcription")
    print("=" * 50)
    
    try:
        # Use 95% quality preset
        config = ConfigFactory.create_95_percent_quality_config()
        
        print(f"✅ Model: {config.model_name}")
        print(f"✅ Model ensemble: {config.enable_model_ensemble}")
        print(f"✅ Speaker diarization: {config.enable_speaker_diarization}")
        print(f"✅ Quality assessment: {config.enable_quality_assessment}")
        print(f"✅ Auto tuning: {config.enable_auto_tuning}")
        print(f"✅ Target quality: {config.target_quality_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ 95% quality test failed: {e}")
        return False


def example_custom_configuration():
    """Custom configuration example."""
    print("\n⚙️  Example 5: Custom Configuration")
    print("=" * 50)
    
    try:
        # Create custom configuration
        config = ConfigFactory.create_optimized_config()
        
        # Customize settings
        config.model_name = "nezamisafa/whisper-persian-v4"
        config.language = "fa"
        config.chunk_duration_ms = 30000  # 30 seconds
        config.overlap_ms = 500  # 500ms overlap
        config.enable_noise_reduction = True
        config.enable_voice_activity_detection = True
        
        print(f"✅ Custom model: {config.model_name}")
        print(f"✅ Custom chunk duration: {config.chunk_duration_ms}ms")
        print(f"✅ Custom overlap: {config.overlap_ms}ms")
        print(f"✅ Noise reduction: {config.enable_noise_reduction}")
        print(f"✅ Voice activity detection: {config.enable_voice_activity_detection}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom configuration test failed: {e}")
        return False


def example_transcription_with_sample_audio():
    """Example of how to transcribe the sample audio file."""
    print("\n🎵 Example 6: Transcription with Sample Audio")
    print("=" * 50)
    
    sample_audio = Path("examples/audio/jalase bi va zirsakht.m4a")
    
    if not sample_audio.exists():
        print(f"⚠️  Sample audio file not found: {sample_audio}")
        print("   This example shows the command to use when you have audio files.")
        return True
    
    print(f"✅ Sample audio found: {sample_audio}")
    print(f"📁 File size: {sample_audio.stat().st_size / (1024*1024):.1f} MB")
    
    print("\n🚀 To transcribe this file, use:")
    print(f"   python main.py {sample_audio}")
    print("\n🔧 With different quality presets:")
    print(f"   python main.py {sample_audio} --quality fast")
    print(f"   python main.py {sample_audio} --quality balanced")
    print(f"   python main.py {sample_audio} --quality high")
    print(f"   python main.py {sample_audio} --quality 95-percent")
    
    return True


def main():
    """Run all examples."""
    print("🚀 FarsiTranscribe Advanced Usage Examples")
    print("=" * 60)
    
    examples = [
        example_persian_optimized_transcription,
        example_memory_efficient_transcription,
        example_high_quality_transcription,
        example_95_percent_quality,
        example_custom_configuration,
        example_transcription_with_sample_audio,
    ]
    
    success_count = 0
    total_count = len(examples)
    
    for example in examples:
        try:
            if example():
                success_count += 1
        except Exception as e:
            print(f"❌ Example failed with error: {e}")
    
    print(f"\n📊 Results: {success_count}/{total_count} examples passed")
    
    if success_count == total_count:
        print("\n🎉 All examples completed successfully!")
        print("\n📚 Next steps:")
        print("1. Try transcribing audio: python main.py <audio_file>")
        print("2. Explore different quality presets")
        print("3. Check the output directory for results")
        print("4. Read the README.md for more details")
    else:
        print(f"\n⚠️  {total_count - success_count} examples had issues.")
        print("   Check the errors above for details.")


if __name__ == "__main__":
    main()

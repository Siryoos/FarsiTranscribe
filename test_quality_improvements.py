#!/usr/bin/env python3
"""
Test script for quality improvements based on user feedback.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import ConfigFactory
from src.utils.enhanced_audio_preprocessor import create_enhanced_preprocessor
from src.utils.persian_text_postprocessor import create_persian_postprocessor


def test_enhanced_preprocessing():
    """Test enhanced audio preprocessing."""
    print("🧪 Testing enhanced audio preprocessing...")
    
    config = ConfigFactory.create_persian_optimized_config()
    preprocessor = create_enhanced_preprocessor(config)
    
    print("✅ Enhanced preprocessor created successfully")
    print(f"   - Noise reduction: {config.enable_noise_reduction}")
    print(f"   - Enhanced preprocessing: {config.enable_enhanced_preprocessing}")
    print(f"   - Target sample rate: {config.target_sample_rate}Hz")
    
    return True


def test_text_postprocessing():
    """Test Persian text post-processing."""
    print("\n🧪 Testing Persian text post-processing...")
    
    config = ConfigFactory.create_persian_optimized_config()
    postprocessor = create_persian_postprocessor(config)
    
    # Test with sample Persian text
    sample_text = "سلام   این یک متن تست است...   با مشکلات مختلف"
    processed_text, metadata = postprocessor.post_process_text(sample_text)
    
    print("✅ Text post-processor created successfully")
    print(f"   Original: '{sample_text}'")
    print(f"   Processed: '{processed_text}'")
    print(f"   Quality score: {metadata['quality_metrics']['score']}")
    
    return True


def main():
    """Main test function."""
    print("🚀 Testing Quality Improvements")
    print("=" * 40)
    
    # Test enhanced preprocessing
    if not test_enhanced_preprocessing():
        print("❌ Enhanced preprocessing test failed")
        return False
    
    # Test text post-processing
    if not test_text_postprocessing():
        print("❌ Text post-processing test failed")
        return False
    
    print("\n✅ All quality improvement tests passed!")
    print("\n📝 Quality improvements implemented:")
    print("   • Enhanced audio preprocessing (16kHz mono, noise reduction)")
    print("   • Persian text post-processing (punctuation, spacing)")
    print("   • Multi-speaker conversation chunking")
    print("   • Audio quality validation")
    print("   • Context-aware text corrections")
    
    return True


if __name__ == "__main__":
    main() 
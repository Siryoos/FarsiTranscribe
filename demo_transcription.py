#!/usr/bin/env python3
"""
Demo script for FarsiTranscribe - demonstrates actual transcription.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import UnifiedAudioTranscriber
from src.core.config import ConfigFactory


def demo_fast_transcription():
    """Demo fast transcription with the sample audio."""
    print("🚀 FarsiTranscribe Demo - Fast Transcription")
    print("=" * 50)
    
    sample_audio = Path("examples/audio/jalase bi va zirsakht.m4a")
    
    if not sample_audio.exists():
        print(f"❌ Sample audio file not found: {sample_audio}")
        return False
    
    print(f"📁 Audio file: {sample_audio}")
    print(f"📊 File size: {sample_audio.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        # Create fast configuration
        config = ConfigFactory.create_fast_config()
        config.output_directory = "output/demo"
        config.enable_sentence_preview = True
        
        print(f"\n⚙️  Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Quality: Fast")
        print(f"   Output: {config.output_directory}")
        print(f"   Preview: {config.enable_sentence_preview}")
        
        print(f"\n🔄 Starting transcription...")
        print("   (This may take a few minutes for the first run)")
        
        # Create output directory
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        
        # Run transcription
        with UnifiedAudioTranscriber(config) as transcriber:
            result = transcriber.transcribe_file(str(sample_audio))
            
            if result:
                print(f"\n✅ Transcription completed!")
                print(f"📝 Text length: {len(result)} characters")
                
                # Save result
                output_file = Path(config.output_directory) / "demo_transcription.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                print(f"💾 Saved to: {output_file}")
                
                # Show preview
                preview_length = min(200, len(result))
                print(f"\n📖 Preview (first {preview_length} characters):")
                print("-" * 50)
                print(result[:preview_length])
                if len(result) > preview_length:
                    print("...")
                
                return True
            else:
                print("❌ No transcription result generated")
                return False
                
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the demo."""
    print("🎙️  Welcome to FarsiTranscribe Demo!")
    print("=" * 50)
    
    print("This demo will transcribe the sample Persian audio file")
    print("using the fast quality preset to demonstrate functionality.")
    print()
    
    # Check if we have the required dependencies
    try:
        import torch
        import transformers
        print("✅ Dependencies: PyTorch and Transformers available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        return
    
    # Run demo
    success = demo_fast_transcription()
    
    if success:
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n📚 Next steps:")
        print(f"1. Try different quality presets:")
        print(f"   python main.py examples/audio/jalase bi va zirsakht.m4a --quality balanced")
        print(f"   python main.py examples/audio/jalase bi va zirsakht.m4a --quality high")
        print(f"2. Check the output directory for results")
        print(f"3. Read the README.md for more features")
    else:
        print(f"\n❌ Demo failed. Check the errors above.")


if __name__ == "__main__":
    main()

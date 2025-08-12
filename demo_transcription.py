#!/usr/bin/env python3
"""Demo script for FarsiTranscribe.

Uses the public CLI to avoid duplicating logic.
"""

import subprocess
from pathlib import Path


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
        print(f"\n🔄 Starting transcription via CLI...")
        output_dir = Path("output/demo")
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "-m",
            "farsi_transcribe",
            str(sample_audio),
            "--preset",
            "fast",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return False


def main():
    """Run the demo."""
    print("🎙️  Welcome to FarsiTranscribe Demo!")
    print("=" * 50)
    
    print("This demo will transcribe the sample Persian audio file")
    print("using the fast quality preset to demonstrate functionality.")
    print()
    
    # Check if we have the required dependencies
    # Basic dependency hint (do not import heavy libs here)
    if not (Path("requirements.txt").exists()):
        print("⚠️  requirements.txt not found; ensure dependencies are installed")
    
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

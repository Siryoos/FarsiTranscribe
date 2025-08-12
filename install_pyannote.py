#!/usr/bin/env python3
"""
Helper script to install pyannote.audio and dependencies.
Run this to set up the advanced speaker diarization system.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🎙️  Installing pyannote.audio for advanced speaker diarization...")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider creating one first: python -m venv .venv")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Install PyTorch first (CPU version for compatibility)
    print("\n📦 Installing PyTorch (CPU version)...")
    if not run_command("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu", "PyTorch installation"):
        print("❌ PyTorch installation failed. Trying alternative...")
        if not run_command("pip install torch torchaudio", "PyTorch installation (alternative)"):
            print("❌ PyTorch installation failed completely. Please install manually.")
            return
    
    # Install pyannote.audio
    print("\n📦 Installing pyannote.audio...")
    if not run_command("pip install pyannote.audio==3.1.1", "pyannote.audio installation"):
        print("❌ pyannote.audio installation failed.")
        return
    
    # Install additional dependencies
    print("\n📦 Installing additional dependencies...")
    dependencies = [
        "soundfile>=0.12.0",
        "rich>=13.0.0",
        "librosa>=0.10.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("\n" + "=" * 60)
    print("✅ Installation completed!")
    print("\n📋 Next steps:")
    print("1. Run your transcription command:")
    print("   python main.py your_audio_file.wav --quality 95-percent")
    print("\n2. For better diarization, specify speaker count:")
    print("   python main.py your_audio_file.wav --quality 95-percent --num-speakers 2")
    print("\n3. If you know speaker range:")
    print("   python main.py your_audio_file.wav --quality 95-percent --min-speakers 2 --max-speakers 4")
    print("\n4. To disable diarization and get full transcript:")
    print("   python main.py your_audio_file.wav --quality 95-percent --no-diarization")
    
    print("\n🎯 The system will automatically:")
    print("   - Use pyannote.audio if available (much better accuracy)")
    print("   - Fall back to basic diarizer if pyannote fails")
    print("   - Fall back to standard transcription if diarization fails")
    
    print("\n💡 First run will download the diarization model (~1GB)")

if __name__ == "__main__":
    main()

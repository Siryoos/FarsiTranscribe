#!/usr/bin/env python3
"""
Project cleanup and migration script.
Removes redundant files and fixes common issues.
"""

import os
import sys
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up redundant files and migrate to unified components."""
    project_root = Path(__file__).parent
    
    # Files to remove (redundant/problematic)
    files_to_remove = [
        "src/core/optimized_transcriber.py",  # Merged into unified
        "src/utils/optimized_chunk_calculator.py",  # Merged into unified
        "main_optimized.py",  # Replaced by main_unified.py  
        "benchmark.py",  # Optional, can be removed
    ]
    
    # Backup original files before removal
    backup_dir = project_root / "backup_original"
    backup_dir.mkdir(exist_ok=True)
    
    files_to_backup = [
        "src/core/transcriber.py",
        "src/utils/chunk_calculator.py",
        "main.py"
    ]
    
    print("🔄 Creating backups...")
    for file_path in files_to_backup:
        full_path = project_root / file_path
        if full_path.exists():
            backup_path = backup_dir / file_path.replace("/", "_")
            shutil.copy2(full_path, backup_path)
            print(f"  ✅ Backed up: {file_path}")
    
    print("\n🔄 Removing redundant files...")
    for file_path in files_to_remove:
        full_path = project_root / file_path
        if full_path.exists():
            os.remove(full_path)
            print(f"  ✅ Removed: {file_path}")
    
    # Rename unified files to replace originals
    renames = [
        ("src/core/transcriber_unified.py", "src/core/transcriber.py"),
        ("src/utils/chunk_calculator_unified.py", "src/utils/chunk_calculator.py"),
        ("main_unified.py", "main.py")
    ]
    
    print("\n🔄 Installing unified components...")
    for old_name, new_name in renames:
        old_path = project_root / old_name
        new_path = project_root / new_name
        
        if old_path.exists():
            if new_path.exists():
                os.remove(new_path)  # Remove original
            shutil.move(old_path, new_path)
            print(f"  ✅ Installed: {new_name}")
    
    print("\n✅ Cleanup completed!")
    print(f"📁 Original files backed up to: {backup_dir}")
    
    return True


def check_dependencies():
    """Check for required dependencies."""
    print("\n🔄 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("whisper", "OpenAI Whisper"), 
        ("pydub", "Pydub"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("numpy", "NumPy")
    ]
    
    missing = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing.append(name)
    
    # Check for ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("  ✅ ffmpeg")
        else:
            print("  ⚠️  ffmpeg - Available but may have issues")
    except Exception:
        print("  ⚠️  ffmpeg - Not found (will use pydub fallback)")
    
    if missing:
        print(f"\n❌ Install missing packages: pip install {' '.join(missing.lower().split())}")
        return False
    
    print("\n✅ All dependencies satisfied!")
    return True


def main():
    """Main cleanup function."""
    print("🧹 FarsiTranscribe Project Cleanup")
    print("=" * 40)
    
    try:
        cleanup_success = cleanup_project()
        deps_success = check_dependencies()
        
        if cleanup_success and deps_success:
            print("\n🎉 Project successfully cleaned and optimized!")
            print("\nUsage:")
            print("  python main.py your_audio.m4a --quality memory-optimized")
            
        else:
            print("\n⚠️  Cleanup completed with warnings. Check messages above.")
            
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

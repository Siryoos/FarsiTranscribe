#!/usr/bin/env python3
"""
Cleanup script for FarsiTranscribe project.
Removes temporary files and organizes the project structure.
"""

import os
import shutil
import glob

def cleanup_temp_files():
    """Remove temporary files and directories."""
    print("🧹 Cleaning up temporary files...")
    
    # Files to remove
    temp_files = [
        "*.pyc",
        "*.pyo",
        "__pycache__",
        "*.log",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    # Directories to remove
    temp_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        "htmlcov"
    ]
    
    removed_count = 0
    
    # Remove temporary files
    for pattern in temp_files:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Removed file: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    # Remove temporary directories
    for dir_name in temp_dirs:
        for dir_path in glob.glob(f"**/{dir_name}", recursive=True):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"🗑️  Removed directory: {dir_path}")
                    removed_count += 1
            except Exception as e:
                print(f"⚠️  Could not remove {dir_path}: {e}")
    
    print(f"✅ Cleanup completed. Removed {removed_count} items.")

def create_output_directories():
    """Create necessary output directories."""
    print("📁 Creating output directories...")
    
    directories = [
        "output",
        "output/logs",
        "output/transcriptions",
        "output/temp"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"⚠️  Could not create {directory}: {e}")

def check_project_structure():
    """Check and report project structure."""
    print("📋 Checking project structure...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "main.py",
        "main.py",
        "src/core/config.py",
        "src/core/transcriber.py",
        "src/utils/terminal_display.py",
        "src/utils/sentence_extractor.py",
        "src/utils/repetition_detector.py",
        "src/utils/file_manager.py"
    ]
    
    required_dirs = [
        "src",
        "src/core",
        "src/utils",
        "examples",
        "tests",
        "output"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    else:
        print("✅ All required files present")
    
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
    else:
        print("✅ All required directories present")

def main():
    """Run cleanup operations."""
    print("🎙️ FarsiTranscribe Project Cleanup")
    print("=" * 50)
    
    # Check project structure
    check_project_structure()
    print()
    
    # Create output directories
    create_output_directories()
    print()
    
    # Clean up temporary files
    cleanup_temp_files()
    print()
    
    print("🎉 Project cleanup completed!")
    print("\nYour project is now clean and organized.")
    print("You can start using it with:")
    print("  ./activate_env.sh")
    print("  python test_system.py")

if __name__ == "__main__":
    main() 
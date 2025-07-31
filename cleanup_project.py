#!/usr/bin/env python3
"""
Cleanup script for FarsiTranscribe project.

This script removes old files and reorganizes the project structure
according to the new modular design.
"""

import shutil
from pathlib import Path
import argparse


# Files and directories to remove
TO_REMOVE = {
    # Old main files
    "main.py",
    "main_multicore.py", 
    "main_quality.py",
    "efficient.py",
    "optimized_transcriber.py",
    "batch_transcribe.py",
    
    # Old scripts
    "cleanup.py",
    "cleanup_final.py",
    "run_optimized.sh",
    "run_transcription.sh",
    
    # Old documentation
    "CHUNK_ANALYSIS_GUIDE.md",
    "CLEANUP_SUMMARY.md",
    "COMPREHENSIVE_FIX_SUMMARY.md",
    "MEMORY_OPTIMIZATION.md",
    
    # Installation scripts
    "install_advanced_preprocessing.sh",
    "install_preprocessing.sh",
    "install_rtl_support.sh",
    "activate_env.sh",
    
    # Old configs
    "config.ini",
    "Makefile",
    "pytest.ini",
    
    # Directories
    "src/",
    "backup_original/",
    "scripts/",
    
    # Log files
    "*.log",
    "transcription.log",
}

# Files to keep
TO_KEEP = {
    "farsi_transcribe/",
    "examples/",
    "tests/",
    "output/",
    "requirements.txt",
    "setup.py",
    "README.md",
    "PROJECT_STRUCTURE.md",
    "LICENSE",
    ".gitignore",
    ".git/",
    "venv/",
}


def confirm_action(message: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"{message} [y/N]: ").lower().strip()
    return response == 'y'


def cleanup_project(dry_run: bool = True, force: bool = False):
    """Clean up the project structure."""
    root = Path.cwd()
    
    print("🧹 FarsiTranscribe Project Cleanup")
    print("=" * 50)
    
    if dry_run:
        print("DRY RUN MODE - No files will be deleted")
        print()
    
    # Find files to remove
    files_to_remove = []
    dirs_to_remove = []
    
    for pattern in TO_REMOVE:
        if pattern.endswith('/'):
            # Directory
            path = root / pattern.rstrip('/')
            if path.exists() and path.is_dir():
                dirs_to_remove.append(path)
        elif '*' in pattern:
            # Glob pattern
            for path in root.glob(pattern):
                if path.is_file():
                    files_to_remove.append(path)
        else:
            # Regular file
            path = root / pattern
            if path.exists() and path.is_file():
                files_to_remove.append(path)
    
    # Display what will be removed
    if files_to_remove or dirs_to_remove:
        print("📄 Files to remove:")
        for f in sorted(files_to_remove):
            print(f"  - {f.relative_to(root)}")
        
        print("\n📁 Directories to remove:")
        for d in sorted(dirs_to_remove):
            print(f"  - {d.relative_to(root)}/")
        
        total_size = sum(f.stat().st_size for f in files_to_remove) / 1024 / 1024
        print(f"\n💾 Total size to free: {total_size:.2f} MB")
    else:
        print("✅ No files to remove - project is already clean!")
        return
    
    # Confirm action
    if not dry_run and not force:
        if not confirm_action("\n⚠️  This will permanently delete files. Continue?"):
            print("❌ Cleanup cancelled.")
            return
    
    # Perform cleanup
    if not dry_run:
        print("\n🔄 Removing files...")
        
        # Remove files
        for f in files_to_remove:
            try:
                f.unlink()
                print(f"  ✓ Removed {f.name}")
            except Exception as e:
                print(f"  ✗ Error removing {f.name}: {e}")
        
        # Remove directories
        for d in dirs_to_remove:
            try:
                shutil.rmtree(d)
                print(f"  ✓ Removed {d.name}/")
            except Exception as e:
                print(f"  ✗ Error removing {d.name}/: {e}")
        
        print("\n✅ Cleanup completed!")
    
    # Show remaining structure
    print("\n📂 Clean project structure:")
    for item in sorted(root.iterdir()):
        if item.name.startswith('.') and item.name != '.gitignore':
            continue
        if item.name in {'__pycache__', 'venv', '*.egg-info'}:
            continue
        
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")


def create_gitignore():
    """Create or update .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
output/
*.log
transcription.log
farsi_transcribe.log
*.txt
*.json
!requirements*.txt
!examples/**/*.txt

# Audio files (keep only examples)
*.mp3
*.wav
*.m4a
*.flac
*.ogg
*.opus
!examples/audio/*

# Temporary files
*.tmp
*.temp
.cache/
"""
    
    gitignore_path = Path(".gitignore")
    gitignore_path.write_text(gitignore_content)
    print(f"✅ Created/Updated .gitignore")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up FarsiTranscribe project structure"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the cleanup (without this flag, runs in dry-run mode)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--update-gitignore",
        action="store_true",
        help="Create/update .gitignore file"
    )
    
    args = parser.parse_args()
    
    # Update gitignore if requested
    if args.update_gitignore:
        create_gitignore()
    
    # Run cleanup
    dry_run = not args.execute
    cleanup_project(dry_run=dry_run, force=args.force)
    
    if dry_run:
        print("\n💡 To actually perform cleanup, run:")
        print("   python cleanup_project.py --execute")


if __name__ == "__main__":
    main()
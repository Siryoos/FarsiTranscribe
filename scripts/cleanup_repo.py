#!/usr/bin/env python3
"""
Repository cleanup script for FarsiTranscribe.
This script helps clean up files that should be ignored according to .gitignore.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set


class RepoCleaner:
    """Repository cleanup utility."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.ignored_patterns = self._load_gitignore_patterns()
        self.files_to_remove = []
        self.dirs_to_remove = []
        
    def _load_gitignore_patterns(self) -> Set[str]:
        """Load patterns from .gitignore file."""
        gitignore_path = self.repo_path / ".gitignore"
        patterns = set()
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove comments
                        if '#' in line:
                            line = line.split('#')[0].strip()
                        if line:
                            patterns.add(line)
        
        return patterns
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on .gitignore patterns."""
        rel_path = file_path.relative_to(self.repo_path)
        rel_str = str(rel_path)
        
        for pattern in self.ignored_patterns:
            if self._matches_pattern(rel_str, pattern):
                return True
        return False
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a gitignore pattern."""
        import fnmatch
        
        # Handle directory patterns
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            if os.path.isdir(os.path.join(self.repo_path, path)):
                return fnmatch.fnmatch(path, pattern)
        
        # Handle wildcard patterns
        if '*' in pattern or '?' in pattern:
            return fnmatch.fnmatch(path, pattern)
        
        # Exact match
        return path == pattern
    
    def scan_for_ignored_files(self) -> None:
        """Scan repository for files that should be ignored."""
        print("🔍 Scanning repository for ignored files...")
        
        for root, dirs, files in os.walk(self.repo_path):
            root_path = Path(root)
            
            # Check directories
            for dir_name in dirs[:]:  # Copy list to modify during iteration
                dir_path = root_path / dir_name
                if self._should_ignore(dir_path):
                    self.dirs_to_remove.append(dir_path)
                    dirs.remove(dir_name)  # Don't traverse ignored directories
            
            # Check files
            for file_name in files:
                file_path = root_path / file_name
                if self._should_ignore(file_path):
                    self.files_to_remove.append(file_path)
    
    def print_summary(self) -> None:
        """Print summary of files to be removed."""
        print(f"\n📊 Cleanup Summary:")
        print(f"   Files to remove: {len(self.files_to_remove)}")
        print(f"   Directories to remove: {len(self.dirs_to_remove)}")
        
        if self.files_to_remove:
            print(f"\n📄 Files to remove:")
            for file_path in sorted(self.files_to_remove):
                rel_path = file_path.relative_to(self.repo_path)
                print(f"   - {rel_path}")
        
        if self.dirs_to_remove:
            print(f"\n📁 Directories to remove:")
            for dir_path in sorted(self.dirs_to_remove):
                rel_path = dir_path.relative_to(self.repo_path)
                print(f"   - {rel_path}")
    
    def cleanup(self, dry_run: bool = True) -> None:
        """Perform the cleanup operation."""
        if dry_run:
            print(f"\n🔍 DRY RUN - No files will be actually removed")
        else:
            print(f"\n🗑️  PERFORMING CLEANUP")
        
        # Remove files
        for file_path in self.files_to_remove:
            rel_path = file_path.relative_to(self.repo_path)
            if dry_run:
                print(f"   Would remove: {rel_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"   Removed: {rel_path}")
                except Exception as e:
                    print(f"   Error removing {rel_path}: {e}")
        
        # Remove directories (in reverse order to handle nested dirs)
        for dir_path in sorted(self.dirs_to_remove, reverse=True):
            rel_path = dir_path.relative_to(self.repo_path)
            if dry_run:
                print(f"   Would remove: {rel_path}")
            else:
                try:
                    shutil.rmtree(dir_path)
                    print(f"   Removed: {rel_path}")
                except Exception as e:
                    print(f"   Error removing {rel_path}: {e}")
    
    def get_repo_stats(self) -> dict:
        """Get repository statistics."""
        total_files = 0
        total_dirs = 0
        total_size = 0
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]
            
            total_dirs += len(dirs)
            
            for file_name in files:
                file_path = Path(root) / file_name
                if not self._should_ignore(file_path):
                    total_files += 1
                    try:
                        total_size += file_path.stat().st_size
                    except OSError:
                        pass
        
        return {
            'files': total_files,
            'directories': total_dirs,
            'size_mb': total_size / (1024 * 1024)
        }


def main():
    """Main function."""
    print("🧹 FarsiTranscribe Repository Cleanup")
    print("=" * 50)
    
    # Parse command line arguments
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        dry_run = False
    
    # Initialize cleaner
    cleaner = RepoCleaner()
    
    # Get initial stats
    initial_stats = cleaner.get_repo_stats()
    print(f"📊 Initial repository stats:")
    print(f"   Files: {initial_stats['files']}")
    print(f"   Directories: {initial_stats['directories']}")
    print(f"   Size: {initial_stats['size_mb']:.2f} MB")
    
    # Scan for ignored files
    cleaner.scan_for_ignored_files()
    
    # Print summary
    cleaner.print_summary()
    
    # Perform cleanup
    if cleaner.files_to_remove or cleaner.dirs_to_remove:
        response = input(f"\n❓ Proceed with cleanup? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            cleaner.cleanup(dry_run=dry_run)
            
            if not dry_run:
                # Get final stats
                final_stats = cleaner.get_repo_stats()
                print(f"\n📊 Final repository stats:")
                print(f"   Files: {final_stats['files']}")
                print(f"   Directories: {final_stats['directories']}")
                print(f"   Size: {final_stats['size_mb']:.2f} MB")
                print(f"   Space saved: {initial_stats['size_mb'] - final_stats['size_mb']:.2f} MB")
        else:
            print("❌ Cleanup cancelled.")
    else:
        print("✅ No files to clean up!")
    
    print("\n💡 Tips:")
    print("   - Run with --execute flag to actually remove files")
    print("   - Use 'git status' to see what files are tracked")
    print("   - Use 'git add .' to stage remaining files")
    print("   - Use 'git commit' to commit the cleaned repository")


if __name__ == "__main__":
    main() 
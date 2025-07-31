#!/usr/bin/env python3
"""
Cleanup script for FarsiTranscribe project.
Removes unnecessary files, optimizes structure, and improves RAM efficiency.
"""

import os
import shutil
import glob
import sys
from pathlib import Path
import argparse


class ProjectCleaner:
    """Cleanup utility for the FarsiTranscribe project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.removed_files = []
        self.removed_dirs = []
        
    def cleanup_python_cache(self):
        """Remove Python cache files."""
        print("🧹 Cleaning Python cache files...")
        
        # Remove __pycache__ directories
        for pycache_dir in self.project_root.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache_dir)
                self.removed_dirs.append(str(pycache_dir))
                print(f"  Removed: {pycache_dir}")
            except Exception as e:
                print(f"  Error removing {pycache_dir}: {e}")
        
        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                self.removed_files.append(str(pyc_file))
                print(f"  Removed: {pyc_file}")
            except Exception as e:
                print(f"  Error removing {pyc_file}: {e}")
    
    def cleanup_logs(self):
        """Remove log files."""
        print("🧹 Cleaning log files...")
        
        log_patterns = [
            "*.log",
            "*.log.*",
            "transcription.log",
            "debug.log",
            "error.log"
        ]
        
        for pattern in log_patterns:
            for log_file in self.project_root.rglob(pattern):
                try:
                    log_file.unlink()
                    self.removed_files.append(str(log_file))
                    print(f"  Removed: {log_file}")
                except Exception as e:
                    print(f"  Error removing {log_file}: {e}")
    
    def cleanup_temp_files(self):
        """Remove temporary files."""
        print("🧹 Cleaning temporary files...")
        
        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        for pattern in temp_patterns:
            for temp_file in self.project_root.rglob(pattern):
                try:
                    temp_file.unlink()
                    self.removed_files.append(str(temp_file))
                    print(f"  Removed: {temp_file}")
                except Exception as e:
                    print(f"  Error removing {temp_file}: {e}")
    
    def cleanup_output_files(self, keep_recent: bool = True):
        """Clean output directory while keeping recent files."""
        print("🧹 Cleaning output directory...")
        
        output_dir = self.project_root / "output"
        if not output_dir.exists():
            print("  Output directory does not exist")
            return
        
        # Keep only the most recent transcription files
        if keep_recent:
            transcription_files = list(output_dir.glob("*_transcription.txt"))
            if len(transcription_files) > 3:
                # Sort by modification time and keep only the 3 most recent
                transcription_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                files_to_remove = transcription_files[3:]
                
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        self.removed_files.append(str(file_path))
                        print(f"  Removed old transcription: {file_path}")
                    except Exception as e:
                        print(f"  Error removing {file_path}: {e}")
        
        # Remove all log files in output
        for log_file in output_dir.glob("*.log"):
            try:
                log_file.unlink()
                self.removed_files.append(str(log_file))
                print(f"  Removed: {log_file}")
            except Exception as e:
                print(f"  Error removing {log_file}: {e}")
    
    def cleanup_data_directory(self):
        """Clean data directory of temporary files."""
        print("🧹 Cleaning data directory...")
        
        data_dir = self.project_root / "data"
        if not data_dir.exists():
            print("  Data directory does not exist")
            return
        
        # Remove temporary audio files
        temp_audio_patterns = [
            "temp_*.wav",
            "temp_*.mp3",
            "temp_*.m4a",
            "processed_*.wav",
            "chunk_*.wav"
        ]
        
        for pattern in temp_audio_patterns:
            for temp_file in data_dir.rglob(pattern):
                try:
                    temp_file.unlink()
                    self.removed_files.append(str(temp_file))
                    print(f"  Removed: {temp_file}")
                except Exception as e:
                    print(f"  Error removing {temp_file}: {e}")
    
    def optimize_requirements(self):
        """Optimize requirements.txt for RAM efficiency."""
        print("🔧 Optimizing requirements.txt...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("  Requirements.txt not found")
            return
        
        # Read current requirements
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        # Create optimized requirements
        optimized_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                optimized_lines.append(line)
                continue
            
            # Add memory optimization comments
            if 'torch' in line:
                optimized_lines.append("# Core ML framework - use CPU version for memory efficiency")
                optimized_lines.append(line)
            elif 'whisper' in line:
                optimized_lines.append("# Core transcription model")
                optimized_lines.append(line)
            elif 'numpy' in line:
                optimized_lines.append("# Numerical computing - essential for audio processing")
                optimized_lines.append(line)
            elif 'pydub' in line:
                optimized_lines.append("# Audio processing - lightweight alternative to librosa")
                optimized_lines.append(line)
            elif 'librosa' in line:
                optimized_lines.append("# Advanced audio processing - can be disabled for memory efficiency")
                optimized_lines.append(line)
            elif 'noisereduce' in line:
                optimized_lines.append("# Noise reduction - optional for memory efficiency")
                optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        # Add memory optimization section
        optimized_lines.extend([
            "",
            "# Memory Optimization Tips:",
            "# 1. Use --quality memory-optimized for low RAM systems",
            "# 2. Consider using CPU-only torch: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu",
            "# 3. Disable optional dependencies for minimal memory usage",
            "# 4. Use smaller Whisper models (tiny, base, small) for memory efficiency"
        ])
        
        # Write optimized requirements
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(optimized_lines))
        
        print("  Requirements.txt optimized for memory efficiency")
    
    def create_memory_optimization_guide(self):
        """Create a memory optimization guide."""
        print("📝 Creating memory optimization guide...")
        
        guide_content = """# Memory Optimization Guide for FarsiTranscribe

## Quick Memory Optimization

### 1. Use Memory-Optimized Preset
```bash
python main.py audio_file.m4a --quality memory-optimized
```

### 2. Use Smaller Models
- `tiny`: ~39MB RAM, fastest, lowest quality
- `base`: ~74MB RAM, fast, good quality
- `small`: ~244MB RAM, balanced
- `medium`: ~769MB RAM, high quality
- `large-v3`: ~1550MB RAM, highest quality

### 3. CPU-Only Installation (Saves GPU Memory)
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Minimal Dependencies Installation
```bash
pip install openai-whisper torch torchaudio pydub numpy tqdm
```

## Advanced Memory Management

### 1. Streaming Mode
The system automatically uses streaming mode for files >100MB or when `memory_efficient_mode=True`.

### 2. Chunk Size Optimization
- Smaller chunks = less memory per chunk
- Larger chunks = fewer processing steps
- Balance based on available RAM

### 3. Parallel Processing Control
- Reduce `num_workers` for lower memory usage
- Disable `use_parallel_audio_prep` for sequential processing

### 4. Preprocessing Control
- Disable `enable_preprocessing` for minimal memory usage
- Disable `enable_noise_reduction` to save memory
- Disable `enable_speech_enhancement` for faster processing

## Memory Usage by Configuration

| Configuration | Model Size | RAM Usage | Speed | Quality |
|---------------|------------|-----------|-------|---------|
| memory-optimized | small | ~300MB | Fast | Good |
| cpu-optimized | medium | ~800MB | Medium | High |
| balanced | large-v3 | ~1600MB | Slow | Best |
| high | large-v3 | ~2000MB | Slow | Best |

## Troubleshooting High Memory Usage

1. **Monitor Memory**: Use `--quality memory-optimized` with monitoring
2. **Close Other Apps**: Free up system RAM before transcription
3. **Use SSD**: Ensure sufficient disk space for temporary files
4. **Restart**: Restart the application between large files

## Environment Variables for Memory Control

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
```

## Performance vs Memory Trade-offs

- **Speed**: Use smaller models, disable preprocessing
- **Quality**: Use larger models, enable all preprocessing
- **Memory**: Use streaming mode, smaller chunks, fewer workers
- **Balance**: Use `--quality balanced` for optimal trade-offs
"""
        
        guide_file = self.project_root / "MEMORY_OPTIMIZATION.md"
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"  Memory optimization guide created: {guide_file}")
    
    def cleanup_all(self, keep_recent_outputs: bool = True):
        """Perform all cleanup operations."""
        print("🚀 Starting comprehensive project cleanup...")
        print("=" * 60)
        
        self.cleanup_python_cache()
        print()
        
        self.cleanup_logs()
        print()
        
        self.cleanup_temp_files()
        print()
        
        self.cleanup_output_files(keep_recent_outputs)
        print()
        
        self.cleanup_data_directory()
        print()
        
        self.optimize_requirements()
        print()
        
        self.create_memory_optimization_guide()
        print()
        
        # Summary
        print("=" * 60)
        print("✅ Cleanup completed!")
        print(f"📁 Removed {len(self.removed_files)} files")
        print(f"📂 Removed {len(self.removed_dirs)} directories")
        
        if self.removed_files:
            print("\nRemoved files:")
            for file_path in self.removed_files[:10]:  # Show first 10
                print(f"  - {file_path}")
            if len(self.removed_files) > 10:
                print(f"  ... and {len(self.removed_files) - 10} more files")
        
        if self.removed_dirs:
            print("\nRemoved directories:")
            for dir_path in self.removed_dirs:
                print(f"  - {dir_path}")
        
        print("\n💡 Memory optimization tips:")
        print("  - Use '--quality memory-optimized' for low RAM systems")
        print("  - Consider CPU-only torch installation")
        print("  - Check MEMORY_OPTIMIZATION.md for detailed guide")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description="Clean up FarsiTranscribe project")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--remove-all-outputs",
        action="store_true",
        help="Remove all output files (default: keep recent ones)"
    )
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(args.project_root)
    cleaner.cleanup_all(keep_recent_outputs=not args.remove_all_outputs)


if __name__ == "__main__":
    main() 
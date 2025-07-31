# FarsiTranscribe

A high-performance, RAM-optimized audio transcription system specifically designed for Persian (Farsi) speech with advanced anti-repetition features and memory management.

## 🚀 Features

- **Persian Language Optimization**: Specialized for Persian/Farsi transcription
- **Memory Efficient**: RAM-optimized processing with streaming support
- **Anti-Repetition**: Advanced deduplication and repetition detection
- **Multiple Quality Presets**: From memory-optimized to high-quality
- **Streaming Processing**: Handles large audio files efficiently
- **GPU/CPU Support**: Optimized for both GPU and CPU-only systems
- **Real-time Preview**: Live transcription preview with sentence extraction

## 💾 Memory Optimization

This project has been extensively optimized for RAM efficiency:

### Quick Start for Low RAM Systems
```bash
# For systems with < 4GB RAM
python main.py audio.m4a --quality memory-optimized

# For systems with 4-8GB RAM  
python main.py audio.m4a --quality cpu-optimized

# For systems with > 8GB RAM
python main.py audio.m4a --quality balanced
```

### Memory Usage by Configuration

| Configuration | Model Size | RAM Usage | Speed | Quality |
|---------------|------------|-----------|-------|---------|
| memory-optimized | small | ~300MB | Fast | Good |
| cpu-optimized | medium | ~800MB | Medium | High |
| balanced | large-v3 | ~1600MB | Slow | Best |
| high | large-v3 | ~2000MB | Slow | Best |

### Memory Optimization Features

- **Streaming Audio Processing**: Processes audio in chunks to reduce memory usage
- **Automatic Memory Cleanup**: Regular garbage collection and CUDA cache clearing
- **Configurable Thresholds**: Adjustable memory limits and cleanup intervals
- **Model Sharing**: Efficient model sharing across processes
- **Optional Preprocessing**: Disable heavy preprocessing for memory savings

## 📦 Installation

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)

### Basic Installation
```bash
git clone https://github.com/yourusername/FarsiTranscribe.git
cd FarsiTranscribe
pip install -r requirements.txt
```

### Memory-Optimized Installation
For systems with limited RAM, use CPU-only PyTorch:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper pydub numpy tqdm
```

## 🎯 Usage

### Basic Transcription
```bash
python main.py examples/audio/BiVaZirsakht.m4a
```

### Memory-Optimized Transcription
```bash
# For low RAM systems
python main.py audio.m4a --quality memory-optimized

# For large files (auto-streaming)
python main.py large_audio.m4a --quality memory-optimized
```

### Quality Presets
```bash
# Fast transcription
python main.py audio.m4a --quality fast

# High quality (requires more RAM)
python main.py audio.m4a --quality high

# CPU optimized
python main.py audio.m4a --quality cpu-optimized

# Memory optimized
python main.py audio.m4a --quality memory-optimized
```

### Advanced Options
```bash
python main.py audio.m4a \
  --model small \
  --language fa \
  --output-dir ./output \
  --device cpu \
  --chunk-duration 15000 \
  --overlap 200
```

## 🧹 Project Cleanup

The project includes comprehensive cleanup tools:

### Automatic Cleanup
```bash
# Clean up unnecessary files and optimize structure
python scripts/cleanup_repo.py

# Remove all output files
python scripts/cleanup_repo.py --remove-all-outputs
```

### Memory Optimization Test
```bash
# Test memory optimization features
python scripts/test_memory_optimization.py
```

## 📊 Performance

### Memory Usage Optimization
- **Streaming Mode**: Automatically enabled for files >100MB
- **Chunk Processing**: Configurable chunk sizes for memory control
- **Parallel Processing**: Optimized worker count based on system resources
- **Garbage Collection**: Automatic memory cleanup during processing

### Speed vs Memory Trade-offs
- **Speed**: Use smaller models, disable preprocessing
- **Quality**: Use larger models, enable all preprocessing  
- **Memory**: Use streaming mode, smaller chunks, fewer workers
- **Balance**: Use `--quality balanced` for optimal trade-offs

## 🔧 Configuration

### Memory Management Settings
```python
# Memory threshold for cleanup (MB)
memory_threshold_mb: int = 1024

# Cleanup interval (seconds)
cleanup_interval_seconds: int = 30

# Streaming chunk size (MB)
streaming_chunk_size_mb: int = 50

# Enable memory monitoring
enable_memory_monitoring: bool = True
```

### Quality Presets
- **memory-optimized**: Smallest model, minimal preprocessing
- **cpu-optimized**: Medium model, moderate preprocessing
- **balanced**: Large model, full preprocessing
- **high**: Largest model, maximum quality

## 📁 Project Structure

```
FarsiTranscribe/
├── src/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── transcriber.py     # Main transcription engine
│   └── utils/
│       ├── audio_preprocessor.py    # Audio processing
│       ├── repetition_detector.py   # Anti-repetition logic
│       ├── performance_monitor.py   # Memory monitoring
│       └── file_manager.py          # Output management
├── scripts/
│   ├── cleanup_repo.py        # Project cleanup
│   └── test_memory_optimization.py  # Memory testing
├── examples/
│   └── audio/                 # Sample audio files
├── output/                    # Transcription outputs
├── requirements.txt           # Dependencies
└── MEMORY_OPTIMIZATION.md     # Detailed optimization guide
```

## 🛠️ Development

### Running Tests
```bash
# Test memory optimization
python scripts/test_memory_optimization.py

# Run transcription tests
python -m pytest tests/
```

### Memory Profiling
```bash
# Monitor memory usage during transcription
python main.py audio.m4a --quality memory-optimized
```

## 📚 Documentation

- [Memory Optimization Guide](MEMORY_OPTIMIZATION.md) - Detailed memory management
- [Configuration Reference](docs/configuration.md) - All configuration options
- [Performance Tuning](docs/performance.md) - Optimization strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test memory usage with `python scripts/test_memory_optimization.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for the transcription model
- Persian language community for feedback and testing
- Contributors for memory optimization improvements

---

**💡 Pro Tip**: For best performance on low-RAM systems, use `--quality memory-optimized` and ensure you have at least 2GB of free RAM before starting transcription. 
# 🚀 FarsiTranscribe Quick Start Guide

Get FarsiTranscribe up and running in minutes!

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies
```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Test Installation
```bash
# Run basic tests
python test_transcription.py

# Run examples
python examples/advanced_usage.py
```

### 3. Transcribe Your First Audio
```bash
# Use the sample audio file
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a

# Or use your own audio file
python main.py path/to/your/audio.mp3
```

## 🎯 Quality Presets

Choose the quality that fits your needs:

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `fast` | ⚡⚡⚡ | ⭐⭐⭐ | Quick transcription, large files |
| `balanced` | ⚡⚡ | ⭐⭐⭐⭐ | Good balance of speed and quality |
| `high` | ⚡ | ⭐⭐⭐⭐⭐ | High quality, smaller files |
| `95-percent` | 🐌 | ⭐⭐⭐⭐⭐ | Maximum quality, research |

## 🔧 Basic Usage

### Simple Transcription
```bash
# Basic transcription with Persian model
python main.py audio_file.mp3

# Specify output directory
python main.py audio_file.mp3 --output-dir ./results

# Use different quality
python main.py audio_file.mp3 --quality high
```

### Advanced Options
```bash
# Use different Whisper models
python main.py audio_file.mp3 --model large

# Disable preview
python main.py audio_file.mp3 --no-preview

# Combine options
python main.py audio_file.mp3 --quality high --output-dir ./results --model large
```

## 📁 Output Files

Transcription results are saved to the output directory with:

- **Text file**: Plain text transcription
- **JSON file**: Structured data with timestamps
- **Segments file**: Time-aligned text segments

## 🎵 Supported Audio Formats

- **Common formats**: MP3, WAV, M4A, FLAC, OGG
- **Video files**: MP4, AVI, MKV (audio extracted)
- **Streaming**: Real-time audio processing

## 🌟 Persian Language Features

- **Native Persian model**: `nezamisafa/whisper-persian-v4`
- **RTL text support**: Right-to-left Persian text
- **Persian optimization**: Language-specific preprocessing
- **Number conversion**: Persian/Arabic number handling

## 🚨 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Audio file not found:**
```bash
# Check file path
ls -la path/to/audio/file.mp3

# Use absolute path if needed
python main.py /full/path/to/audio.mp3
```

**Memory issues:**
```bash
# Use memory-optimized preset
python main.py audio_file.mp3 --quality memory-optimized

# Process smaller chunks
# (Edit config.py to reduce chunk_duration_ms)
```

### Performance Tips

- **Fast processing**: Use `--quality fast` for large files
- **High quality**: Use `--quality 95-percent` for research
- **Memory efficient**: Use `--quality memory-optimized` for limited RAM
- **GPU acceleration**: Install CUDA version of PyTorch if available

## 📚 Next Steps

### Learn More
- **Examples**: Check `examples/` directory for detailed usage
- **Configuration**: Explore `src/core/config.py` for advanced settings
- **Documentation**: Read `README.md` for comprehensive information

### Advanced Features
- **Custom models**: Use your own fine-tuned Whisper models
- **Batch processing**: Process multiple files simultaneously
- **API integration**: Use as a Python library in your projects

### Get Help
- **Issues**: Report bugs on GitHub
- **Examples**: Check working examples in the repository
- **Documentation**: Comprehensive guides and API reference

## 🎉 You're Ready!

FarsiTranscribe is now ready to transcribe your Persian audio files with high accuracy and performance. Start with the sample audio file to see it in action, then try your own files!

---

**Need help?** Check the examples, run the tests, or explore the documentation. Happy transcribing! 🎙️✨

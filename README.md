# FarsiTranscribe 2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, efficient, and extensible audio transcription system optimized for Persian/Farsi language using OpenAI's Whisper model.

## 🌟 Features

- **🎯 Persian-Optimized**: Uses fine-tuned Persian Whisper model (`nezamisafa/whisper-persian-v4`) for superior Farsi transcription
- **📦 Modular Design**: Clean architecture with separate modules for easy extension
- **⚡ Performance**: Efficient chunking and memory management for large files
- **🔧 Flexible Configuration**: Multiple presets and customization options
- **🔌 Extensible**: Hook system for adding custom functionality
- **📊 Multiple Output Formats**: Text, JSON, and timestamped segments
- **🖥️ CPU/GPU Support**: Optimized for both CPU and GPU processing
- **🤗 Hugging Face Integration**: Native support for Hugging Face models

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Extending FarsiTranscribe](#-extending-farsitranscribe)
- [Examples](#-examples)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio format support)
- CUDA toolkit (optional, for GPU support)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/siryoos/FarsiTranscribe.git
cd FarsiTranscribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## 🎯 Quick Start

### Persian Model Integration

This project now uses the **`nezamisafa/whisper-persian-v4`** model by default - a fine-tuned Whisper large-v3 model specifically optimized for Persian/Farsi transcription. This model provides superior accuracy for Persian language compared to the standard Whisper models.

### Command Line

```bash
# Basic transcription (uses Persian model by default)
python main.py audio.mp3

# High quality transcription with Persian model
python main.py audio.mp3 --quality high

# Use specific model (Persian model is default)
python main.py audio.mp3 --model nezamisafa/whisper-persian-v4

# Use standard Whisper model
python main.py audio.mp3 --model large

# Custom output directory
python main.py audio.mp3 --output-dir results/
```

### Python API

```python
from src.core.config import TranscriptionConfig, ConfigFactory
from src import UnifiedAudioTranscriber

# Create configuration with Persian model
config = ConfigFactory.create_persian_optimized_config()

# Initialize transcriber
with UnifiedAudioTranscriber(config) as transcriber:
    transcription = transcriber.transcribe_file("audio.mp3")
    print(transcription)
```

## 🏗️ Architecture

FarsiTranscribe follows a clean, modular architecture:

```
farsi_transcribe/
├── __init__.py       # Package exports
├── config.py         # Configuration management
├── audio.py          # Audio processing module
├── core.py           # Core transcription engine
├── utils.py          # Utilities and result management
├── cli.py            # Command-line interface
└── extensions/       # Extension modules (future)
```

### Key Components

1. **TranscriptionConfig**: Centralized configuration management
2. **AudioProcessor**: Handles audio loading, preprocessing, and chunking
3. **FarsiTranscriber**: Main transcription engine with hook support
4. **TranscriptionResult**: Results container with save methods
5. **TextProcessor**: Persian text normalization and processing

## 📖 Usage

### Configuration Presets

FarsiTranscribe provides several built-in presets:

- **fast**: Quick transcription with smaller model
- **balanced**: Good balance of speed and quality (default)
- **high-quality**: Best quality with larger model
- **memory-efficient**: For systems with limited RAM
- **persian-optimized**: Optimized specifically for Persian audio
- **gpu-optimized**: Optimized for GPU processing

### Custom Configuration

```python
from farsi_transcribe import TranscriptionConfig, FarsiTranscriber

# Create custom configuration
config = TranscriptionConfig(
    model_name="large-v3",
    language="fa",
    chunk_duration=30,
    overlap=5,
    device="cuda",
    persian_normalization=True,
    output_formats=["txt", "json", "segments"]
)

# Use with transcriber
transcriber = FarsiTranscriber(config)
```

### Streaming Large Files

For very large audio files, use streaming mode:

```python
# Command line
python -m farsi_transcribe large_audio.mp3 --stream

# Python API
result = transcriber.transcribe_stream("large_audio.mp3")
```

## 🔧 Configuration

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_name` | str | "base" | Whisper model size |
| `language` | str | "fa" | Language code |
| `chunk_duration` | int | 30 | Chunk duration in seconds |
| `overlap` | int | 3 | Overlap between chunks |
| `device` | str | auto | "cuda" or "cpu" |
| `persian_normalization` | bool | True | Normalize Persian text |
| `remove_diacritics` | bool | False | Remove Arabic diacritics |
| `output_formats` | list | ["txt", "json"] | Output formats |

### Environment Variables

```bash
# Set default model
export FARSI_TRANSCRIBE_MODEL=large-v3

# Set default output directory
export FARSI_TRANSCRIBE_OUTPUT_DIR=/path/to/output

# Force CPU usage
export FARSI_TRANSCRIBE_DEVICE=cpu
```

## 📚 API Reference

### FarsiTranscriber

```python
class FarsiTranscriber:
    def __init__(self, config: TranscriptionConfig)
    def transcribe_file(self, audio_path: Path) -> TranscriptionResult
    def transcribe_stream(self, audio_path: Path) -> TranscriptionResult
    def add_extension(self, extension) -> None
    def set_model(self, model_name: str) -> None
```

### TranscriptionResult

```python
class TranscriptionResult:
    text: str                    # Transcribed text
    chunks: List[Dict]          # Chunk information
    metadata: Dict              # Processing metadata
    
    def save_text(self, path: Path)
    def save_json(self, path: Path)
    def save_segments(self, path: Path)
```

## 🔌 Extending FarsiTranscribe

### Creating Extensions

Extensions allow you to add custom functionality without modifying core code:

```python
class CustomExtension:
    def install(self, transcriber):
        # Add pre-processing hook
        transcriber.hooks.add_pre_chunk_hook(self.pre_process)
        
        # Add post-processing hook
        transcriber.hooks.add_post_merge_hook(self.post_process)
    
    def pre_process(self, chunk):
        # Custom preprocessing
        return chunk
    
    def post_process(self, text):
        # Custom postprocessing
        return text

# Use extension
transcriber = FarsiTranscriber(config)
transcriber.add_extension(CustomExtension())
```

### Hook Points

1. **pre_chunk_hooks**: Before processing each audio chunk
2. **post_chunk_hooks**: After transcribing each chunk
3. **pre_merge_hooks**: Before merging chunk results
4. **post_merge_hooks**: After final text merge

## 📝 Examples

### Batch Processing

```python
from pathlib import Path
from farsi_transcribe import FarsiTranscriber, ConfigPresets

def batch_transcribe(audio_dir, output_dir):
    config = ConfigPresets.balanced()
    config.output_directory = Path(output_dir)
    
    with FarsiTranscriber(config) as transcriber:
        for audio_file in Path(audio_dir).glob("*.mp3"):
            print(f"Processing: {audio_file.name}")
            result = transcriber.transcribe_file(audio_file)
            
            # Save with same base name
            base_name = audio_file.stem
            result.save_text(output_dir / f"{base_name}.txt")
            result.save_json(output_dir / f"{base_name}.json")
```

### Custom Post-Processing

```python
class PunctuationExtension:
    """Add punctuation to transcribed text."""
    
    def install(self, transcriber):
        transcriber.hooks.add_post_merge_hook(self.add_punctuation)
    
    def add_punctuation(self, text):
        # Simple rule-based punctuation
        sentences = text.split()
        result = []
        
        for i, word in enumerate(sentences):
            result.append(word)
            # Add period at end of sentences (simplified)
            if i < len(sentences) - 1 and len(word) > 3:
                next_word = sentences[i + 1]
                if next_word[0].isupper():
                    result[-1] += '.'
        
        return ' '.join(result)
```

## ⚡ Performance Tips

### For Large Files

1. Use streaming mode for files over 1 hour
2. Increase chunk duration to reduce overhead
3. Use GPU if available
4. Adjust overlap based on audio content

### Memory Optimization

```python
# Memory-efficient configuration
config = ConfigPresets.memory_efficient()
config.clear_cache_every = 5  # Clear cache every 5 chunks
config.max_memory_gb = 2.0    # Limit memory usage
```

### GPU Optimization

```python
# GPU-optimized configuration
config = ConfigPresets.gpu_optimized()
config.batch_size = 4        # Process multiple chunks
config.use_fp16 = True       # Use half precision
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Use memory-efficient preset
   python -m farsi_transcribe audio.mp3 --preset memory-efficient
   ```

2. **Slow Processing**
   ```bash
   # Use faster model
   python -m farsi_transcribe audio.mp3 --model base --preset fast
   ```

3. **Poor Quality**
   ```bash
   # Use better model and settings
   python -m farsi_transcribe audio.mp3 --preset persian-optimized
   ```

### Debug Mode

```bash
# Enable verbose logging
python -m farsi_transcribe audio.mp3 --verbose

# Check log file
cat farsi_transcribe.log
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- The Persian NLP community
- All contributors and users

## 📮 Contact

For questions and support:
- Create an issue on GitHub
- Email: your.email@example.com

---

Made with ❤️ for the Persian-speaking community 
# FarsiTranscribe

A modular audio transcription system with anti-repetition features, specifically optimized for Persian (Farsi) audio transcription.

## Features

- 🎙️ **Persian-Optimized**: Specifically tuned for Persian/Farsi audio transcription
- 🔄 **Anti-Repetition**: Advanced algorithms to detect and remove repetitive content
- 🧩 **Modular Design**: Clean, maintainable codebase with separated concerns
- ⚡ **Performance Optimized**: GPU acceleration with intelligent batch processing
- 🎯 **Quality Control**: Multiple quality presets and confidence thresholds
- 📊 **Real-time Preview**: Live transcription preview with sentence extraction

## Project Structure

```
FarsiTranscribe/
├── src/                    # Main source code
│   ├── core/              # Core transcription logic
│   │   ├── config.py      # Configuration management
│   │   └── transcriber.py # Main transcription engine
│   └── utils/             # Utility modules
│       ├── file_manager.py
│       ├── repetition_detector.py
│       ├── sentence_extractor.py
│       └── terminal_display.py
├── tests/                 # Test suite
├── examples/              # Example usage and sample data
│   └── audio/            # Sample audio files
├── scripts/              # Utility scripts
├── output/               # Transcription output directory
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Modern Python packaging
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/farsitranscribe.git
   cd farsitranscribe
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run transcription:**
   ```bash
   python main.py path/to/your/audio/file.wav
   ```

## Usage

### Basic Transcription

```bash
# Basic transcription with default settings
python main.py audio_file.wav

# Persian-optimized transcription (recommended for Persian audio)
python main.py audio_file.wav --language fa
```

### Quality Presets

```bash
# Fast transcription (base model, lower quality)
python main.py audio_file.wav --quality fast

# Balanced transcription (default)
python main.py audio_file.wav --quality balanced

# High quality transcription (large model, best quality)
python main.py audio_file.wav --quality high
```

### Advanced Options

```bash
# Custom model and settings
python main.py audio_file.wav \
  --model large-v3 \
  --language fa \
  --output-dir ./output \
  --device cuda \
  --chunk-duration 20000 \
  --overlap 500
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--quality` | Quality preset (fast/balanced/high) | balanced |
| `--model` | Whisper model size | auto |
| `--language` | Language code | fa |
| `--output-dir` | Output directory | ./output |
| `--device` | Device (auto/cpu/cuda) | auto |
| `--no-preview` | Disable sentence preview | False |

## Configuration

The system uses intelligent configuration management with presets optimized for different use cases:

- **Persian Optimized**: Best for Persian/Farsi audio
- **Fast**: Quick transcription with base model
- **High Quality**: Maximum accuracy with large model
- **Custom**: Fully configurable parameters

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure

- **`src/core/`**: Core transcription logic and configuration
- **`src/utils/`**: Utility modules for file management, repetition detection, etc.
- **`tests/`**: Comprehensive test suite
- **`examples/`**: Sample usage and audio files
- **`scripts/`**: Utility scripts for debugging and maintenance

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [OpenAI Whisper](https://github.com/openai/whisper)
- Enhanced with custom anti-repetition algorithms
- Optimized specifically for Persian language transcription 
# FarsiTranscribe Project Structure

## 📁 Clean Project Layout

```
FarsiTranscribe/
├── farsi_transcribe/              # Main package directory
│   ├── __init__.py               # Package exports and version
│   ├── __main__.py               # Entry point for python -m farsi_transcribe
│   ├── config.py                 # Configuration management
│   ├── audio.py                  # Audio processing module
│   ├── core.py                   # Core transcription engine
│   ├── utils.py                  # Utilities and result management
│   └── cli.py                    # Command-line interface
│
├── examples/                      # Example scripts and audio
│   ├── basic_usage.py            # Usage examples
│   └── audio/                    # Sample audio files
│       └── *.mp3/m4a             # Persian audio samples
│
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_config.py            # Configuration tests
│   ├── test_audio.py             # Audio processing tests
│   ├── test_core.py              # Core functionality tests
│   └── test_utils.py             # Utility tests
│
├── docs/                          # Additional documentation
│   ├── API.md                    # API reference
│   ├── EXTENSIONS.md             # Extension development guide
│   └── PERFORMANCE.md            # Performance tuning guide
│
├── output/                        # Default output directory
│
├── requirements.txt               # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup configuration
├── README.md                     # Main documentation
├── PROJECT_STRUCTURE.md          # This file
├── LICENSE                       # MIT license
└── .gitignore                    # Git ignore rules
```

## 🧩 Module Descriptions

### Core Package (`farsi_transcribe/`)

#### `__init__.py`
- Package initialization
- Exports main classes and functions
- Version information

#### `config.py`
- `TranscriptionConfig`: Main configuration dataclass
- `ConfigPresets`: Factory for common configurations
- Validation and optimization logic

#### `audio.py`
- `AudioProcessor`: Audio loading and preprocessing
- `AudioChunk`: Chunk representation
- Support for multiple audio formats
- Streaming capabilities

#### `core.py`
- `FarsiTranscriber`: Main transcription engine
- `TranscriptionHooks`: Extension system
- Memory management
- Model handling

#### `utils.py`
- `TranscriptionResult`: Result container
- `TextProcessor`: Persian text processing
- `TranscriptionManager`: Output management
- Helper functions

#### `cli.py`
- Command-line argument parsing
- Configuration from CLI args
- Progress display
- Result formatting

## 🔌 Extension Points

The architecture is designed for extensibility:

1. **Hooks System** - Add custom processing at various stages
2. **Config Classes** - Easy to extend configuration
3. **Audio Processing** - Pluggable preprocessors
4. **Text Processing** - Custom normalization/cleaning
5. **Output Formats** - Add new output formats

## 🗑️ Files to Remove

When cleaning up the project, remove:

### Redundant Main Files
- `main.py` (old version)
- `main_multicore.py`
- `main_quality.py`
- `efficient.py`
- `optimized_transcriber.py`
- `batch_transcribe.py`

### Old Scripts
- `cleanup.py`
- `cleanup_final.py`
- `run_optimized.sh`
- `run_transcription.sh`

### Old Documentation
- `CHUNK_ANALYSIS_GUIDE.md`
- `CLEANUP_SUMMARY.md`
- `COMPREHENSIVE_FIX_SUMMARY.md`
- `MEMORY_OPTIMIZATION.md`

### Old Source Structure
- `src/` directory (replaced by `farsi_transcribe/`)
- `backup_original/` directory

### Installation Scripts
- `install_*.sh` files
- `activate_env.sh`

## 📝 Migration Guide

### From Old Structure to New

1. **Imports**
   ```python
   # Old
   from src import UnifiedAudioTranscriber
   from src.core.config import ConfigFactory
   
   # New
   from farsi_transcribe import FarsiTranscriber, ConfigPresets
   ```

2. **Configuration**
   ```python
   # Old
   config = ConfigFactory.create_optimized_config()
   
   # New
   config = ConfigPresets.balanced()
   ```

3. **Transcription**
   ```python
   # Old
   with UnifiedAudioTranscriber(config) as transcriber:
       transcription = transcriber.transcribe_file(audio)
   
   # New
   with FarsiTranscriber(config) as transcriber:
       result = transcriber.transcribe_file(audio)
       text = result.text
   ```

## 🚀 Quick Setup

```bash
# Clone and setup
git clone <repo>
cd FarsiTranscribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install package
pip install -e .

# Run transcription
python -m farsi_transcribe examples/audio/sample.mp3
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black farsi_transcribe/

# Lint
flake8 farsi_transcribe/
```

## 📦 Building

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## 🔍 Key Design Decisions

1. **Modular Architecture** - Clear separation of concerns
2. **Configuration-Driven** - Easy to customize behavior
3. **Hook System** - Extensible without modifying core
4. **Memory Efficient** - Streaming and chunk processing
5. **Persian-First** - Optimized for Farsi/Persian text

## 📈 Future Enhancements

- [ ] Plugin system for custom models
- [ ] Web API interface
- [ ] Real-time transcription
- [ ] Multi-language support
- [ ] Cloud storage integration
- [ ] Batch processing CLI
- [ ] GUI application

---

This structure provides a clean, maintainable, and extensible foundation for FarsiTranscribe.
# FarsiTranscribe 🎙️

A modular, object-oriented audio transcription system with advanced anti-repetition features, specifically optimized for Persian (Farsi) audio transcription.

## Features ✨

- **Modular Architecture**: Clean separation of concerns with extensible components
- **Anti-Repetition Technology**: Advanced algorithms to detect and remove repetitive content
- **Multi-Format Support**: Handles various audio formats (WAV, MP3, M4A, etc.)
- **Quality Presets**: Fast, balanced, and high-quality transcription modes
- **GPU Acceleration**: Automatic CUDA detection and optimization
- **Progress Tracking**: Real-time progress bars and sentence previews
- **Multiple Output Formats**: TXT, MD, JSON export options
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation 📦

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Quick Start 🚀

### Command Line Usage

```bash
# Basic transcription
python main.py audio_file.wav

# High quality transcription
python main.py audio_file.wav --quality high

# Fast transcription
python main.py audio_file.wav --quality fast

# Custom configuration
python main.py audio_file.wav --model large-v3 --language fa --output-dir ./output
```

### Python API Usage

```python
from src import UnifiedAudioTranscriber
from src.core.config import ConfigFactory

# Create configuration
config = ConfigFactory.create_optimized_config(
    model_size="large-v3",
    language="fa",
    output_dir="./output"
)

# Run transcription
with UnifiedAudioTranscriber(config) as transcriber:
    transcription = transcriber.transcribe_file("audio_file.wav")
    print(f"Transcription: {transcription}")
```

## Project Structure 📁

```
FarsiTranscribe/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   └── transcriber.py      # Main transcription engine
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── file_manager.py     # File I/O operations
│       ├── repetition_detector.py  # Anti-repetition algorithms
│       └── sentence_extractor.py   # Text processing utilities
├── examples/                    # Usage examples
│   └── basic_usage.py
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Configuration Options ⚙️

### Quality Presets

- **Fast**: Optimized for speed, uses smaller models
- **Balanced**: Good balance of speed and quality (default)
- **High**: Maximum quality, uses largest models

### Model Options

- `tiny`: Fastest, lowest quality
- `base`: Fast, basic quality
- `small`: Good balance
- `medium`: Better quality
- `large`: High quality
- `large-v2`: Very high quality
- `large-v3`: Best quality (recommended)

### Advanced Options

```bash
# Custom chunk duration (milliseconds)
python main.py audio.wav --chunk-duration 15000

# Custom overlap between chunks
python main.py audio.wav --overlap 500

# Custom repetition threshold (0.0-1.0)
python main.py audio.wav --repetition-threshold 0.8

# Maximum word repetitions
python main.py audio.wav --max-word-repetition 2

# Force CPU usage
python main.py audio.wav --device cpu

# Disable preview
python main.py audio.wav --no-preview
```

## Output Files 📄

The system generates multiple output files:

- `{filename}_unified_transcription.txt`: Original transcription
- `{filename}_cleaned_transcription.txt`: Cleaned transcription (recommended)
- `{filename}_metadata.json`: Processing metadata and statistics
- `transcription.log`: Detailed processing log

## Extending the System 🔧

### Adding New Audio Processors

```python
from src.core.transcriber import AudioProcessor

class CustomAudioProcessor(AudioProcessor):
    def process_audio(self, audio_file_path: str) -> AudioSegment:
        # Custom audio processing logic
        pass
    
    def create_chunks(self, audio: AudioSegment) -> List[Tuple[int, int]]:
        # Custom chunking logic
        pass
```

### Adding New Transcription Engines

```python
from src.core.transcriber import WhisperTranscriber

class CustomWhisperTranscriber(WhisperTranscriber):
    def transcribe_chunk(self, chunk: np.ndarray) -> str:
        # Custom transcription logic
        pass
```

### Custom Configuration

```python
from src.core.config import TranscriptionConfig

config = TranscriptionConfig(
    model_name="large-v3",
    language="fa",
    chunk_duration_ms=20000,
    overlap_ms=200,
    repetition_threshold=0.85,
    max_word_repetition=2
)
```

## Performance Tips 💡

1. **GPU Usage**: Ensure CUDA is available for best performance
2. **Memory Management**: Large models require significant RAM/VRAM
3. **Batch Size**: Adjust based on your GPU memory
4. **Chunk Size**: Larger chunks reduce processing overhead but may miss context
5. **Quality vs Speed**: Choose appropriate quality preset for your needs

## Troubleshooting 🔧

### Common Issues

**CUDA Out of Memory:**
- Reduce batch size: `--batch-size 1`
- Use smaller model: `--model base`
- Force CPU: `--device cpu`

**Poor Transcription Quality:**
- Use larger model: `--model large-v3`
- Increase chunk duration: `--chunk-duration 30000`
- Use high quality preset: `--quality high`

**Repetitive Output:**
- Adjust repetition threshold: `--repetition-threshold 0.7`
- Increase max word repetition: `--max-word-repetition 3`

### Logs

Check `transcription.log` for detailed error information and processing statistics.

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Built on OpenAI's Whisper model
- Uses PyTorch for GPU acceleration
- Audio processing powered by pydub
- Progress tracking with tqdm

## Support 💬

For issues, questions, or contributions, please open an issue on GitHub. 
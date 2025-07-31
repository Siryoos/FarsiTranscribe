# Quick Start Guide

## 🚀 Get Started in 30 Seconds

### 1. Activate Environment
```bash
source venv/bin/activate
```

### 2. Run Transcription
```bash
# Basic transcription (recommended)
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a

# Persian-optimized transcription
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --language fa

# High quality transcription
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --quality high

# Fast transcription
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --quality fast
```

### 3. Check Output
```bash
# View transcription results
cat output/unified_transcription.txt
```

## 📁 File Locations

- **Sample Audio**: `examples/audio/jalase bi va zirsakht.m4a`
- **Output Directory**: `output/`
- **Main Script**: `main.py`

## 🔧 Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--quality` | Quality preset | `--quality high` |
| `--language` | Language code | `--language fa` |
| `--output-dir` | Output directory | `--output-dir ./my_output` |
| `--device` | Processing device | `--device cpu` |

## 🆘 Need Help?

```bash
# Show all options
python main.py --help

# Run tests
pytest tests/

# Setup development environment
python scripts/setup_dev.py
```

## 📝 Example Commands

```bash
# Quick test with fast quality
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --quality fast

# Full quality Persian transcription
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --language fa --quality high

# Custom output directory
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --output-dir ./my_transcriptions
```

That's it! 🎉 
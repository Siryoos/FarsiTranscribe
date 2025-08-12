# Advanced Speaker Diarization with pyannote.audio

This project now includes **two speaker diarization systems**:

1. **🎯 Advanced (Recommended)**: pyannote.audio - Much better accuracy, handles overlaps
2. **🔄 Basic (Fallback)**: MFCC-based - Simple but less accurate

## Quick Start

### 1. Install Dependencies

```bash
# Run the helper script
python install_pyannote.py

# Or install manually
pip install pyannote.audio==3.1.1 torch torchaudio soundfile rich
```

### 2. Basic Usage

```bash
# Automatic speaker detection
python main.py your_audio.wav --quality 95-percent

# Specify exact speaker count (recommended for known conversations)
python main.py your_audio.wav --quality 95-percent --num-speakers 2

# Specify speaker range
python main.py your_audio.wav --quality 95-percent --min-speakers 2 --max-speakers 4

# Disable diarization (get full transcript)
python main.py your_audio.wav --quality 95-percent --no-diarization
```

## What You Get

### With Speaker Diarization Enabled:
- **Speaker-labeled transcript**: `[Speaker 0] (0:00-0:15): Hello, how are you?`
- **Per-speaker files**: Individual transcript files for each speaker
- **Segment JSON**: Detailed timing and confidence data
- **Unified output**: Complete transcript with speaker labels

### Output Files:
```
output/
├── your_audio_speaker_transcription.txt    # Speaker-labeled transcript
├── your_audio_speaker_segments.json        # Detailed segment data
├── your_audio_unified_transcription.txt    # Unified with speaker labels
└── your_audio_cleaned_transcription.txt    # Cleaned version
```

## Advanced Features

### Speaker Count Control
```bash
# For 2-person conversations (most accurate)
--num-speakers 2

# For meetings with 3-6 people
--min-speakers 3 --max-speakers 6

# Let system auto-detect (less accurate)
# (no flags)
```

### Quality Presets
```bash
--quality 95-percent      # Maximum quality, includes diarization
--quality high            # High quality with diarization
--quality balanced        # Balanced quality
--quality fast            # Fast processing, no diarization
```

## How It Works

### 1. Audio Analysis
- Loads entire audio file for diarization
- Converts to proper format (16kHz mono, float32)

### 2. Speaker Detection
- **pyannote.audio**: Uses neural networks to detect speech segments
- **Basic fallback**: Energy-based VAD + MFCC clustering

### 3. Segment Processing
- Each speech segment is transcribed individually
- Speaker labels are preserved throughout

### 4. Output Generation
- Creates speaker-labeled transcript
- Saves individual speaker files
- Generates detailed metadata

## Troubleshooting

### Low Coverage Issues
If diarization only detects a small portion of your audio:

```bash
# Force full transcription
python main.py your_audio.wav --no-diarization

# Try specifying speaker count
python main.py your_audio.wav --num-speakers 2

# Check audio quality (16kHz mono recommended)
```

### Installation Issues
```bash
# If pyannote.audio fails to install
pip install --upgrade pip setuptools wheel
pip install pyannote.audio==3.1.1

# For CUDA issues, use CPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
```bash
# Use memory-optimized preset
python main.py your_audio.wav --quality memory-optimized

# Force CPU mode
python main.py your_audio.wav --quality 95-percent --force-cpu
```

## Performance Tips

### For Long Files (>1 hour):
- Consider splitting into 15-30 minute chunks
- Use `--num-speakers` if you know the count
- Monitor memory usage

### For Best Accuracy:
- Use `--num-speakers` when you know the exact count
- Ensure audio is 16kHz mono
- Avoid heavily compressed formats

### For Speed:
- Use `--quality fast` for quick results
- Use `--no-diarization` for full transcript without speaker separation

## Technical Details

### pyannote.audio Features:
- **Neural VAD**: Better speech detection than energy-based methods
- **Speaker Embeddings**: More accurate speaker clustering
- **Overlap Handling**: Can detect when multiple people speak simultaneously
- **Robust Clustering**: Handles varying audio conditions better

### Fallback System:
- Automatically falls back to basic diarizer if pyannote fails
- Falls back to standard transcription if diarization fails completely
- Ensures you always get a complete transcript

### Audio Processing:
- Automatic format conversion to 16kHz mono
- Float32 normalization to [-1, 1] range
- Proper handling of different bit depths

## Examples

### 2-Person Conversation:
```bash
python main.py meeting.wav --quality 95-percent --num-speakers 2
```

### Multi-Person Meeting:
```bash
python main.py conference.wav --quality 95-percent --min-speakers 3 --max-speakers 8
```

### Quick Full Transcript:
```bash
python main.py long_audio.wav --quality 95-percent --no-diarization
```

### Memory-Constrained Environment:
```bash
python main.py audio.wav --quality memory-optimized --force-cpu
```

## Support

If you encounter issues:
1. Check the logs for error messages
2. Try the `--no-diarization` flag to get full transcript
3. Verify audio format (16kHz mono recommended)
4. Check available memory and CPU resources

The system is designed to be robust and will always provide a complete transcription, even if the advanced features fail.

# Enhanced Preview Display System

## Overview

The Enhanced Preview Display System provides a real-time, interactive preview of the transcription process, similar to the interface shown in your image. It displays:

- **Real-time chunk progress** with percentage bars
- **Chunk-by-chunk transcription details** with Persian text
- **Timing information** for each chunk
- **Progress overview** showing all chunks and their status
- **Speaker identification** (when available)
- **Interactive command prompt** (Ctrl+K)

## Features

### 🎯 Real-time Progress Tracking
- Live updates every 500ms
- Progress bars for each chunk
- Status indicators (🔄 transcribing, ✅ completed, ❌ failed)

### 📊 Detailed Chunk Information
- Chunk ID and numbering
- Transcription progress percentage
- Visual progress bars using Unicode characters
- Persian text preview with proper RTL support
- Speaker icons (🔊 default, 👤X for identified speakers)

### ⏱️ Timing and Duration
- Start and end times for each chunk
- Chunk duration in appropriate units (s, m, h)
- Estimated time remaining
- Elapsed time tracking

### 🎨 Rich Terminal Support
- Beautiful formatting with Rich library
- Color-coded status indicators
- Responsive layout
- Fallback to simple display for basic terminals

## Installation

The enhanced preview system is included by default. To ensure optimal experience:

```bash
# Install Rich library for enhanced display
pip install rich

# Or install all requirements
pip install -r requirements.txt
```

## Usage

### Automatic Activation

The enhanced preview automatically activates when:

1. `enable_sentence_preview` is `True` in configuration
2. Rich library is available
3. Running transcription with `main.py`

### Manual Usage

```python
from src.utils.enhanced_preview_display import create_preview_display

# Create preview display
preview = create_preview_display(
    total_chunks=100,
    estimated_duration=300,  # 5 minutes
    use_enhanced=True
)

# Use as context manager
with preview:
    # Add chunks
    preview.add_chunk(chunk_id=0, start_time=0, end_time=3, duration=3)
    
    # Update progress
    preview.update_chunk_progress(chunk_id=0, progress=50.0)
    
    # Set current chunk
    preview.set_current_chunk(chunk_id=0)
    
    # Add transcribed text
    preview.update_chunk_progress(chunk_id=0, progress=100.0, text="سلام، چطور هستید؟")
```

## Display Modes

### Enhanced Mode (Rich Library Available)
- Full color support
- Progress bars and animations
- Responsive layout
- Professional appearance

### Simple Mode (Fallback)
- Basic text-based display
- Unicode progress bars
- Compatible with all terminals

## Configuration

### Preview Settings

```python
from src.core.config import TranscriptionConfig

config = TranscriptionConfig(
    enable_sentence_preview=True,      # Enable preview system
    preview_sentence_count=2,          # Number of sentences to preview
    # ... other settings
)
```

### Command Line Options

```bash
# Enable preview (default)
python main.py audio_file.mp3

# Disable preview
python main.py audio_file.mp3 --no-preview

# Force enhanced preview
export RICH_FORCE_COLOR=1
python main.py audio_file.mp3
```

## Testing

Test the enhanced preview system:

```bash
# Test with simulation
python test_enhanced_preview.py

# Test with actual audio
python main.py examples/audio/sample.m4a --quality fast
```

## Customization

### Display Settings

```python
# Modify display behavior
preview.max_preview_length = 100      # Text preview length
preview.show_timing = True            # Show timing information
preview.show_progress_bars = True     # Show progress bars
preview.show_speaker_icons = True     # Show speaker icons
```

### Custom Chunk Information

```python
# Add custom chunk data
preview.add_chunk(
    chunk_id=0,
    start_time=0.0,
    end_time=3.0,
    duration=3.0,
    speaker_id="Speaker_1"  # Custom speaker ID
)
```

## Troubleshooting

### Common Issues

1. **Preview not showing**: Check `enable_sentence_preview` setting
2. **Poor formatting**: Install Rich library (`pip install rich`)
3. **Text not displaying**: Check terminal encoding and font support
4. **Performance issues**: Reduce update frequency in display loop

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check terminal capabilities
from src.utils.enhanced_preview_display import get_terminal_capabilities
caps = get_terminal_capabilities()
print(caps)
```

## Integration with Main System

The enhanced preview automatically integrates with:

- **UnifiedAudioTranscriber**: Main transcription engine
- **StreamingAudioProcessor**: Audio chunk processing
- **SpeakerDiarizer**: Speaker identification
- **MemoryManager**: Resource management

## Performance Considerations

- **Update frequency**: 500ms default (adjustable)
- **Memory usage**: Minimal overhead
- **CPU usage**: Low impact on transcription
- **Threading**: Non-blocking display updates

## Future Enhancements

- **Web interface**: Browser-based preview
- **Export options**: Save preview as HTML/PDF
- **Custom themes**: User-defined color schemes
- **Keyboard shortcuts**: Interactive controls
- **Real-time editing**: In-place text correction

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review terminal capabilities
3. Test with simple mode
4. Check Rich library installation
5. Verify configuration settings

---

**Note**: The enhanced preview system is designed to work seamlessly with the existing transcription pipeline while providing a much better user experience during long transcription processes.

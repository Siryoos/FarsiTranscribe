# RTL (Right-to-Left) Display Enhancement for FarsiTranscribe

## Problem Solved

Your Persian/Farsi text was displaying incorrectly in the terminal because:

1. **BiDi Text Issues**: Persian text requires right-to-left rendering
2. **Character Shaping**: Persian characters change form based on position
3. **Terminal Limitations**: Most terminals lack complex text layout support
4. **Unicode Handling**: Incomplete Unicode support in display components

## Solution Architecture

### Enhanced Components

1. **`rtl_terminal_display.py`** - Core RTL display system with:
   - Automatic capability detection
   - RTL text processing with caching
   - Multiple display modes with fallbacks
   - Performance optimization

2. **`terminal_display.py`** - Updated backward-compatible wrapper
3. **`sentence_extractor.py`** - Enhanced preview formatting

### Display Modes

The system automatically selects the best display mode:

- **RTL_RICH**: Full RTL + Rich formatting (best experience)
- **RTL_PLAIN**: RTL support with plain text
- **LTR_RICH**: Rich formatting without RTL (fallback)
- **LTR_PLAIN**: Basic text display
- **FALLBACK**: Minimal compatibility mode

## Installation & Setup

### Quick Setup

```bash
# 1. Install RTL libraries
./install_rtl_support.sh

# 2. Test the display
python test_rtl_display.py

# 3. Run transcription
python main.py your_audio.m4a --language fa
```

### Manual Installation

```bash
# Install RTL libraries
pip install python-bidi>=0.4.2 arabic-reshaper>=3.0.0

# Upgrade existing dependencies
pip install --upgrade rich colorama tqdm
```

## Technical Details

### RTL Text Processing

```python
# The system automatically:
1. Reshapes Arabic/Persian characters for proper joining
2. Applies bidirectional algorithm for RTL layout
3. Caches results for performance
4. Falls back gracefully if libraries unavailable
```

### Performance Features

- **Caching**: Processed text is cached to avoid recomputation
- **Lazy Loading**: RTL libraries loaded only when needed
- **Capability Detection**: Automatic terminal feature detection
- **Fallback Mechanisms**: Multiple fallback levels ensure compatibility

### Architecture Benefits

- **Modular Design**: Separate concerns for different display components
- **Scalable**: Easy to extend with new display modes
- **Backward Compatible**: Existing code continues to work
- **Performance Optimized**: Minimal overhead when RTL not needed

## Usage Examples

### Automatic Usage (Recommended)

The system works automatically when you run transcription:

```bash
python main.py examples/audio/jalase\ bi\ va\ zirsakht.m4a --language fa
```

### Direct Usage

```python
from src.utils.rtl_terminal_display import enhanced_rtl_display

# Display Persian text with proper RTL rendering
enhanced_rtl_display.print_persian_preview("سلام دنیا", 1)

# Check system capabilities
enhanced_rtl_display.print_configuration_info()
```

## Troubleshooting

### Common Issues

1. **Text still appears wrong**
   - Run: `python test_rtl_display.py`
   - Check terminal Unicode support
   - Install Persian-capable font

2. **Missing dependencies**
   - Run: `./install_rtl_support.sh`
   - Or manually: `pip install python-bidi arabic-reshaper`

3. **Performance issues**
   - RTL processing is cached for performance
   - Fallback modes available for slower systems

### Terminal Compatibility

**Best Experience:**
- iTerm2 (macOS)
- Terminal.app (macOS) 
- Windows Terminal
- GNOME Terminal (Linux)

**Font Recommendations:**
- Noto Sans Arabic
- Iran Sans
- Vazir Font
- Any Unicode-compliant font with Persian support

## Configuration

### Environment Variables

```bash
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
```

### Terminal Settings

1. Set encoding to UTF-8
2. Install Persian-capable font
3. Enable Unicode support

## Files Modified/Added

### New Files
- `src/utils/rtl_terminal_display.py` - Core RTL system
- `install_rtl_support.sh` - Installation script
- `test_rtl_display.py` - Test script
- `RTL_ENHANCEMENT_README.md` - This documentation

### Modified Files
- `src/utils/terminal_display.py` - Enhanced with RTL support
- `src/utils/sentence_extractor.py` - Better preview formatting
- `requirements.txt` - Added RTL libraries

## Performance Impact

- **Minimal Overhead**: RTL processing only when needed
- **Caching**: Processed text cached for reuse
- **Fallbacks**: Automatic degradation for better performance
- **Memory Efficient**: Smart cache management

## Future Enhancements

- Additional language support (Arabic, Hebrew, etc.)
- Advanced text shaping options
- Custom display themes
- Terminal-specific optimizations

## Testing

Run comprehensive tests:

```bash
# Test all functionality
python test_rtl_display.py

# Test specific components
python -c "from src.utils.rtl_terminal_display import enhanced_rtl_display; enhanced_rtl_display.print_configuration_info()"
```

## Support

If you encounter issues:

1. Run the test script: `python test_rtl_display.py`
2. Check terminal Unicode support
3. Verify RTL libraries installation
4. Try different display modes manually

The system is designed to gracefully degrade, so even if RTL libraries aren't available, you'll get readable output with improved formatting.

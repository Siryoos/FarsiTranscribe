#!/bin/bash

# FarsiTranscribe RTL Enhancement Installation Script
# This script installs RTL libraries and tests Persian text display

set -e

echo "🔧 FarsiTranscribe RTL Enhancement Installation"
echo "================================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Activating venv..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ venv directory not found. Please create a virtual environment first:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        exit 1
    fi
fi

echo ""
echo "📦 Installing RTL libraries for Persian text support..."

# Install RTL libraries
pip install python-bidi>=0.4.2 arabic-reshaper>=3.0.0

echo ""
echo "🔄 Upgrading existing dependencies..."

# Upgrade other dependencies to ensure compatibility
pip install --upgrade rich colorama tqdm

echo ""
echo "🧪 Testing Persian text display..."

# Create a test script
cat > test_rtl_display.py << 'EOF'
#!/usr/bin/env python3
"""Test script for RTL display functionality."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.rtl_terminal_display import enhanced_rtl_display
    
    print("🎉 RTL Display System Test")
    print("=" * 40)
    
    # Test Persian text samples
    test_texts = [
        "سلام دنیا، این یک تست نمایش متن فارسی است.",
        "آیا این متن به درستی نمایش داده می‌شود؟",
        "تست شماره ۱۲۳ با اعداد فارسی",
        "Mixed text: Persian متن فارسی and English"
    ]
    
    for i, text in enumerate(test_texts, 1):
        enhanced_rtl_display.print_persian_preview(text, i)
        print()
    
    # Print configuration
    enhanced_rtl_display.print_configuration_info()
    
    print("\n✅ RTL Display test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure you're running this from the FarsiTranscribe directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
EOF

# Run the test
echo "Running RTL display test..."
python test_rtl_display.py

# Clean up test file
rm -f test_rtl_display.py

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Your Persian text should now display correctly"
echo "2. Run your transcription with: python main.py [audio_file] --language fa"
echo "3. If you still see display issues, check your terminal's Unicode support"
echo ""
echo "💡 Tips for better display:"
echo "   • Use a modern terminal (iTerm2, Terminal.app, etc.)"
echo "   • Ensure your terminal font supports Persian characters"
echo "   • Set terminal encoding to UTF-8"
echo ""

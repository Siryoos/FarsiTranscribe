#!/bin/bash

# FarsiTranscribe - Virtual Environment Runner
# This script activates the virtual environment and runs the transcription system

echo "🎙️ FarsiTranscribe - Virtual Environment Setup"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if packages are installed
echo "📦 Checking dependencies..."
if ! python -c "import rich, colorama, blessed" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Virtual environment is ready!"
echo ""
echo "To run transcription:"
echo "  python main.py <audio_file>"
echo ""
echo "To run the enhanced script:"
echo "  python main.py"
echo ""
echo "To test Unicode support:"
echo "  python test_unicode.py"
echo ""
echo "To deactivate virtual environment:"
echo "  deactivate"
echo ""

# Keep the virtual environment active
echo "🔧 Virtual environment is now active. You can run your transcription commands."
echo "Type 'deactivate' to exit the virtual environment." 
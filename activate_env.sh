#!/bin/bash

# Simple virtual environment activation script
echo "🎙️ Activating FarsiTranscribe Virtual Environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python -c "import rich, colorama, blessed, whisper" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed."
fi

echo "✅ Virtual environment is ready!"
echo ""
echo "You can now run:"
echo "  python main.py <audio_file>"
echo "  python main.py"
echo ""
echo "Type 'deactivate' to exit the virtual environment." 
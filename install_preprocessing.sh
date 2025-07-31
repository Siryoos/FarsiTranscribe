#!/bin/bash
# Quick wins audio preprocessing installation

echo "📦 Installing audio preprocessing dependencies..."

# Install preprocessing libraries
pip install noisereduce>=3.0.0 librosa>=0.10.0 webrtcvad>=2.0.10 scipy>=1.9.0

echo "✅ Audio preprocessing dependencies installed"
echo ""
echo "🔧 Testing new capabilities..."

# Test preprocessing availability
python -c "
from src.utils.audio_preprocessor import get_preprocessing_capabilities
caps = get_preprocessing_capabilities()
print('📊 Preprocessing Capabilities:')
for feature, available in caps.items():
    status = '✅' if available else '❌'
    print(f'   {status} {feature}')
"

echo ""
echo "🚀 Ready to use enhanced transcription:"
echo "   python main.py [audio_file] --language fa --quality cpu-optimized"

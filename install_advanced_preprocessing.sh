#!/bin/bash
# Advanced preprocessing installation with Facebook Denoiser

echo "🚀 Installing Advanced Audio Preprocessing..."

# Install Facebook Denoiser (optional but recommended)
echo "📦 Installing Facebook Denoiser..."
pip install torch torchaudio
pip install git+https://github.com/facebookresearch/denoiser.git

# Verify installation
echo "🧪 Testing advanced capabilities..."
python -c "
try:
    from src.utils.advanced_audio_preprocessor import get_advanced_preprocessing_capabilities
    caps = get_advanced_preprocessing_capabilities()
    print('📊 Advanced Capabilities:')
    for feature, available in caps.items():
        status = '✅' if available else '❌'
        print(f'   {status} {feature}')
    
    if caps['facebook_denoiser']:
        print('\\n🎉 Facebook Denoiser ready!')
    else:
        print('\\n⚠️  Facebook Denoiser not available')
        
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "🎯 Advanced Usage:"
echo "   # Persian-optimized with all features:"
echo "   python main.py [audio] --quality advanced-persian"
echo ""
echo "   # Facebook Denoiser for noisy audio:"
echo "   python main.py [audio] --quality facebook-denoiser"

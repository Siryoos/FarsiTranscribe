"""
Setup configuration for FarsiTranscribe package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="farsitranscribe",
    version="2.0.0",
    author="FarsiTranscribe Team",
    author_email="siryoosa@gmail.com",
    description="A clean, efficient, and extensible audio transcription system optimized for Persian/Farsi language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siryoos/FarsiTranscribe",
    packages=find_packages(include=["farsi_transcribe*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Natural Language :: Persian",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai-whisper>=20231117",
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0,<3.0",
        "transformers>=4.35.0",
        "huggingface-hub>=0.19.0",
        "requests>=2.28.0",
        "urllib3>=1.26.0",
        "pydub>=0.25.1",
        "ffmpeg-python>=0.2.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.9.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "hazm>=0.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "farsi-transcribe=farsi_transcribe.cli:main",
            "farsitranscribe=farsi_transcribe.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "farsi_transcribe": ["*.json", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/siryoos/FarsiTranscribe/issues",
        "Source": "https://github.com/siryoos/FarsiTranscribe",
        "Documentation": "https://github.com/siryoos/FarsiTranscribe/wiki",
    },
    keywords=[
        "speech-recognition",
        "transcription",
        "persian",
        "farsi",
        "whisper",
        "audio-processing",
        "nlp",
        "speech-to-text",
    ],
) 
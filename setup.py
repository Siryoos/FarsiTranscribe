#!/usr/bin/env python3
"""
Setup script for FarsiTranscribe package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="farsitranscribe",
    version="1.0.0",
    author="FarsiTranscribe Team",
    author_email="contact@farsitranscribe.com",
    description="A modular audio transcription system with anti-repetition features for Persian (Farsi)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/farsitranscribe",
    packages=find_packages(),
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
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "farsitranscribe=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "transcription",
        "audio",
        "speech-to-text",
        "persian",
        "farsi",
        "whisper",
        "nlp",
        "ai",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/farsitranscribe/issues",
        "Source": "https://github.com/yourusername/farsitranscribe",
        "Documentation": "https://farsitranscribe.readthedocs.io/",
    },
) 
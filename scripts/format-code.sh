#!/bin/bash
# Format all Python files with Black

echo "🎨 Formatting Python code with Black..."

# Install black
pip install black

# Format all Python files
black src/ tests/ main.py

echo "✅ Code formatting complete!"

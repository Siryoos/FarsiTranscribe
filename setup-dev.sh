#!/bin/bash
# Setup script for FarsiTranscribe development environment

echo "🚀 Setting up FarsiTranscribe development environment..."

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Run pre-commit on all files
echo "🔍 Running pre-commit on all files..."
pre-commit run --all-files || true

# Create output directories
echo "📁 Creating required directories..."
mkdir -p output data tests/fixtures

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Sign up for Codecov at https://codecov.io and add CODECOV_TOKEN to GitHub secrets"
echo "2. Enable Dependabot in GitHub repository settings"
echo "3. Run 'pytest' to ensure tests pass"
echo "4. Create your first release with 'git tag v2.0.0 && git push --tags'"

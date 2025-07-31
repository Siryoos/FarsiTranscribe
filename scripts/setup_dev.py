#!/usr/bin/env python3
"""
Development environment setup script for FarsiTranscribe.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔄 Creating virtual environment...")
    if run_command("python -m venv venv", "Creating virtual environment"):
        print("✅ Virtual environment created")
        return True
    return False


def install_dependencies():
    """Install project dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing production dependencies"),
        ("pip install -e .[dev]", "Installing development dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def run_tests():
    """Run basic tests to verify installation."""
    print("🧪 Running basic tests...")
    if run_command("python -m pytest tests/ -v --tb=short", "Running tests"):
        print("✅ Tests passed")
        return True
    else:
        print("⚠️ Some tests failed, but installation may still be functional")
        return True


def setup_git_hooks():
    """Setup git hooks for code quality."""
    hooks_dir = Path(".git/hooks")
    if not hooks_dir.exists():
        print("⚠️ Git hooks directory not found, skipping git hooks setup")
        return True
    
    pre_commit_hook = hooks_dir / "pre-commit"
    if not pre_commit_hook.exists():
        hook_content = """#!/bin/sh
# Pre-commit hook for FarsiTranscribe
echo "Running pre-commit checks..."

# Run black formatting
black --check src/ tests/ main.py
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run 'make format' to fix."
    exit 1
fi

# Run flake8 linting
flake8 src/ tests/ main.py
if [ $? -ne 0 ]; then
    echo "❌ Linting check failed. Fix the issues above."
    exit 1
fi

echo "✅ Pre-commit checks passed"
"""
        try:
            pre_commit_hook.write_text(hook_content)
            pre_commit_hook.chmod(0o755)
            print("✅ Git pre-commit hook installed")
        except Exception as e:
            print(f"⚠️ Could not install git hook: {e}")
    
    return True


def print_next_steps():
    """Print next steps for the developer."""
    print("\n" + "="*60)
    print("🎉 Development environment setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   source venv/bin/activate  # On Unix/macOS")
    print("   venv\\Scripts\\activate     # On Windows")
    print("\n2. Run the transcription system:")
    print("   python main.py examples/audio/jalase\\ bi\\ va\\ zirsakht.m4a")
    print("\n3. Run tests:")
    print("   pytest tests/")
    print("\n4. Format code:")
    print("   make format")
    print("\n5. Check code quality:")
    print("   make lint")
    print("\n6. View available commands:")
    print("   make help")
    print("\nHappy coding! 🚀")


def main():
    """Main setup function."""
    print("🚀 Setting up FarsiTranscribe development environment...")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    run_tests()
    
    # Setup git hooks
    setup_git_hooks()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 
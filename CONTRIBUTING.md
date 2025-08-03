# Contributing to FarsiTranscribe

We love your input! We want to make contributing to FarsiTranscribe as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, track issues and feature requests, and accept pull requests.

## Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (follows PEP 8).
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/FarsiTranscribe.git
cd FarsiTranscribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Run linting
flake8 src/
black --check src/
mypy src/
```

## Testing

We use pytest for testing. Tests are located in the `tests/` directory.

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transcriber.py

# Run with coverage
pytest --cov=src
```

## Code Style

We follow PEP 8 guidelines:

- Use `black` for automatic formatting
- Use `flake8` for linting
- Use `mypy` for type checking
- Maximum line length: 88 characters (black default)

## Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/siryoos/FarsiTranscribe/issues).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We're open to feature requests! Please:

1. Check if the feature is already requested in issues
2. Describe the problem you're trying to solve
3. Describe the solution you'd like
4. Consider the project's scope and goals

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to contact the maintainer at siryoosa@gmail.com or open an issue for discussion.

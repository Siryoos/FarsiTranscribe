.PHONY: help install install-dev test test-coverage lint format clean build docs run-example

# Default target
help:
	@echo "FarsiTranscribe - Available commands:"
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  test-fast    - Run only fast tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo ""
	@echo "Development:"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  run-example  - Run basic usage example"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# Code Quality
lint:
	flake8 src/ tests/ main.py
	mypy src/ main.py

format:
	black src/ tests/ main.py examples/
	isort src/ tests/ main.py examples/

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python setup.py sdist bdist_wheel

docs:
	@echo "Documentation building not yet implemented"
	@echo "Use 'make help' for available commands"

# Examples
run-example:
	@echo "Running basic usage example..."
	@echo "Please update the audio file path in examples/basic_usage.py first"
	@python examples/basic_usage.py

# Quick development setup
dev-setup: install-dev format lint test-fast
	@echo "Development environment setup complete!"

# Full CI pipeline
ci: install-dev format lint test-coverage
	@echo "CI pipeline completed successfully!" 
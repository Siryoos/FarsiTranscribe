# Changelog

All notable changes to FarsiTranscribe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2024-12-15

### Added
- CI/CD pipeline with GitHub Actions
- Code coverage tracking with Codecov
- Pre-commit hooks for code quality
- Automated dependency updates with Dependabot
- Security scanning with Trivy
- Release automation workflow
- Comprehensive test matrix (Python 3.8-3.11, multiple OS)

### Changed
- Consolidated utility modules to eliminate code duplication
- Improved PEP 8 compliance across codebase
- Enhanced documentation with badges and status indicators

### Fixed
- Memory leaks in audio processing
- Unicode handling for Persian text
- Performance bottlenecks in large file processing

## [1.0.0] - 2024-11-01

### Added
- Initial release of FarsiTranscribe
- Persian-optimized Whisper model integration
- Modular architecture
- Multiple output formats
- Hook system for extensions
- GPU/CPU support

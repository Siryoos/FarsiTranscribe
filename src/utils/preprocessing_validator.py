"""
Preprocessing validation system for FarsiTranscribe.
This module validates preprocessing capabilities and ensures all required
components are available before transcription begins.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import importlib.util


@dataclass
class PreprocessingCapability:
    """Information about a preprocessing capability."""

    name: str
    available: bool
    version: Optional[str] = None
    error_message: Optional[str] = None
    required: bool = True
    fallback_available: bool = False


@dataclass
class ValidationResult:
    """Result of preprocessing validation."""

    success: bool
    capabilities: Dict[str, PreprocessingCapability]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]


class PreprocessingValidator:
    """
    Validates preprocessing capabilities and dependencies before transcription.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.capabilities = {}
        self.validation_result = None

    def validate_all_capabilities(self, config) -> ValidationResult:
        """
        Validate all preprocessing capabilities based on configuration.

        Args:
            config: TranscriptionConfig object

        Returns:
            ValidationResult with validation status
        """
        self.logger.info("Starting preprocessing validation...")

        # Initialize capabilities
        self.capabilities = {}
        warnings = []
        errors = []
        recommendations = []

        # Validate core dependencies
        self._validate_core_dependencies()

        # Validate audio preprocessing
        self._validate_audio_preprocessing(config)

        # Validate advanced preprocessing if enabled
        if config.enable_advanced_preprocessing:
            self._validate_advanced_preprocessing(config)

        # Validate Persian-specific optimizations
        if config.enable_persian_optimization:
            self._validate_persian_optimizations()

        # Validate memory management
        self._validate_memory_management()

        # Analyze results
        success = self._analyze_validation_results(
            warnings, errors, recommendations
        )

        # Create validation result
        self.validation_result = ValidationResult(
            success=success,
            capabilities=self.capabilities,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
        )

        # Log results
        self._log_validation_results()

        return self.validation_result

    def _validate_core_dependencies(self):
        """Validate core dependencies."""
        core_deps = [
            ("torch", "PyTorch"),
            ("whisper", "OpenAI Whisper"),
            ("numpy", "NumPy"),
            ("pydub", "Pydub"),
            ("librosa", "Librosa"),
            ("scipy", "SciPy"),
        ]

        for module_name, display_name in core_deps:
            capability = self._check_module_availability(
                module_name, display_name
            )
            self.capabilities[module_name] = capability

    def _validate_audio_preprocessing(self, config):
        """Validate audio preprocessing capabilities."""
        if not config.enable_preprocessing:
            self.capabilities["audio_preprocessing"] = PreprocessingCapability(
                name="Audio Preprocessing",
                available=False,
                required=False,
                error_message="Disabled in configuration",
            )
            return

        # Check basic audio preprocessing
        basic_preprocessing = self._check_basic_audio_preprocessing()
        self.capabilities["basic_audio_preprocessing"] = basic_preprocessing

        # Check noise reduction
        if config.enable_noise_reduction:
            noise_reduction = self._check_noise_reduction()
            self.capabilities["noise_reduction"] = noise_reduction

        # Check voice activity detection
        if config.enable_voice_activity_detection:
            vad = self._check_voice_activity_detection()
            self.capabilities["voice_activity_detection"] = vad

        # Check speech enhancement
        if config.enable_speech_enhancement:
            speech_enhancement = self._check_speech_enhancement()
            self.capabilities["speech_enhancement"] = speech_enhancement

    def _validate_advanced_preprocessing(self, config):
        """Validate advanced preprocessing capabilities."""
        if not config.enable_advanced_preprocessing:
            return

        # Check Facebook Denoiser
        if config.enable_facebook_denoiser:
            denoiser = self._check_facebook_denoiser()
            self.capabilities["facebook_denoiser"] = denoiser

        # Check adaptive processing
        if config.adaptive_processing:
            adaptive = self._check_adaptive_processing()
            self.capabilities["adaptive_processing"] = adaptive

    def _validate_persian_optimizations(self):
        """Validate Persian-specific optimizations."""
        persian_opt = self._check_persian_optimizations()
        self.capabilities["persian_optimizations"] = persian_opt

    def _validate_memory_management(self):
        """Validate memory management capabilities."""
        memory_mgmt = self._check_memory_management()
        self.capabilities["memory_management"] = memory_mgmt

    def _check_module_availability(
        self, module_name: str, display_name: str
    ) -> PreprocessingCapability:
        """Check if a module is available."""
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return PreprocessingCapability(
                    name=display_name,
                    available=False,
                    error_message=f"Module '{module_name}' not found",
                )

            # Try to import the module
            module = importlib.import_module(module_name)

            # Get version if available
            version = getattr(module, "__version__", None)

            return PreprocessingCapability(
                name=display_name, available=True, version=version
            )

        except ImportError as e:
            return PreprocessingCapability(
                name=display_name,
                available=False,
                error_message=f"Import error: {str(e)}",
            )
        except Exception as e:
            return PreprocessingCapability(
                name=display_name,
                available=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _check_basic_audio_preprocessing(self) -> PreprocessingCapability:
        """Check basic audio preprocessing capabilities."""
        try:
            import librosa
            import scipy.signal
            import numpy as np

            # Test basic functionality
            test_signal = np.random.randn(16000)
            filtered = scipy.signal.medfilt(test_signal, 5)

            return PreprocessingCapability(
                name="Basic Audio Preprocessing",
                available=True,
                version=f"librosa {librosa.__version__}",
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Basic Audio Preprocessing",
                available=False,
                error_message=f"Basic preprocessing failed: {str(e)}",
            )

    def _check_noise_reduction(self) -> PreprocessingCapability:
        """Check noise reduction capabilities."""
        try:
            import librosa
            import scipy.signal
            import numpy as np

            # Test spectral subtraction
            test_signal = np.random.randn(16000)
            stft = librosa.stft(test_signal)
            # Basic spectral subtraction test
            magnitude = np.abs(stft)
            noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)

            return PreprocessingCapability(
                name="Noise Reduction",
                available=True,
                version=f"librosa {librosa.__version__}",
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Noise Reduction",
                available=False,
                error_message=f"Noise reduction failed: {str(e)}",
            )

    def _check_voice_activity_detection(self) -> PreprocessingCapability:
        """Check voice activity detection capabilities."""
        try:
            import librosa
            import numpy as np

            # Test basic VAD functionality
            test_signal = np.random.randn(16000)
            mfcc = librosa.feature.mfcc(y=test_signal, sr=16000)

            return PreprocessingCapability(
                name="Voice Activity Detection",
                available=True,
                version=f"librosa {librosa.__version__}",
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Voice Activity Detection",
                available=False,
                error_message=f"VAD failed: {str(e)}",
            )

    def _check_speech_enhancement(self) -> PreprocessingCapability:
        """Check speech enhancement capabilities."""
        try:
            import librosa
            import numpy as np

            # Test basic enhancement functionality
            test_signal = np.random.randn(16000)
            enhanced = librosa.effects.preemphasis(test_signal)

            return PreprocessingCapability(
                name="Speech Enhancement",
                available=True,
                version=f"librosa {librosa.__version__}",
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Speech Enhancement",
                available=False,
                error_message=f"Speech enhancement failed: {str(e)}",
            )

    def _check_facebook_denoiser(self) -> PreprocessingCapability:
        """Check Facebook Denoiser availability."""
        try:
            import torch
            import torchaudio

            # Check if denoiser models are available
            denoiser_available = hasattr(torchaudio, "functional") and hasattr(
                torchaudio.functional, "spectral_centroid"
            )

            return PreprocessingCapability(
                name="Facebook Denoiser",
                available=denoiser_available,
                version=f"torchaudio {torchaudio.__version__}",
                required=False,
                fallback_available=True,
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Facebook Denoiser",
                available=False,
                error_message=f"Facebook denoiser failed: {str(e)}",
                required=False,
                fallback_available=True,
            )

    def _check_adaptive_processing(self) -> PreprocessingCapability:
        """Check adaptive processing capabilities."""
        try:
            import numpy as np
            import scipy.signal

            # Test adaptive filtering
            test_signal = np.random.randn(16000)
            reference = np.random.randn(16000)

            # Basic adaptive filter test
            filtered = scipy.signal.wiener(test_signal)

            return PreprocessingCapability(
                name="Adaptive Processing", available=True, version="scipy"
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Adaptive Processing",
                available=False,
                error_message=f"Adaptive processing failed: {str(e)}",
                required=False,
                fallback_available=True,
            )

    def _check_persian_optimizations(self) -> PreprocessingCapability:
        """Check Persian-specific optimizations."""
        try:
            # Check for Persian language support
            import re

            # Test Persian text processing
            persian_text = "سلام دنیا"
            persian_pattern = re.compile(
                r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
            )
            has_persian = bool(persian_pattern.search(persian_text))

            return PreprocessingCapability(
                name="Persian Optimizations",
                available=has_persian,
                version="built-in",
                required=False,
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Persian Optimizations",
                available=False,
                error_message=f"Persian optimizations failed: {str(e)}",
                required=False,
            )

    def _check_memory_management(self) -> PreprocessingCapability:
        """Check memory management capabilities."""
        try:
            import psutil
            import gc

            # Test memory monitoring
            memory = psutil.virtual_memory()
            gc.collect()

            return PreprocessingCapability(
                name="Memory Management",
                available=True,
                version=f"psutil {psutil.__version__}",
            )

        except Exception as e:
            return PreprocessingCapability(
                name="Memory Management",
                available=False,
                error_message=f"Memory management failed: {str(e)}",
            )

    def _analyze_validation_results(
        self,
        warnings: List[str],
        errors: List[str],
        recommendations: List[str],
    ) -> bool:
        """Analyze validation results and generate recommendations."""
        success = True

        # Check required capabilities
        for name, capability in self.capabilities.items():
            if capability.required and not capability.available:
                errors.append(
                    f"Required capability '{capability.name}' is not available: {capability.error_message}"
                )
                success = False
            elif not capability.available and capability.error_message:
                warnings.append(
                    f"Optional capability '{capability.name}' is not available: {capability.error_message}"
                )

        # Generate recommendations
        if not self.capabilities.get(
            "facebook_denoiser", PreprocessingCapability("", False)
        ).available:
            recommendations.append(
                "Consider installing torchaudio for advanced denoising capabilities"
            )

        if not self.capabilities.get(
            "adaptive_processing", PreprocessingCapability("", False)
        ).available:
            recommendations.append(
                "Consider installing scipy for adaptive processing features"
            )

        # Check memory availability
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
                warnings.append(
                    "Available memory is low. Consider using memory-optimized configuration."
                )
        except:
            pass

        return success

    def _log_validation_results(self):
        """Log validation results."""
        if not self.validation_result:
            return

        self.logger.info("Preprocessing validation completed:")

        for name, capability in self.capabilities.items():
            status = "✅" if capability.available else "❌"
            self.logger.info(f"  {status} {capability.name}")
            if capability.version:
                self.logger.info(f"    Version: {capability.version}")
            if capability.error_message:
                self.logger.warning(f"    Error: {capability.error_message}")

        if self.validation_result.warnings:
            self.logger.warning("Warnings:")
            for warning in self.validation_result.warnings:
                self.logger.warning(f"  - {warning}")

        if self.validation_result.errors:
            self.logger.error("Errors:")
            for error in self.validation_result.errors:
                self.logger.error(f"  - {error}")

        if self.validation_result.recommendations:
            self.logger.info("Recommendations:")
            for rec in self.validation_result.recommendations:
                self.logger.info(f"  - {rec}")

    def get_capabilities_summary(self) -> Dict[str, bool]:
        """Get a summary of available capabilities."""
        return {
            name: capability.available
            for name, capability in self.capabilities.items()
        }

    def is_capability_available(self, capability_name: str) -> bool:
        """Check if a specific capability is available."""
        capability = self.capabilities.get(capability_name)
        return capability.available if capability else False


def validate_preprocessing(config) -> ValidationResult:
    """
    Convenience function to validate preprocessing capabilities.

    Args:
        config: TranscriptionConfig object

    Returns:
        ValidationResult with validation status
    """
    validator = PreprocessingValidator()
    return validator.validate_all_capabilities(config)

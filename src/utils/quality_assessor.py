"""
Advanced Quality Assessment and Auto-Tuning System.
Evaluates transcription quality and automatically optimizes parameters.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter
import difflib
from sklearn.metrics import jaccard_score
import warnings

warnings.filterwarnings("ignore")

from ..core.config import TranscriptionConfig


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for transcription."""

    overall_score: float
    word_accuracy: float
    sentence_fluency: float
    punctuation_accuracy: float
    speaker_separation: float
    confidence_score: float
    noise_level: float
    repetition_score: float
    context_coherence: float
    details: Dict[str, Any]


class QualityAssessor:
    """Advanced quality assessment and auto-tuning system."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.quality_threshold = 0.95  # Target quality threshold
        self.optimization_iterations = 3

    def assess_transcription_quality(
        self, transcription: str, audio_metadata: Dict[str, Any]
    ) -> QualityMetrics:
        """Comprehensive quality assessment of transcription."""
        try:
            self.logger.info("Assessing transcription quality...")

            # Calculate various quality metrics
            word_accuracy = self._calculate_word_accuracy(transcription)
            sentence_fluency = self._calculate_sentence_fluency(transcription)
            punctuation_accuracy = self._calculate_punctuation_accuracy(
                transcription
            )
            speaker_separation = self._calculate_speaker_separation(
                transcription
            )
            confidence_score = self._calculate_confidence_score(audio_metadata)
            noise_level = self._calculate_noise_level(audio_metadata)
            repetition_score = self._calculate_repetition_score(transcription)
            context_coherence = self._calculate_context_coherence(
                transcription
            )

            # Calculate overall score (weighted average)
            overall_score = (
                word_accuracy * 0.25
                + sentence_fluency * 0.20
                + punctuation_accuracy * 0.15
                + speaker_separation * 0.10
                + confidence_score * 0.15
                + (1.0 - noise_level) * 0.10
                + (1.0 - repetition_score) * 0.05
            )

            details = {
                "word_accuracy": word_accuracy,
                "sentence_fluency": sentence_fluency,
                "punctuation_accuracy": punctuation_accuracy,
                "speaker_separation": speaker_separation,
                "confidence_score": confidence_score,
                "noise_level": noise_level,
                "repetition_score": repetition_score,
                "context_coherence": context_coherence,
                "audio_metadata": audio_metadata,
            }

            metrics = QualityMetrics(
                overall_score=overall_score,
                word_accuracy=word_accuracy,
                sentence_fluency=sentence_fluency,
                punctuation_accuracy=punctuation_accuracy,
                speaker_separation=speaker_separation,
                confidence_score=confidence_score,
                noise_level=noise_level,
                repetition_score=repetition_score,
                context_coherence=context_coherence,
                details=details,
            )

            self.logger.info(
                f"Quality assessment complete. Overall score: {overall_score:.3f}"
            )
            return metrics

        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            # Return default metrics
            return QualityMetrics(
                overall_score=0.5,
                word_accuracy=0.5,
                sentence_fluency=0.5,
                punctuation_accuracy=0.5,
                speaker_separation=0.5,
                confidence_score=0.5,
                noise_level=0.5,
                repetition_score=0.5,
                context_coherence=0.5,
                details={"error": str(e)},
            )

    def _calculate_word_accuracy(self, transcription: str) -> float:
        """Calculate word-level accuracy based on Persian language patterns."""
        if not transcription.strip():
            return 0.0

        words = transcription.split()
        if not words:
            return 0.0

        # Persian word patterns
        persian_pattern = re.compile(
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+"
        )

        valid_words = 0
        total_words = len(words)

        for word in words:
            # Check if word contains Persian characters
            if persian_pattern.search(word):
                # Check for common Persian word patterns
                if len(word) >= 2 and not re.match(
                    r"^[^\u0600-\u06FF]*$", word
                ):
                    valid_words += 1
                else:
                    # Check for valid Persian word structure
                    if self._is_valid_persian_word(word):
                        valid_words += 1

        return valid_words / total_words if total_words > 0 else 0.0

    def _is_valid_persian_word(self, word: str) -> bool:
        """Check if a word follows valid Persian word patterns."""
        # Remove punctuation
        word = re.sub(
            r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]",
            "",
            word,
        )

        if len(word) < 2:
            return False

        # Check for common Persian word endings
        valid_endings = ["ی", "ان", "ها", "ات", "ین", "ون", "ها", "ها"]
        if any(word.endswith(ending) for ending in valid_endings):
            return True

        # Check for common Persian prefixes
        valid_prefixes = ["می", "نمی", "بی", "با", "در", "از", "به", "که"]
        if any(word.startswith(prefix) for prefix in valid_prefixes):
            return True

        # Basic length and character check
        return len(word) >= 2 and re.search(r"[\u0600-\u06FF]", word)

    def _calculate_sentence_fluency(self, transcription: str) -> float:
        """Calculate sentence fluency and naturalness."""
        if not transcription.strip():
            return 0.0

        sentences = re.split(r"[،؛؟!\.]", transcription)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        fluency_scores = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) < 3:
                fluency_scores.append(0.3)  # Very short sentences
                continue

            # Check sentence structure
            score = 1.0

            # Penalize very long sentences
            if len(words) > 25:
                score *= 0.7

            # Check for proper word spacing
            if re.search(r"\s{2,}", sentence):
                score *= 0.9

            # Check for proper Persian sentence endings
            if not re.search(r"[،؛؟!\.]$", sentence):
                score *= 0.8

            fluency_scores.append(score)

        return np.mean(fluency_scores) if fluency_scores else 0.0

    def _calculate_punctuation_accuracy(self, transcription: str) -> float:
        """Calculate punctuation accuracy."""
        if not transcription.strip():
            return 0.0

        # Count punctuation marks
        punctuation_marks = len(re.findall(r"[،؛؟!\.]", transcription))
        sentences = len(re.split(r"[،؛؟!\.]", transcription))

        if sentences <= 1:
            return 0.5  # Neutral score for single sentence

        # Expected punctuation ratio (roughly 1.5 marks per sentence)
        expected_ratio = 1.5
        actual_ratio = punctuation_marks / sentences

        # Calculate accuracy based on ratio
        if actual_ratio == 0:
            return 0.0
        elif abs(actual_ratio - expected_ratio) <= 0.5:
            return 1.0
        else:
            return max(
                0.0, 1.0 - abs(actual_ratio - expected_ratio) / expected_ratio
            )

    def _calculate_speaker_separation(self, transcription: str) -> float:
        """Calculate speaker separation quality."""
        # This is a simplified version - would need actual speaker labels
        # For now, estimate based on conversation patterns

        # Look for conversation indicators
        conversation_indicators = [
            r"گفت",
            r"گفتند",
            r"پرسید",
            r"پاسخ داد",
            r"اضافه کرد",
            r"گفت:",
            r"گفتند:",
            r"پرسید:",
            r"پاسخ داد:",
            r"اضافه کرد:",
        ]

        indicator_count = sum(
            len(re.findall(pattern, transcription))
            for pattern in conversation_indicators
        )

        # Normalize based on text length
        words = transcription.split()
        if len(words) < 10:
            return 0.5  # Neutral for short texts

        ratio = indicator_count / len(words)

        # Optimal ratio is around 0.01-0.02
        if 0.005 <= ratio <= 0.03:
            return 1.0
        elif ratio > 0.03:
            return max(0.0, 1.0 - (ratio - 0.03) / 0.02)
        else:
            return ratio / 0.005

    def _calculate_confidence_score(
        self, audio_metadata: Dict[str, Any]
    ) -> float:
        """Calculate confidence score from audio metadata."""
        # Extract confidence from metadata
        ensemble_confidence = audio_metadata.get("ensemble_confidence", 0.5)
        individual_confidences = audio_metadata.get(
            "individual_confidences", {}
        )

        if individual_confidences:
            # Calculate weighted average confidence
            weights = audio_metadata.get("weights", {})
            total_weight = sum(weights.values())

            if total_weight > 0:
                weighted_confidence = (
                    sum(
                        individual_confidences.get(name, 0.5)
                        * weights.get(name, 0.1)
                        for name in individual_confidences
                    )
                    / total_weight
                )
                return weighted_confidence

        return ensemble_confidence

    def _calculate_noise_level(self, audio_metadata: Dict[str, Any]) -> float:
        """Calculate noise level from audio metadata."""
        # Extract noise-related metrics
        signal_to_noise = audio_metadata.get("signal_to_noise_ratio", 20.0)
        audio_quality = audio_metadata.get("audio_quality_score", 0.8)

        # Convert SNR to noise level (0-1, where 0 is no noise)
        # Typical SNR: 20dB = good, 10dB = poor, 30dB = excellent
        if signal_to_noise >= 25:
            noise_level = 0.1
        elif signal_to_noise >= 15:
            noise_level = 0.3
        elif signal_to_noise >= 10:
            noise_level = 0.6
        else:
            noise_level = 0.9

        # Combine with audio quality
        combined_noise = (noise_level + (1.0 - audio_quality)) / 2

        return min(1.0, max(0.0, combined_noise))

    def _calculate_repetition_score(self, transcription: str) -> float:
        """Calculate repetition level (lower is better)."""
        if not transcription.strip():
            return 0.0

        words = transcription.split()
        if len(words) < 5:
            return 0.0

        # Count word repetitions
        word_counts = Counter(words)
        total_words = len(words)

        # Calculate repetition ratio
        repetition_ratio = (
            sum(count - 1 for count in word_counts.values()) / total_words
        )

        # Normalize to 0-1 scale
        return min(
            1.0, repetition_ratio * 2
        )  # Scale factor for better sensitivity

    def _calculate_context_coherence(self, transcription: str) -> float:
        """Calculate context coherence and logical flow."""
        if not transcription.strip():
            return 0.0

        sentences = re.split(r"[،؛؟!\.]", transcription)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.5  # Neutral for single sentence

        # Check for logical connectors
        connectors = [
            r"و",
            r"اما",
            r"ولی",
            r"اگر",
            r"چون",
            r"زیرا",
            r"بنابراین",
            r"همچنین",
            r"علاوه بر",
            r"در نتیجه",
            r"به عبارت دیگر",
        ]

        connector_count = sum(
            len(re.findall(connector, transcription))
            for connector in connectors
        )

        # Normalize by sentence count
        connector_ratio = connector_count / len(sentences)

        # Optimal ratio is around 0.5-1.0 connectors per sentence
        if 0.3 <= connector_ratio <= 1.2:
            return 1.0
        elif connector_ratio > 1.2:
            return max(0.0, 1.0 - (connector_ratio - 1.2) / 0.8)
        else:
            return connector_ratio / 0.3

    def auto_tune_parameters(
        self,
        current_metrics: QualityMetrics,
        current_config: TranscriptionConfig,
    ) -> TranscriptionConfig:
        """Automatically tune parameters based on quality metrics."""
        self.logger.info("Auto-tuning parameters for quality improvement...")

        optimized_config = current_config.clone()

        # Adjust parameters based on quality metrics
        if current_metrics.word_accuracy < 0.8:
            # Improve word accuracy
            optimized_config.temperature = 0.0  # More deterministic
            optimized_config.condition_on_previous_text = True
            optimized_config.chunk_duration_ms = min(
                30000, current_config.chunk_duration_ms + 5000
            )

        if current_metrics.sentence_fluency < 0.7:
            # Improve sentence fluency
            optimized_config.overlap_ms = min(
                1000, current_config.overlap_ms + 200
            )
            optimized_config.enable_enhanced_preprocessing = True
            optimized_config.enable_text_postprocessing = True

        if current_metrics.punctuation_accuracy < 0.6:
            # Improve punctuation
            optimized_config.enable_text_postprocessing = True
            optimized_config.temperature = 0.0

        if current_metrics.noise_level > 0.5:
            # Reduce noise impact
            optimized_config.noise_threshold = max(
                0.2, current_config.noise_threshold - 0.1
            )
            optimized_config.enable_enhanced_preprocessing = True

        if current_metrics.repetition_score > 0.3:
            # Reduce repetition
            optimized_config.repetition_threshold = min(
                0.9, current_config.repetition_threshold + 0.05
            )
            optimized_config.max_word_repetition = max(
                1, current_config.max_word_repetition - 1
            )

        if current_metrics.confidence_score < 0.7:
            # Improve confidence
            optimized_config.min_chunk_confidence = max(
                0.5, current_config.min_chunk_confidence - 0.1
            )
            optimized_config.temperature = 0.0

        self.logger.info("Parameter auto-tuning complete")
        return optimized_config

    def get_quality_report(self, metrics: QualityMetrics) -> str:
        """Generate a detailed quality report."""
        report = f"""
=== TRANSCRIPTION QUALITY REPORT ===
Overall Quality Score: {metrics.overall_score:.1%}

DETAILED METRICS:
• Word Accuracy: {metrics.word_accuracy:.1%}
• Sentence Fluency: {metrics.sentence_fluency:.1%}
• Punctuation Accuracy: {metrics.punctuation_accuracy:.1%}
• Speaker Separation: {metrics.speaker_separation:.1%}
• Confidence Score: {metrics.confidence_score:.1%}
• Noise Level: {metrics.noise_level:.1%}
• Repetition Score: {metrics.repetition_score:.1%}
• Context Coherence: {metrics.context_coherence:.1%}

QUALITY ASSESSMENT:
"""

        if metrics.overall_score >= 0.95:
            report += "✅ EXCELLENT QUALITY - Target achieved!\n"
        elif metrics.overall_score >= 0.85:
            report += "🟡 GOOD QUALITY - Minor improvements possible\n"
        elif metrics.overall_score >= 0.70:
            report += "🟠 MODERATE QUALITY - Significant improvements needed\n"
        else:
            report += "🔴 POOR QUALITY - Major improvements required\n"

        # Add specific recommendations
        recommendations = []

        if metrics.word_accuracy < 0.8:
            recommendations.append(
                "• Improve word accuracy: Consider using larger models or better audio preprocessing"
            )

        if metrics.sentence_fluency < 0.7:
            recommendations.append(
                "• Improve sentence fluency: Increase overlap between chunks and enhance text post-processing"
            )

        if metrics.punctuation_accuracy < 0.6:
            recommendations.append(
                "• Improve punctuation: Enable text post-processing and use deterministic generation"
            )

        if metrics.noise_level > 0.5:
            recommendations.append(
                "• Reduce noise: Enhance audio preprocessing and adjust noise thresholds"
            )

        if metrics.repetition_score > 0.3:
            recommendations.append(
                "• Reduce repetition: Adjust repetition thresholds and improve deduplication"
            )

        if recommendations:
            report += "\nRECOMMENDATIONS:\n" + "\n".join(recommendations)

        return report


def create_quality_assessor(config: TranscriptionConfig) -> QualityAssessor:
    """Create quality assessor instance."""
    return QualityAssessor(config)

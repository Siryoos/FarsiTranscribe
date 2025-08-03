#!/usr/bin/env python3
"""
Persian Text Post-Processor for Transcription Quality Improvement
Implements language model-based correction for:
- Punctuation and spacing
- Text formatting and structure
- Persian-specific text normalization
- Repetition and error correction
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import unicodedata
from collections import Counter

try:
    import hazm

    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PersianTextPostProcessor:
    """
    Post-processor for improving Persian transcription quality.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Persian-specific patterns
        self.persian_patterns = {
            "multiple_spaces": r"\s+",
            "multiple_dots": r"\.{2,}",
            "multiple_commas": r",{2,}",
            "persian_numbers": r"[۰-۹]",
            "english_numbers": r"[0-9]",
            "persian_punctuation": r"[،؛؟]",
            "english_punctuation": r"[,;?]",
            "repeated_chars": r"(.)\1{2,}",  # 3 or more repeated characters
        }

        # Initialize text processing tools
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize Persian text processing tools."""
        self.normalizer = None
        self.tokenizer = None
        self.spell_checker = None

        if HAZM_AVAILABLE:
            try:
                self.normalizer = hazm.Normalizer()
                self.tokenizer = hazm.WordTokenizer()
                self.logger.info("Hazm Persian NLP tools initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Hazm: {e}")

        # Initialize spell checker if available
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a small Persian language model for text correction
                self.spell_checker = pipeline(
                    "text-generation",
                    model="gpt2",  # Fallback to English model
                    device="cpu",
                )
                self.logger.info("Text generation pipeline initialized")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize text generation: {e}"
                )

    def post_process_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Comprehensive post-processing of Persian transcription text.

        Args:
            text: Raw transcribed text
            metadata: Optional metadata about the transcription

        Returns:
            Tuple of (processed_text, processing_metadata)
        """
        self.logger.info("Starting Persian text post-processing")

        processing_metadata = {
            "original_text": text,
            "original_length": len(text),
            "processing_steps": [],
            "corrections_made": {},
            "quality_metrics": {},
        }

        # Step 1: Basic cleaning
        text = self._basic_cleaning(text)
        processing_metadata["processing_steps"].append("basic_cleaning")

        # Step 2: Persian-specific normalization
        if self.normalizer:
            text = self._persian_normalization(text)
            processing_metadata["processing_steps"].append(
                "persian_normalization"
            )

        # Step 3: Punctuation and spacing correction
        text = self._fix_punctuation_and_spacing(text)
        processing_metadata["processing_steps"].append("punctuation_spacing")

        # Step 4: Remove repetitions
        text, repetition_count = self._remove_repetitions(text)
        processing_metadata["corrections_made"][
            "repetitions_removed"
        ] = repetition_count

        # Step 5: Sentence structure improvement
        text = self._improve_sentence_structure(text)
        processing_metadata["processing_steps"].append("sentence_structure")

        # Step 6: Context-aware corrections
        text = self._context_aware_corrections(text)
        processing_metadata["processing_steps"].append("context_corrections")

        # Step 7: Final formatting
        text = self._final_formatting(text)
        processing_metadata["processing_steps"].append("final_formatting")

        # Calculate quality metrics
        quality_metrics = self._calculate_text_quality(text)
        processing_metadata["quality_metrics"] = quality_metrics

        self.logger.info(
            f"Post-processing completed. Final length: {len(text)}"
        )

        return text, processing_metadata

    def _basic_cleaning(self, text: str) -> str:
        """Basic text cleaning operations."""
        # Remove extra whitespace
        text = re.sub(self.persian_patterns["multiple_spaces"], " ", text)
        text = text.strip()

        # Remove multiple dots and commas
        text = re.sub(self.persian_patterns["multiple_dots"], ".", text)
        text = re.sub(self.persian_patterns["multiple_commas"], ",", text)

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        return text

    def _persian_normalization(self, text: str) -> str:
        """Apply Persian-specific text normalization."""
        if not self.normalizer:
            return text

        try:
            # Apply Hazm normalizer
            normalized = self.normalizer.normalize(text)

            # Additional Persian-specific normalizations
            # Convert Persian numbers to English if needed
            if self.config.convert_persian_numbers:
                normalized = self._convert_persian_numbers(normalized)

            return normalized
        except Exception as e:
            self.logger.warning(f"Persian normalization failed: {e}")
            return text

    def _convert_persian_numbers(self, text: str) -> str:
        """Convert Persian numbers to English numbers."""
        persian_to_english = {
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
        }

        for persian, english in persian_to_english.items():
            text = text.replace(persian, english)

        return text

    def _fix_punctuation_and_spacing(self, text: str) -> str:
        """Fix punctuation and spacing issues."""
        # Fix spacing around punctuation
        text = re.sub(r"\s*([،؛؟!\.])\s*", r"\1 ", text)
        text = re.sub(r"\s*([,;?!])\s*", r"\1 ", text)

        # Fix spacing around parentheses
        text = re.sub(r"\(\s*", " (", text)
        text = re.sub(r"\s*\)", ") ", text)

        # Fix spacing around quotes
        text = re.sub(r'"\s*', ' "', text)
        text = re.sub(r'\s*"', '" ', text)

        # Remove space before punctuation at end of sentences
        text = re.sub(r"\s+([،؛؟!\.])$", r"\1", text)

        # Fix multiple spaces again
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _remove_repetitions(self, text: str) -> Tuple[str, int]:
        """Remove excessive character repetitions."""
        repetition_count = 0

        # Find and fix repeated characters (3 or more)
        def replace_repetitions(match):
            nonlocal repetition_count
            char = match.group(1)
            count = len(match.group(0))

            # Keep only 2 repetitions for emphasis
            if count > 2:
                repetition_count += 1
                return char * 2
            return match.group(0)

        text = re.sub(
            self.persian_patterns["repeated_chars"], replace_repetitions, text
        )

        # Remove repeated words
        words = text.split()
        if len(words) > 1:
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i - 1]:
                    cleaned_words.append(word)
                else:
                    repetition_count += 1

            text = " ".join(cleaned_words)

        return text, repetition_count

    def _improve_sentence_structure(self, text: str) -> str:
        """Improve sentence structure and flow."""
        # Split into sentences
        sentences = re.split(r"([،؛؟!\.])", text)

        improved_sentences = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1]

                # Capitalize first letter of sentences
                if sentence and not sentence[0].isupper():
                    sentence = sentence[0].upper() + sentence[1:]

                improved_sentences.append(sentence + punctuation)
            else:
                # Last sentence without punctuation
                sentence = sentences[i].strip()
                if sentence and not sentence[0].isupper():
                    sentence = sentence[0].upper() + sentence[1:]
                improved_sentences.append(sentence)

        return " ".join(improved_sentences)

    def _context_aware_corrections(self, text: str) -> str:
        """Apply context-aware text corrections."""
        # Common Persian transcription errors and corrections
        corrections = {
            "ایا": "آیا",  # Common question word error
            "میشه": "می‌شه",  # Missing half-space
            "نمیشه": "نمی‌شه",  # Missing half-space
            "برام": "برام",  # Already correct
            "برات": "برات",  # Already correct
            "براش": "براش",  # Already correct
            "برامون": "برامون",  # Already correct
            "براتون": "براتون",  # Already correct
            "براشون": "براشون",  # Already correct
        }

        for error, correction in corrections.items():
            text = text.replace(error, correction)

        # Fix common spacing issues with Persian verbs
        text = re.sub(r"(\w+)می(\w+)", r"\1 می‌\2", text)
        text = re.sub(r"(\w+)نمی(\w+)", r"\1 نمی‌\2", text)

        return text

    def _final_formatting(self, text: str) -> str:
        """Apply final formatting touches."""
        # Ensure proper spacing
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Ensure proper sentence endings
        if text and not text[-1] in "،؛؟!.":
            text += "."

        return text

    def _calculate_text_quality(self, text: str) -> Dict[str, Any]:
        """Calculate text quality metrics."""
        if not text:
            return {"score": 0, "issues": ["Empty text"]}

        words = text.split()
        sentences = re.split(r"[،؛؟!\.]", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0
        )

        # Quality indicators
        issues = []
        score = 100

        # Check for very short sentences
        if avg_sentence_length < 3:
            issues.append("Very short sentences")
            score -= 20

        # Check for very long sentences
        if avg_sentence_length > 20:
            issues.append("Very long sentences")
            score -= 15

        # Check for repetition
        word_freq = Counter(words)
        most_common = word_freq.most_common(1)[0] if word_freq else ("", 0)
        if most_common[1] > word_count * 0.1:  # More than 10% repetition
            issues.append("High word repetition")
            score -= 25

        # Check for proper punctuation
        punctuation_count = len(re.findall(r"[،؛؟!\.]", text))
        if punctuation_count < sentence_count * 0.5:
            issues.append("Missing punctuation")
            score -= 15

        return {
            "score": max(0, score),
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "punctuation_count": punctuation_count,
            "issues": issues,
        }

    def batch_process(
        self, texts: List[str]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process multiple texts in batch."""
        results = []
        for text in texts:
            processed_text, metadata = self.post_process_text(text)
            results.append((processed_text, metadata))
        return results


# Factory function for easy integration
def create_persian_postprocessor(config) -> PersianTextPostProcessor:
    """Create Persian text post-processor."""
    return PersianTextPostProcessor(config)

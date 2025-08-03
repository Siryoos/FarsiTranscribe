"""
Advanced repetition detection and removal utilities.
"""

import re
from typing import List
from difflib import SequenceMatcher
from ..core.config import TranscriptionConfig


class RepetitionDetector:
    """Advanced repetition detection and removal utility."""

    @staticmethod
    def detect_word_repetition(text: str, max_repetitions: int = 3) -> str:
        """Remove excessive word repetitions from text."""
        if not text:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        cleaned_words = []
        i = 0

        while i < len(words):
            current_word = words[i]
            cleaned_words.append(current_word)

            # Count consecutive repetitions
            repetition_count = 1
            j = i + 1

            while j < len(words) and words[j] == current_word:
                repetition_count += 1
                j += 1

            # Skip excessive repetitions
            if repetition_count > max_repetitions:
                i = j  # Skip all repetitions beyond the limit
            else:
                # Add remaining valid repetitions
                for _ in range(min(repetition_count - 1, max_repetitions - 1)):
                    if j > i + 1:
                        cleaned_words.append(current_word)
                i = j

        return " ".join(cleaned_words)

    @staticmethod
    def detect_phrase_repetition(text: str, min_phrase_length: int = 3) -> str:
        """Remove repetitive phrases from text."""
        if not text:
            return text

        words = text.split()
        if len(words) < min_phrase_length * 2:
            return text

        cleaned_words = []
        i = 0

        while i < len(words):
            # Try different phrase lengths
            found_repetition = False

            for phrase_len in range(
                min_phrase_length, min(10, len(words) - i + 1)
            ):
                if i + phrase_len * 2 > len(words):
                    break

                phrase1 = words[i : i + phrase_len]
                phrase2 = words[i + phrase_len : i + phrase_len * 2]

                if phrase1 == phrase2:
                    # Found repetition, add only one instance
                    cleaned_words.extend(phrase1)

                    # Skip all subsequent repetitions of this phrase
                    j = i + phrase_len * 2
                    while (
                        j + phrase_len <= len(words)
                        and words[j : j + phrase_len] == phrase1
                    ):
                        j += phrase_len

                    i = j
                    found_repetition = True
                    break

            if not found_repetition:
                cleaned_words.append(words[i])
                i += 1

        return " ".join(cleaned_words)

    @staticmethod
    def similarity_ratio(text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    @classmethod
    def clean_repetitive_text(
        cls, text: str, config: TranscriptionConfig
    ) -> str:
        """Comprehensive repetition cleaning."""
        if not text:
            return text

        # Step 1: Remove excessive word repetitions
        cleaned = cls.detect_word_repetition(text, config.max_word_repetition)

        # Step 2: Remove phrase repetitions
        cleaned = cls.detect_phrase_repetition(cleaned)

        # Step 3: Final cleanup
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    @staticmethod
    def find_overlapping_sequences(
        text1: str, text2: str, min_overlap: int = 5
    ) -> tuple:
        """Find overlapping sequences between two texts."""
        words1 = text1.split()
        words2 = text2.split()

        if len(words1) < min_overlap or len(words2) < min_overlap:
            return 0, 0, 0

        # Find the longest common subsequence
        max_overlap = 0
        overlap_start1 = 0
        overlap_start2 = 0

        for i in range(len(words1) - min_overlap + 1):
            for j in range(len(words2) - min_overlap + 1):
                overlap_length = 0
                while (
                    i + overlap_length < len(words1)
                    and j + overlap_length < len(words2)
                    and words1[i + overlap_length]
                    == words2[j + overlap_length]
                ):
                    overlap_length += 1

                if (
                    overlap_length > max_overlap
                    and overlap_length >= min_overlap
                ):
                    max_overlap = overlap_length
                    overlap_start1 = i
                    overlap_start2 = j

        return max_overlap, overlap_start1, overlap_start2

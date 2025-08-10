"""
Persian text normalizer for Farsi voice transcription fine-tuning.
Handles text cleaning, normalization, and preparation for Whisper models.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hazm
from hazm import Normalizer, word_tokenize
import unicodedata


@dataclass
class TextNormalizationConfig:
    """Configuration for text normalization."""
    
    # Basic normalization
    normalize_whitespace: bool = True
    remove_extra_spaces: bool = True
    normalize_punctuation: bool = True
    
    # Persian-specific settings
    normalize_persian_numbers: bool = True
    normalize_persian_text: bool = True
    remove_diacritics: bool = True
    normalize_arabic_chars: bool = True
    
    # Text cleaning
    remove_special_chars: bool = False
    remove_emojis: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    
    # Case and formatting
    lowercase: bool = False  # Keep Persian case as is
    remove_brackets: bool = False
    
    # Quality settings
    min_text_length: int = 3
    max_text_length: int = 1000


class PersianTextNormalizer:
    """Persian text normalizer optimized for Farsi voice transcription."""
    
    def __init__(self, config: TextNormalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Hazm normalizer
        try:
            self.hazm_normalizer = Normalizer()
        except Exception as e:
            self.logger.warning(f"Failed to initialize Hazm normalizer: {e}")
            self.hazm_normalizer = None
        
        # Persian number mapping
        self.persian_to_english_numbers = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'
        }
        
        # Arabic character mapping to Persian
        self.arabic_to_persian = {
            'ي': 'ی', 'ك': 'ک', 'ة': 'ه', 'ؤ': 'و', 'إ': 'ا',
            'أ': 'ا', 'آ': 'آ', 'ئ': 'ی', 'ء': 'ه'
        }
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.emoji_pattern = re.compile(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        self.extra_spaces_pattern = re.compile(r'\s+')
        self.bracket_pattern = re.compile(r'[\(\)\[\]\{\}]')
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Persian text for fine-tuning.
        
        Args:
            text: Input Persian text
            
        Returns:
            Normalized Persian text
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Apply normalization pipeline
            normalized_text = text
            
            # Basic cleaning
            if self.config.remove_urls:
                normalized_text = self._remove_urls(normalized_text)
            
            if self.config.remove_emails:
                normalized_text = self._remove_emails(normalized_text)
            
            if self.config.remove_emojis:
                normalized_text = self._remove_emojis(normalized_text)
            
            # Persian-specific normalization
            if self.config.normalize_persian_text:
                normalized_text = self._normalize_persian_text(normalized_text)
            
            if self.config.normalize_persian_numbers:
                normalized_text = self._normalize_persian_numbers(normalized_text)
            
            if self.config.normalize_arabic_chars:
                normalized_text = self._normalize_arabic_chars(normalized_text)
            
            # Text formatting
            if self.config.normalize_whitespace:
                normalized_text = self._normalize_whitespace(normalized_text)
            
            if self.config.remove_extra_spaces:
                normalized_text = self._remove_extra_spaces(normalized_text)
            
            if self.config.remove_diacritics:
                normalized_text = self._remove_diacritics(normalized_text)
            
            if self.config.remove_brackets:
                normalized_text = self._remove_brackets(normalized_text)
            
            # Final validation
            normalized_text = self._validate_text(normalized_text)
            
            return normalized_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error normalizing text: {str(e)}")
            return text
    
    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of normalized texts
        """
        normalized_texts = []
        
        for i, text in enumerate(texts):
            try:
                normalized_text = self.normalize_text(text)
                if normalized_text:  # Only add non-empty texts
                    normalized_texts.append(normalized_text)
            except Exception as e:
                self.logger.warning(f"Failed to normalize text {i}: {str(e)}")
                continue
        
        return normalized_texts
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis and special characters, keeping Persian text."""
        # Keep Persian/Arabic characters, remove others
        return self.emoji_pattern.sub('', text)
    
    def _normalize_persian_text(self, text: str) -> str:
        """Apply Hazm-based Persian text normalization."""
        if self.hazm_normalizer:
            try:
                # Apply Hazm normalization
                normalized = self.hazm_normalizer.normalize(text)
                return normalized
            except Exception as e:
                self.logger.warning(f"Hazm normalization failed: {e}")
                return text
        return text
    
    def _normalize_persian_numbers(self, text: str) -> str:
        """Convert Persian numbers to English numbers."""
        for persian_num, english_num in self.persian_to_english_numbers.items():
            text = text.replace(persian_num, english_num)
        return text
    
    def _normalize_arabic_chars(self, text: str) -> str:
        """Convert Arabic characters to Persian equivalents."""
        for arabic_char, persian_char in self.arabic_to_persian.items():
            text = text.replace(arabic_char, persian_char)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace characters with standard space
        text = re.sub(r'[\u200B-\u200D\uFEFF]', ' ', text)
        return text
    
    def _remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces and normalize spacing."""
        # Replace multiple spaces with single space
        text = self.extra_spaces_pattern.sub(' ', text)
        return text
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks from text."""
        # Remove combining diacritical marks
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'[\u0300-\u036f]', '', text)
        return text
    
    def _remove_brackets(self, text: str) -> str:
        """Remove brackets and parentheses."""
        return self.bracket_pattern.sub('', text)
    
    def _validate_text(self, text: str) -> str:
        """Validate and clean text according to quality settings."""
        # Check length constraints
        if len(text) < self.config.min_text_length:
            return ""
        
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        return text
    
    def create_normalization_report(self, input_texts: List[str], output_texts: List[str]) -> Dict:
        """Create a report of the normalization results."""
        report = {
            "total_texts": len(input_texts),
            "successful_normalizations": len(output_texts),
            "failed_normalizations": len(input_texts) - len(output_texts),
            "success_rate": len(output_texts) / len(input_texts) if input_texts else 0,
            "text_length_stats": {},
            "normalization_changes": {}
        }
        
        # Analyze text length changes
        if input_texts and output_texts:
            input_lengths = [len(text) for text in input_texts]
            output_lengths = [len(text) for text in output_texts]
            
            report["text_length_stats"] = {
                "input_avg_length": sum(input_lengths) / len(input_lengths),
                "output_avg_length": sum(output_lengths) / len(output_lengths),
                "input_min_length": min(input_lengths),
                "output_min_length": min(output_lengths),
                "input_max_length": max(input_lengths),
                "output_max_length": max(output_lengths)
            }
        
        return report
    
    def get_text_statistics(self, text: str) -> Dict:
        """Get detailed statistics about a text."""
        stats = {
            "length": len(text),
            "word_count": len(text.split()),
            "persian_char_count": len(re.findall(r'[\u0600-\u06FF]', text)),
            "english_char_count": len(re.findall(r'[a-zA-Z]', text)),
            "number_count": len(re.findall(r'\d', text)),
            "punctuation_count": len(re.findall(r'[^\w\s]', text)),
            "has_persian_numbers": any(char in self.persian_to_english_numbers for char in text),
            "has_arabic_chars": any(char in self.arabic_to_persian for char in text)
        }
        
        return stats

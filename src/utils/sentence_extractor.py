"""
Enhanced utility class for extracting and processing sentences.
"""

import re
from typing import List
from .repetition_detector import RepetitionDetector
from ..core.config import TranscriptionConfig


class SentenceExtractor:
    """Enhanced utility class for extracting and processing sentences."""
    
    @staticmethod
    def extract_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """Extract and clean sentences from transcription text."""
        if not text or not text.strip():
            return []
        
        # Clean the text first
        cleaned_text = RepetitionDetector.clean_repetitive_text(
            text.strip(), 
            TranscriptionConfig()  # Use default config for cleaning
        )
        
        # Enhanced sentence splitting for Persian text
        sentence_pattern = r'[.!?؟]+(?:\s+|$)'
        sentences = re.split(sentence_pattern, cleaned_text)
        
        valid_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # Minimum meaningful length
                valid_sentences.append(sentence)
        
        return valid_sentences[:max_sentences]
    
    @staticmethod
    def format_sentence_preview(sentences: List[str], part_number: int) -> str:
        """Format sentences for preview display."""
        if not sentences:
            return f"Part {part_number}: [No meaningful content]"
        
        preview_lines = [f"Part {part_number} Preview:"]
        for i, sentence in enumerate(sentences, 1):
            # Truncate very long sentences for preview
            display_sentence = sentence[:100] + "..." if len(sentence) > 100 else sentence
            preview_lines.append(f"  {i}. {display_sentence}")
        
        return "\n".join(preview_lines)
    
    @staticmethod
    def split_into_paragraphs(text: str, min_paragraph_length: int = 50) -> List[str]:
        """Split text into meaningful paragraphs."""
        if not text:
            return []
        
        # Split by double newlines or periods followed by newlines
        paragraphs = re.split(r'\n\s*\n|\.\s*\n', text)
        
        valid_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) >= min_paragraph_length:
                valid_paragraphs.append(paragraph)
        
        return valid_paragraphs
    
    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases from text."""
        if not text:
            return []
        
        # Simple phrase extraction based on common patterns
        # This can be enhanced with more sophisticated NLP techniques
        sentences = SentenceExtractor.extract_sentences(text, max_phrases * 2)
        
        key_phrases = []
        for sentence in sentences:
            # Extract phrases of 3-7 words
            words = sentence.split()
            for i in range(len(words) - 2):
                for j in range(i + 3, min(i + 8, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    if len(phrase) > 10:  # Minimum phrase length
                        key_phrases.append(phrase)
            
            if len(key_phrases) >= max_phrases:
                break
        
        return key_phrases[:max_phrases] 
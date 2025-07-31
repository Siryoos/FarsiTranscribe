"""
Unit tests for utility modules.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.utils.repetition_detector import RepetitionDetector
from src.utils.sentence_extractor import SentenceExtractor
from src.utils.file_manager import TranscriptionFileManager
from src.core.config import TranscriptionConfig


class TestRepetitionDetector:
    """Test cases for RepetitionDetector class."""
    
    def test_detect_word_repetition(self):
        """Test word repetition detection."""
        detector = RepetitionDetector()
        
        # Test normal text
        text = "hello world hello world"
        result = detector.detect_word_repetition(text, max_repetitions=2)
        assert result == "hello world hello world"
        
        # Test excessive repetition
        text = "hello hello hello hello hello"
        result = detector.detect_word_repetition(text, max_repetitions=2)
        assert result == "hello hello"
        
        # Test empty text
        result = detector.detect_word_repetition("", max_repetitions=2)
        assert result == ""
        
        # Test single word
        result = detector.detect_word_repetition("hello", max_repetitions=2)
        assert result == "hello"
    
    def test_detect_phrase_repetition(self):
        """Test phrase repetition detection."""
        detector = RepetitionDetector()
        
        # Test normal text
        text = "hello world how are you"
        result = detector.detect_phrase_repetition(text, min_phrase_length=3)
        assert result == "hello world how are you"
        
        # Test phrase repetition
        text = "hello world hello world hello world"
        result = detector.detect_phrase_repetition(text, min_phrase_length=2)
        assert result == "hello world"
        
        # Test short text
        text = "hello world"
        result = detector.detect_phrase_repetition(text, min_phrase_length=3)
        assert result == "hello world"
    
    def test_similarity_ratio(self):
        """Test similarity ratio calculation."""
        detector = RepetitionDetector()
        
        # Test identical texts
        ratio = detector.similarity_ratio("hello world", "hello world")
        assert ratio == 1.0
        
        # Test similar texts
        ratio = detector.similarity_ratio("hello world", "hello there")
        assert 0.0 < ratio < 1.0
        
        # Test different texts
        ratio = detector.similarity_ratio("hello world", "goodbye universe")
        assert ratio < 0.5
        
        # Test empty texts
        ratio = detector.similarity_ratio("", "hello")
        assert ratio == 0.0
        
        ratio = detector.similarity_ratio("hello", "")
        assert ratio == 0.0
    
    def test_clean_repetitive_text(self):
        """Test comprehensive text cleaning."""
        config = TranscriptionConfig()
        detector = RepetitionDetector()
        
        # Test normal text
        text = "hello world how are you"
        result = detector.clean_repetitive_text(text, config)
        assert result == "hello world how are you"
        
        # Test repetitive text
        text = "hello hello hello world world world"
        result = detector.clean_repetitive_text(text, config)
        # Should reduce repetitions based on config
        assert len(result.split()) < len(text.split())
        
        # Test empty text
        result = detector.clean_repetitive_text("", config)
        assert result == ""
    
    def test_find_overlapping_sequences(self):
        """Test overlapping sequence detection."""
        detector = RepetitionDetector()
        
        # Test overlapping sequences
        text1 = "hello world how are you"
        text2 = "how are you today"
        overlap, start1, start2 = detector.find_overlapping_sequences(text1, text2, min_overlap=3)
        assert overlap >= 3
        
        # Test non-overlapping sequences
        text1 = "hello world"
        text2 = "goodbye universe"
        overlap, start1, start2 = detector.find_overlapping_sequences(text1, text2, min_overlap=3)
        assert overlap == 0


class TestSentenceExtractor:
    """Test cases for SentenceExtractor class."""
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        extractor = SentenceExtractor()
        
        # Test normal text
        text = "Hello world. How are you? I am fine."
        sentences = extractor.extract_sentences(text, max_sentences=3)
        assert len(sentences) > 0
        assert all(len(sentence) > 5 for sentence in sentences)
        
        # Test empty text
        sentences = extractor.extract_sentences("", max_sentences=3)
        assert sentences == []
        
        # Test short text
        sentences = extractor.extract_sentences("Hi", max_sentences=3)
        assert sentences == []
        
        # Test Persian text with Persian punctuation
        text = "سلام دنیا. چطوری؟ من خوبم."
        sentences = extractor.extract_sentences(text, max_sentences=3)
        assert len(sentences) > 0
    
    def test_format_sentence_preview(self):
        """Test sentence preview formatting."""
        extractor = SentenceExtractor()
        
        # Test with sentences
        sentences = ["Hello world.", "How are you?"]
        preview = extractor.format_sentence_preview(sentences, 1)
        assert "Part 1 Preview:" in preview
        assert "1. Hello world." in preview
        assert "2. How are you?" in preview
        
        # Test without sentences
        preview = extractor.format_sentence_preview([], 1)
        assert "Part 1: [No meaningful content]" in preview
    
    def test_split_into_paragraphs(self):
        """Test paragraph splitting."""
        extractor = SentenceExtractor()
        
        # Test normal text
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = extractor.split_into_paragraphs(text, min_paragraph_length=10)
        assert len(paragraphs) >= 2
        
        # Test short text
        paragraphs = extractor.split_into_paragraphs("Short", min_paragraph_length=10)
        assert paragraphs == []
        
        # Test empty text
        paragraphs = extractor.split_into_paragraphs("", min_paragraph_length=10)
        assert paragraphs == []
    
    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        extractor = SentenceExtractor()
        
        # Test normal text
        text = "Hello world. How are you today? I am doing well."
        phrases = extractor.extract_key_phrases(text, max_phrases=3)
        assert len(phrases) <= 3
        assert all(len(phrase) > 10 for phrase in phrases)
        
        # Test short text
        phrases = extractor.extract_key_phrases("Short", max_phrases=3)
        assert phrases == []


class TestTranscriptionFileManager:
    """Test cases for TranscriptionFileManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TranscriptionConfig(output_directory=self.temp_dir)
        self.file_manager = TranscriptionFileManager("test_file", self.temp_dir, self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_unified_transcription(self):
        """Test transcription saving."""
        content = "Hello world. This is a test transcription."
        
        success = self.file_manager.save_unified_transcription(content)
        assert success is True
        
        # Check that files were created
        assert self.file_manager.unified_file_path.exists()
        assert self.file_manager.cleaned_file_path.exists()
        assert self.file_manager.metadata_file_path.exists()
    
    def test_get_transcription_info(self):
        """Test transcription info retrieval."""
        content = "Hello world. This is a test transcription."
        self.file_manager.save_unified_transcription(content)
        
        info = self.file_manager.get_transcription_info()
        
        assert info["base_filename"] == "test_file"
        assert info["original_exists"] is True
        assert info["cleaned_exists"] is True
        assert info["original_characters"] > 0
        assert info["cleaned_characters"] > 0
    
    def test_load_transcription(self):
        """Test transcription loading."""
        content = "Hello world. This is a test transcription."
        self.file_manager.save_unified_transcription(content)
        
        # Test loading cleaned version
        loaded_content = self.file_manager.load_transcription(use_cleaned=True)
        assert loaded_content == content
        
        # Test loading original version
        loaded_content = self.file_manager.load_transcription(use_cleaned=False)
        assert loaded_content == content
    
    def test_export_to_formats(self):
        """Test format export."""
        content = "Hello world. This is a test transcription."
        
        exported_files = self.file_manager.export_to_formats(content, ["txt", "md"])
        
        assert "txt" in exported_files
        assert "md" in exported_files
        
        # Check that files were created
        txt_path = Path(exported_files["txt"])
        md_path = Path(exported_files["md"])
        
        assert txt_path.exists()
        assert md_path.exists()
        
        # Check content
        with open(txt_path, "r", encoding="utf-8") as f:
            assert f.read() == content
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "# Transcription: test_file" in md_content
            assert content in md_content


if __name__ == "__main__":
    pytest.main([__file__]) 
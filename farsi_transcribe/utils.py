"""
Utility classes and functions for FarsiTranscribe.

This module provides helper classes for result management, text processing,
and output formatting.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """
    Container for transcription results.
    
    This class holds the transcribed text, chunk information, and metadata
    from the transcription process.
    """
    text: str
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    @property
    def duration(self) -> float:
        """
        Returns the audio duration in seconds from the transcription metadata.
        
        Returns:
            float: Audio duration in seconds, or 0.0 if not available.
        """
        return self.metadata.get('duration_seconds', 0.0)
    
    @property
    def processing_time(self) -> float:
        """
        Returns the processing time in seconds from the transcription metadata.
        
        Returns:
            float: Processing time in seconds, or 0.0 if not available.
        """
        return self.metadata.get('processing_time', 0.0)
    
    @property
    def real_time_factor(self) -> float:
        """
        Compute the ratio of processing time to audio duration for the transcription.
        
        Returns:
            float: The real-time factor, or 0.0 if the audio duration is zero.
        """
        if self.duration > 0:
            return self.processing_time / self.duration
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the TranscriptionResult instance to a dictionary representation.
        
        Returns:
            dict: A dictionary containing all fields and values of the transcription result.
        """
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the transcription result to a JSON-formatted string.
        
        Parameters:
            indent (int): Number of spaces to use for indentation in the output JSON string.
        
        Returns:
            str: The JSON representation of the transcription result.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save_text(self, output_path: Path):
        """
        Saves the transcribed text to a UTF-8 encoded file at the specified path, creating parent directories if necessary.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.text, encoding='utf-8')
        logger.info(f"Saved transcription text to: {output_path}")
    
    def save_json(self, output_path: Path):
        """
        Save the entire transcription result as a JSON file at the specified path, creating parent directories if necessary.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding='utf-8')
        logger.info(f"Saved transcription JSON to: {output_path}")
    
    def save_segments(self, output_path: Path):
        """
        Save the transcription chunks to a file, formatting each as a segment with start and end timestamps.
        
        Each line in the output file represents a chunk in the format: "[start_time - end_time] text". The file is saved with UTF-8 encoding, and parent directories are created if they do not exist.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for chunk in self.chunks:
            start_time = chunk.get('start_time', 0)
            end_time = chunk.get('end_time', 0)
            text = chunk.get('text', '').strip()
            if text:
                lines.append(f"[{start_time:.1f}s - {end_time:.1f}s] {text}")
        
        output_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Saved transcription segments to: {output_path}")


class TextProcessor:
    """
    Text processing utilities for Persian/Farsi text.
    
    This class provides methods for normalizing, cleaning, and processing
    Persian text to improve transcription quality.
    """
    
    def __init__(self, language: str = 'fa'):
        """
        Initialize a TextProcessor for Persian text normalization and cleaning.
        
        Parameters:
        	language (str): Language code for processing, defaults to 'fa' (Farsi/Persian).
        """
        self.language = language
        
        # Persian-specific characters and replacements
        self.persian_chars = {
            'ك': 'ک',  # Arabic Kaf to Persian Kaf
            'ي': 'ی',  # Arabic Yeh to Persian Yeh
            'ى': 'ی',  # Alef Maksura to Persian Yeh
            'ة': 'ه',  # Teh Marbuta to Heh
            '٠': '۰', '١': '۱', '٢': '۲', '٣': '۳', '٤': '۴',
            '٥': '۵', '٦': '۶', '٧': '۷', '٨': '۸', '٩': '۹'  # Arabic to Persian digits
        }
        
        # Diacritic marks to potentially remove
        self.diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    
    def normalize_persian(self, text: str) -> str:
        """
        Replaces Arabic characters and digits in the input text with their Persian equivalents and normalizes whitespace.
        
        Parameters:
            text (str): The input text to normalize.
        
        Returns:
            str: The normalized Persian text.
        """
        if not text:
            return text
        
        # Replace Arabic characters with Persian equivalents
        for arabic, persian in self.persian_chars.items():
            text = text.replace(arabic, persian)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritical marks from the given text.
        
        Parameters:
            text (str): The input text to process.
        
        Returns:
            str: The text with all Arabic diacritics removed.
        """
        return self.diacritics.sub('', text)
    
    def clean_text(self, text: str, remove_diacritics: bool = False) -> str:
        """
        Cleans and normalizes Persian text by standardizing characters, optionally removing diacritics, and normalizing whitespace.
        
        Parameters:
            text (str): The input text to be cleaned.
            remove_diacritics (bool): If True, removes Arabic diacritical marks from the text.
        
        Returns:
            str: The cleaned and normalized text.
        """
        # Normalize Persian characters
        text = self.normalize_persian(text)
        
        # Remove diacritics if requested
        if remove_diacritics:
            text = self.remove_diacritics(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Splits Persian text into sentences using common sentence delimiters.
        
        Parameters:
            text (str): The input Persian text to segment.
        
        Returns:
            List[str]: A list of sentences, each ending with its delimiter if present.
        """
        # Persian sentence delimiters
        delimiters = r'[.!?؟।۔]'
        
        # Split by delimiters but keep them
        sentences = re.split(f'({delimiters})', text)
        
        # Reconstruct sentences with their delimiters
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence:
                result.append(sentence)
        
        # Handle last sentence if no delimiter
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return result


class TranscriptionManager:
    """
    Manages transcription output and file operations.
    
    This class handles saving transcriptions in various formats and
    managing output directories.
    """
    
    def __init__(self, output_directory: Path):
        """
        Initialize a TranscriptionManager to handle output file operations in the specified directory.
        
        The output directory is created if it does not already exist.
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, 
                   result: TranscriptionResult,
                   base_name: str,
                   formats: List[str]) -> Dict[str, Path]:
        """
                   Save a transcription result to the output directory in one or more specified formats.
                   
                   Parameters:
                       result (TranscriptionResult): The transcription result to save.
                       base_name (str): The base filename (without extension) for output files.
                       formats (List[str]): List of formats to save ('txt', 'json', 'segments').
                   
                   Returns:
                       Dict[str, Path]: A dictionary mapping each saved format to its corresponding output file path.
                   """
        saved_files = {}
        
        if 'txt' in formats:
            txt_path = self.output_directory / f"{base_name}.txt"
            result.save_text(txt_path)
            saved_files['txt'] = txt_path
        
        if 'json' in formats:
            json_path = self.output_directory / f"{base_name}.json"
            result.save_json(json_path)
            saved_files['json'] = json_path
        
        if 'segments' in formats:
            segments_path = self.output_directory / f"{base_name}_segments.txt"
            result.save_segments(segments_path)
            saved_files['segments'] = segments_path
        
        return saved_files
    
    def create_summary(self, result: TranscriptionResult) -> str:
        """
        Generate a formatted summary string containing key statistics and metadata from a transcription result.
        
        Parameters:
            result (TranscriptionResult): The transcription result to summarize.
        
        Returns:
            str: A multi-line string summarizing audio duration, processing time, real-time factor, model name, number of chunks, and text statistics.
        """
        summary = f"""
Transcription Summary
====================
Audio Duration: {result.duration:.1f} seconds
Processing Time: {result.processing_time:.1f} seconds
Real-time Factor: {result.real_time_factor:.2f}x
Model: {result.metadata.get('model', 'Unknown')}
Chunks Processed: {len(result.chunks)}
Total Characters: {len(result.text)}
Total Words: {len(result.text.split())}
"""
        return summary.strip()
    
    def save_summary(self, result: TranscriptionResult, base_name: str) -> Path:
        """
        Save a textual summary of the transcription result to a file in the output directory.
        
        Parameters:
            result (TranscriptionResult): The transcription result to summarize.
            base_name (str): The base filename for the summary file.
        
        Returns:
            Path: The path to the saved summary file.
        """
        summary_path = self.output_directory / f"{base_name}_summary.txt"
        summary = self.create_summary(result)
        summary_path.write_text(summary, encoding='utf-8')
        logger.info(f"Saved summary to: {summary_path}")
        return summary_path
"""
Dataset builder for Whisper fine-tuning.
Prepares audio-text pairs in the format required for Whisper training.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Audio, Features, Value, Sequence
import torch
import torchaudio
import librosa


@dataclass
class DatasetBuilderConfig:
    """Configuration for dataset building."""
    
    # Data settings
    audio_directory: str = ""
    transcript_directory: str = ""
    output_directory: str = ""
    
    # Audio processing
    target_sample_rate: int = 16000
    max_audio_length: float = 30.0  # seconds
    min_audio_length: float = 1.0   # seconds
    
    # Text processing
    language: str = "fa"  # Persian/Farsi
    normalize_text: bool = True
    remove_empty_transcripts: bool = True
    
    # Dataset splitting
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    
    # Quality control
    min_text_length: int = 3
    max_text_length: int = 500
    quality_threshold: float = 0.7
    
    # Output format
    output_format: str = "huggingface"  # huggingface, json, csv
    save_metadata: bool = True


class WhisperDatasetBuilder:
    """Dataset builder for Whisper fine-tuning with Farsi voice data."""
    
    def __init__(self, config: DatasetBuilderConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
        
        # Initialize paths
        self.audio_path = Path(config.audio_directory)
        self.transcript_path = Path(config.transcript_directory)
        self.output_path = Path(config.output_directory)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.audio_directory:
            raise ValueError("audio_directory must be specified")
        
        if not self.config.transcript_directory:
            raise ValueError("transcript_directory must be specified")
        
        if not self.config.output_directory:
            raise ValueError("output_directory must be specified")
        
        # Validate split ratios
        total_split = self.config.train_split + self.config.validation_split + self.config.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_split}")
    
    def build_dataset(self) -> Dict[str, Dataset]:
        """
        Build the complete dataset for fine-tuning.
        
        Returns:
            Dictionary containing train, validation, and test datasets
        """
        try:
            self.logger.info("Starting dataset building process...")
            
            # Step 1: Collect and validate data
            data_pairs = self._collect_data_pairs()
            
            if not data_pairs:
                raise ValueError("No valid data pairs found")
            
            self.logger.info(f"Collected {len(data_pairs)} valid data pairs")
            
            # Step 2: Split data
            train_data, val_data, test_data = self._split_data(data_pairs)
            
            # Step 3: Create datasets
            train_dataset = self._create_dataset(train_data, "train")
            val_dataset = self._create_dataset(val_data, "validation")
            test_dataset = self._create_dataset(test_data, "test")
            
            # Step 4: Save datasets
            self._save_datasets({
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            })
            
            # Step 5: Create metadata
            if self.config.save_metadata:
                self._create_metadata(data_pairs, train_data, val_data, test_data)
            
            self.logger.info("Dataset building completed successfully")
            
            return {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            }
            
        except Exception as e:
            self.logger.error(f"Error building dataset: {str(e)}")
            raise
    
    def _collect_data_pairs(self) -> List[Dict[str, str]]:
        """Collect and validate audio-transcript pairs."""
        data_pairs = []
        
        # Find all audio files
        audio_files = self._find_audio_files()
        transcript_files = self._find_transcript_files()
        
        # Create mapping between audio and transcript files
        audio_transcript_map = self._create_audio_transcript_mapping(audio_files, transcript_files)
        
        # Process each pair
        for audio_file, transcript_file in audio_transcript_map.items():
            try:
                # Validate audio file
                audio_info = self._validate_audio_file(audio_file)
                if not audio_info:
                    continue
                
                # Validate transcript file
                transcript_text = self._validate_transcript_file(transcript_file)
                if not transcript_text:
                    continue
                
                # Create data pair
                data_pair = {
                    "audio_path": str(audio_file),
                    "transcript": transcript_text,
                    "duration": audio_info["duration"],
                    "sample_rate": audio_info["sample_rate"],
                    "file_size": audio_info["file_size"]
                }
                
                data_pairs.append(data_pair)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {audio_file}: {str(e)}")
                continue
        
        return data_pairs
    
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the audio directory."""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for audio_file in self.audio_path.rglob('*'):
            if audio_file.suffix.lower() in audio_extensions:
                audio_files.append(audio_file)
        
        return audio_files
    
    def _find_transcript_files(self) -> List[Path]:
        """Find all transcript files in the transcript directory."""
        transcript_extensions = {'.txt', '.json', '.csv'}
        transcript_files = []
        
        for transcript_file in self.transcript_path.rglob('*'):
            if transcript_file.suffix.lower() in transcript_extensions:
                transcript_files.append(transcript_file)
        
        return transcript_files
    
    def _create_audio_transcript_mapping(self, audio_files: List[Path], transcript_files: List[Path]) -> Dict[Path, Path]:
        """Create mapping between audio files and their corresponding transcript files."""
        mapping = {}
        
        for audio_file in audio_files:
            # Try to find corresponding transcript file
            transcript_file = self._find_corresponding_transcript(audio_file, transcript_files)
            if transcript_file:
                mapping[audio_file] = transcript_file
        
        return mapping
    
    def _find_corresponding_transcript(self, audio_file: Path, transcript_files: List[Path]) -> Optional[Path]:
        """Find the transcript file corresponding to an audio file."""
        audio_stem = audio_file.stem
        
        # Try exact match first
        for transcript_file in transcript_files:
            if transcript_file.stem == audio_stem:
                return transcript_file
        
        # Try partial match (e.g., audio_001.wav -> transcript_001.txt)
        for transcript_file in transcript_files:
            if audio_stem in transcript_file.stem or transcript_file.stem in audio_stem:
                return transcript_file
        
        return None
    
    def _validate_audio_file(self, audio_file: Path) -> Optional[Dict]:
        """Validate audio file and extract metadata."""
        try:
            # Load audio to get duration and sample rate
            audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
            
            duration = len(audio) / sr
            file_size = audio_file.stat().st_size
            
            # Check duration constraints
            if duration < self.config.min_audio_length or duration > self.config.max_audio_length:
                return None
            
            # Check file size (reasonable limits)
            if file_size < 1024 or file_size > 100 * 1024 * 1024:  # 1KB to 100MB
                return None
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "file_size": file_size
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to validate audio file {audio_file}: {str(e)}")
            return None
    
    def _validate_transcript_file(self, transcript_file: Path) -> Optional[str]:
        """Validate transcript file and extract text."""
        try:
            # Read transcript content
            if transcript_file.suffix.lower() == '.json':
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle different JSON formats
                    if isinstance(data, dict):
                        text = data.get('text', data.get('transcript', data.get('caption', '')))
                    else:
                        text = str(data)
            else:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            
            # Validate text
            if not text or len(text) < self.config.min_text_length:
                return None
            
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length]
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Failed to read transcript {transcript_file}: {str(e)}")
            return None
    
    def _split_data(self, data_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation, and test sets."""
        # Shuffle data
        np.random.shuffle(data_pairs)
        
        total_samples = len(data_pairs)
        train_end = int(total_samples * self.config.train_split)
        val_end = train_end + int(total_samples * self.config.validation_split)
        
        train_data = data_pairs[:train_end]
        val_data = data_pairs[train_end:val_end]
        test_data = data_pairs[val_end:]
        
        self.logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _create_dataset(self, data: List[Dict], split_name: str) -> Dataset:
        """Create a HuggingFace dataset from data."""
        # Prepare data for dataset creation
        dataset_data = []
        
        for item in data:
            dataset_item = {
                "audio": item["audio_path"],
                "transcript": item["transcript"],
                "duration": item["duration"],
                "sample_rate": item["sample_rate"],
                "language": self.config.language
            }
            dataset_data.append(dataset_item)
        
        # Create dataset
        dataset = Dataset.from_list(dataset_data)
        
        # Add audio feature
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.target_sample_rate))
        
        return dataset
    
    def _save_datasets(self, datasets: Dict[str, Dataset]):
        """Save datasets in the specified format."""
        if self.config.output_format == "huggingface":
            # Save as HuggingFace dataset
            dataset_dict = DatasetDict(datasets)
            dataset_dict.save_to_disk(str(self.output_path / "dataset"))
            
        elif self.config.output_format == "json":
            # Save as JSON files
            for split_name, dataset in datasets.items():
                output_file = self.output_path / f"{split_name}.json"
                dataset.to_json(str(output_file))
                
        elif self.config.output_format == "csv":
            # Save as CSV files
            for split_name, dataset in datasets.items():
                output_file = self.output_path / f"{split_name}.csv"
                dataset.to_csv(str(output_file), index=False)
    
    def _create_metadata(self, all_data: List[Dict], train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
        """Create metadata about the dataset."""
        metadata = {
            "dataset_info": {
                "total_samples": len(all_data),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "test_samples": len(test_data),
                "language": self.config.language,
                "target_sample_rate": self.config.target_sample_rate
            },
            "audio_statistics": self._calculate_audio_statistics(all_data),
            "text_statistics": self._calculate_text_statistics(all_data),
            "split_ratios": {
                "train": self.config.train_split,
                "validation": self.config.validation_split,
                "test": self.config.test_split
            }
        }
        
        # Save metadata
        metadata_file = self.output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _calculate_audio_statistics(self, data: List[Dict]) -> Dict:
        """Calculate audio-related statistics."""
        durations = [item["duration"] for item in data]
        file_sizes = [item["file_size"] for item in data]
        
        return {
            "duration": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "total_hours": sum(durations) / 3600
            },
            "file_size": {
                "mean": np.mean(file_sizes),
                "std": np.std(file_sizes),
                "min": np.min(file_sizes),
                "max": np.max(file_sizes),
                "total_gb": sum(file_sizes) / (1024**3)
            }
        }
    
    def _calculate_text_statistics(self, data: List[Dict]) -> Dict:
        """Calculate text-related statistics."""
        text_lengths = [len(item["transcript"]) for item in data]
        word_counts = [len(item["transcript"].split()) for item in data]
        
        return {
            "text_length": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths)
            },
            "word_count": {
                "mean": np.mean(word_counts),
                "std": np.std(word_counts),
                "min": np.min(word_counts),
                "max": np.max(word_counts)
            }
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the built dataset."""
        dataset_path = self.output_path / "dataset"
        
        if not dataset_path.exists():
            return {"error": "Dataset not found"}
        
        try:
            dataset_dict = DatasetDict.load_from_disk(str(dataset_path))
            
            info = {
                "splits": list(dataset_dict.keys()),
                "total_samples": sum(len(split) for split in dataset_dict.values()),
                "split_samples": {split: len(dataset) for split, dataset in dataset_dict.items()}
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}

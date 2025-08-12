#!/usr/bin/env python3
"""
Dataset downloader for Farsi voice fine-tuning.
Downloads and prepares various Persian audio datasets for Whisper fine-tuning.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import zipfile
import tarfile
from tqdm import tqdm
import hashlib
import time

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import only what we need, with fallbacks
try:
    from src.finetuning.configs.finetuning_config import (
        FineTuningConfig,
        DataConfig,
    )
except ImportError:
    # Create minimal fallback classes if imports fail
    class DataConfig:
        def __init__(self):
            pass

    class FineTuningConfig:
        def __init__(self):
            self.data = DataConfig()


class DatasetDownloader:
    """Downloader for various Persian audio datasets."""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Dataset configurations
        self.datasets = {
            "local_sample": {
                "name": "Local Persian Audio Sample",
                "description": "Local sample audio files for testing and development",
                "url": "local",  # Special case for local files
                "size_mb": 5,  # Approximate size
                "format": "local",
                "subdir": "local_sample",
                "license": "Project Internal",
                "citation": "FarsiTranscribe Project Sample Data",
            },
            "common_voice_fa": {
                "name": "Common Voice Persian (Farsi)",
                "description": "Mozilla Common Voice dataset for Persian language - Requires special access",
                "url": "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
                "size_mb": 2500,  # Approximate size
                "format": "huggingface",
                "subdir": "common_voice_17_0",
                "license": "CC0-1.0",
                "citation": "Ardila et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus.",
                "requires_auth": True,
            },
            "huggingface_fa": {
                "name": "HuggingFace Persian Audio",
                "description": "Persian audio datasets from HuggingFace",
                "url": "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
                "size_mb": 100,
                "format": "huggingface",
                "subdir": "huggingface_fa",
                "license": "CC0-1.0",
                "citation": "HuggingFace Common Voice Persian Dataset",
                "requires_auth": True,
            },
            "sample_audio": {
                "name": "Sample Audio Dataset",
                "description": "Small sample audio dataset for testing and development",
                "url": "local",
                "size_mb": 50,
                "format": "local",
                "subdir": "sample_audio",
                "license": "MIT",
                "citation": "FarsiTranscribe Sample Data",
            },
        }

    def download_file(
        self,
        url: str,
        filepath: Path,
        expected_size_mb: Optional[float] = None,
    ) -> bool:
        """Download a file with progress bar and validation."""
        try:
            self.logger.info(f"Downloading: {url}")
            self.logger.info(f"Destination: {filepath}")

            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if expected_size_mb and total_size == 0:
                total_size = int(expected_size_mb * 1024 * 1024)

            with open(filepath, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filepath.name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Validate file size
            actual_size = filepath.stat().st_size
            if (
                expected_size_mb
                and actual_size < expected_size_mb * 1024 * 1024 * 0.9
            ):
                self.logger.warning(
                    f"File size seems too small: {actual_size / (1024*1024):.1f} MB"
                )

            self.logger.info(
                f"Download completed: {actual_size / (1024*1024):.1f} MB"
            )
            return True

        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            return False

    def extract_archive(
        self, archive_path: Path, extract_dir: Path, format_type: str
    ) -> bool:
        """Extract downloaded archive."""
        try:
            self.logger.info(
                f"Extracting {archive_path.name} to {extract_dir}"
            )

            if format_type == "tar.gz":
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
            elif format_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif format_type == "pt":
                # For .pt files, just copy to the target directory
                extract_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(archive_path, extract_dir / archive_path.name)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            self.logger.info("Extraction completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return False

    def download_dataset(
        self, dataset_key: str, force_download: bool = False
    ) -> bool:
        """Download a specific dataset."""
        if dataset_key not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False

        dataset = self.datasets[dataset_key]
        dataset_dir = self.output_dir / dataset_key

        # Check if already downloaded
        if dataset_dir.exists() and not force_download:
            self.logger.info(
                f"Dataset {dataset_key} already exists at {dataset_dir}"
            )
            return True

        # Handle local sample dataset
        if dataset_key == "local_sample":
            return self._create_local_sample_dataset(dataset_dir)

        # Handle sample audio dataset
        if dataset_key == "sample_audio":
            return self._create_sample_audio_dataset(dataset_dir)

        # Handle HuggingFace datasets
        if dataset["format"] == "huggingface":
            return self._download_huggingface_dataset(dataset, dataset_dir)

        # Download archive
        archive_name = f"{dataset_key}.{dataset['format']}"
        archive_path = self.output_dir / archive_name

        if not self.download_file(
            dataset["url"], archive_path, dataset["size_mb"]
        ):
            return False

        # Extract archive
        if not self.extract_archive(
            archive_path, self.output_dir, dataset["format"]
        ):
            return False

        # Clean up archive
        archive_path.unlink()

        # Verify extraction
        if not dataset_dir.exists():
            self.logger.error(
                f"Expected dataset directory not found: {dataset_dir}"
            )
            return False

        self.logger.info(
            f"Dataset {dataset_key} downloaded and extracted successfully"
        )
        return True

    def _create_local_sample_dataset(self, dataset_dir: Path) -> bool:
        """Create a local sample dataset from existing audio files."""
        try:
            self.logger.info("Creating local sample dataset...")

            # Create dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Look for existing audio files in the project
            project_root = Path(__file__).parent.parent.parent
            examples_audio = project_root / "examples" / "audio"

            if examples_audio.exists():
                # Copy existing audio files
                import shutil

                for audio_file in examples_audio.glob("*.m4a"):
                    shutil.copy2(audio_file, dataset_dir / audio_file.name)
                    self.logger.info(f"Copied: {audio_file.name}")

                # Create a simple metadata file
                metadata = {
                    "dataset_name": "Local Persian Audio Sample",
                    "description": "Sample audio files from FarsiTranscribe project",
                    "files": [f.name for f in dataset_dir.glob("*.m4a")],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": len(list(dataset_dir.glob("*.m4a"))),
                }

                with open(
                    dataset_dir / "metadata.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                self.logger.info(
                    f"Local sample dataset created successfully with {metadata['total_files']} files"
                )
                return True
            else:
                self.logger.warning("No examples/audio directory found")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create local sample dataset: {e}")
            return False

    def _create_sample_audio_dataset(self, dataset_dir: Path) -> bool:
        """Create a sample audio dataset for testing and development."""
        try:
            self.logger.info("Creating sample audio dataset...")

            # Create dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Look for existing audio files in the project
            project_root = Path(__file__).parent.parent.parent
            examples_audio = project_root / "examples" / "audio"

            if examples_audio.exists():
                # Copy existing audio files
                import shutil

                audio_files = (
                    list(examples_audio.glob("*.m4a"))
                    + list(examples_audio.glob("*.mp3"))
                    + list(examples_audio.glob("*.wav"))
                )

                if audio_files:
                    for audio_file in audio_files:
                        shutil.copy2(audio_file, dataset_dir / audio_file.name)
                        self.logger.info(f"Copied: {audio_file.name}")

                    # Create metadata file
                    metadata = {
                        "dataset_name": "Sample Audio Dataset",
                        "description": "Sample audio files for testing and development",
                        "files": [f.name for f in audio_files],
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_files": len(audio_files),
                        "total_duration_estimate": "varies",
                        "format": "mixed",
                    }

                    with open(
                        dataset_dir / "metadata.json", "w", encoding="utf-8"
                    ) as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

                    self.logger.info(
                        f"Sample audio dataset created successfully with {len(audio_files)} files"
                    )
                    return True
                else:
                    self.logger.warning(
                        "No audio files found in examples/audio directory"
                    )
                    return False
            else:
                self.logger.warning("No examples/audio directory found")
                return False

        except Exception as e:
            self.logger.error(f"Failed to create sample audio dataset: {e}")
            return False

    def _download_huggingface_dataset(
        self, dataset: Dict, dataset_dir: Path
    ) -> bool:
        """Download a HuggingFace dataset."""
        try:
            self.logger.info(
                f"Downloading HuggingFace dataset: {dataset['name']}"
            )

            # Create dataset directory
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                self.logger.error(
                    "datasets library not installed. Install with: pip install datasets"
                )
                return False

            # Extract dataset name from URL
            dataset_name = dataset["url"].split("/")[-1]

            # Load the dataset
            self.logger.info(f"Loading dataset: {dataset_name}")
            ds = load_dataset(dataset_name, "fa")  # Load Persian subset

            # Save dataset info
            dataset_info = {
                "name": dataset["name"],
                "description": dataset["description"],
                "license": dataset["license"],
                "citation": dataset["citation"],
                "dataset_id": dataset_name,
                "features": (
                    str(ds["train"].features)
                    if "train" in ds
                    else "No train split"
                ),
                "num_examples": len(ds["train"]) if "train" in ds else 0,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(
                dataset_dir / "dataset_info.json", "w", encoding="utf-8"
            ) as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)

            # Save a sample of the dataset for inspection
            if "train" in ds and len(ds["train"]) > 0:
                sample_data = ds["train"].select(
                    range(min(10, len(ds["train"])))
                )
                sample_data.save_to_disk(str(dataset_dir / "sample"))
                self.logger.info(
                    f"Saved sample data with {len(sample_data)} examples"
                )

            self.logger.info(
                f"HuggingFace dataset {dataset_name} downloaded successfully"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to download HuggingFace dataset: {e}")
            return False

    def download_all_datasets(
        self, force_download: bool = False
    ) -> Dict[str, bool]:
        """Download all available datasets."""
        results = {}

        self.logger.info("Starting download of all datasets...")
        self.logger.info(f"Output directory: {self.output_dir}")

        for dataset_key in self.datasets:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(
                f"Processing dataset: {self.datasets[dataset_key]['name']}"
            )
            self.logger.info(f"{'='*50}")

            try:
                success = self.download_dataset(dataset_key, force_download)
                results[dataset_key] = success

                if success:
                    self.logger.info(f"✅ {dataset_key}: SUCCESS")
                else:
                    self.logger.error(f"❌ {dataset_key}: FAILED")

            except Exception as e:
                self.logger.error(f"❌ {dataset_key}: ERROR - {e}")
                results[dataset_key] = False

        return results

    def list_datasets(self) -> None:
        """List all available datasets with information."""
        print("\n📚 Available Datasets for Download")
        print("=" * 60)

        for key, dataset in self.datasets.items():
            print(f"\n🔹 {dataset['name']}")
            print(f"   Key: {key}")
            print(f"   Description: {dataset['description']}")
            print(f"   Size: ~{dataset['size_mb']} MB")
            print(f"   Format: {dataset['format']}")
            print(f"   License: {dataset['license']}")
            print(f"   Citation: {dataset['citation']}")

        print(f"\n📁 Output directory: {self.output_dir}")
        print("\n💡 Usage:")
        print("   python download_datasets.py --dataset common_voice_fa")
        print("   python download_datasets.py --all")
        print("   python download_datasets.py --list")

    def create_dataset_info(self) -> None:
        """Create dataset information file."""
        info_file = self.output_dir / "dataset_info.json"

        dataset_info = {
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_datasets": len(self.datasets),
            "datasets": {},
        }

        for key, dataset in self.datasets.items():
            dataset_path = self.output_dir / key
            dataset_info["datasets"][key] = {
                "name": dataset["name"],
                "description": dataset["description"],
                "license": dataset["license"],
                "citation": dataset["citation"],
                "downloaded": dataset_path.exists(),
                "path": str(dataset_path) if dataset_path.exists() else None,
                "size_mb": dataset["size_mb"],
            }

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset info saved to: {info_file}")

    def get_download_stats(self) -> Dict[str, int]:
        """Get statistics about downloaded datasets."""
        stats = {
            "total": len(self.datasets),
            "downloaded": 0,
            "total_size_mb": 0,
        }

        for key, dataset in self.datasets.items():
            dataset_path = self.output_dir / key
            if dataset_path.exists():
                stats["downloaded"] += 1
                stats["total_size_mb"] += dataset["size_mb"]

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download Persian audio datasets for fine-tuning"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to download (e.g., common_voice_fa)",
    )

    parser.add_argument(
        "--all", action="store_true", help="Download all available datasets"
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available datasets"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for datasets",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing datasets",
    )

    parser.add_argument(
        "--info", action="store_true", help="Create dataset information file"
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatasetDownloader(args.output_dir)

    if args.list:
        downloader.list_datasets()
        return

    if args.info:
        downloader.create_dataset_info()
        return

    if args.dataset:
        # Download specific dataset
        if args.dataset not in downloader.datasets:
            print(f"❌ Unknown dataset: {args.dataset}")
            print("Use --list to see available datasets")
            return

        success = downloader.download_dataset(args.dataset, args.force)
        if success:
            print(f"✅ Dataset {args.dataset} downloaded successfully")
        else:
            print(f"❌ Failed to download dataset {args.dataset}")
            sys.exit(1)

    elif args.all:
        # Download all datasets
        results = downloader.download_all_datasets(args.force)

        print(f"\n📊 Download Summary")
        print("=" * 30)

        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        for dataset_key, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{dataset_key}: {status}")

        print(
            f"\nOverall: {success_count}/{total_count} datasets downloaded successfully"
        )

        if success_count < total_count:
            print("Some datasets failed to download. Check the logs above.")
            sys.exit(1)

    else:
        print("❌ Please specify --dataset, --all, --list, or --info")
        parser.print_help()
        return

    # Create dataset info
    downloader.create_dataset_info()

    # Show statistics
    stats = downloader.get_download_stats()
    print(f"\n📈 Dataset Statistics")
    print(f"Total available: {stats['total']}")
    print(f"Downloaded: {stats['downloaded']}")
    print(f"Total size: {stats['total_size_mb']} MB")

    print(f"\n🎉 Dataset download completed!")
    print(f"Datasets are available in: {downloader.output_dir}")


if __name__ == "__main__":
    main()

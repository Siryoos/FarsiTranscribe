"""
Enhanced file management with text post-processing capabilities.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from ..core.config import TranscriptionConfig
from .repetition_detector import RepetitionDetector


class TranscriptionFileManager:
    """Enhanced file management with text post-processing capabilities."""

    def __init__(
        self,
        base_filename: str,
        output_directory: str,
        config: TranscriptionConfig,
    ):
        self.base_filename = base_filename
        self.output_directory = Path(output_directory)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Define output file paths
        self.unified_file_path = (
            self.output_directory
            / f"{base_filename}{config.unified_filename_suffix}"
        )
        self.cleaned_file_path = (
            self.output_directory
            / f"{base_filename}_cleaned_transcription.txt"
        )
        self.metadata_file_path = (
            self.output_directory / f"{base_filename}_metadata.json"
        )

    def save_unified_transcription(self, content: str) -> bool:
        """Save unified transcription with cleaning."""
        try:
            # Save original version
            with open(self.unified_file_path, "w", encoding="utf-8") as file:
                file.write(content)

            # Create and save cleaned version
            cleaned_content = RepetitionDetector.clean_repetitive_text(
                content, self.config
            )
            with open(self.cleaned_file_path, "w", encoding="utf-8") as file:
                file.write(cleaned_content)

            # Save metadata
            self._save_metadata(content, cleaned_content)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save transcription: {e}")
            return False

    def _save_metadata(
        self, original_content: str, cleaned_content: str
    ) -> None:
        """Save metadata about the transcription."""
        import json

        metadata = {
            "base_filename": self.base_filename,
            "original_size": len(original_content),
            "cleaned_size": len(cleaned_content),
            "original_words": len(original_content.split()),
            "cleaned_words": len(cleaned_content.split()),
            "reduction_percentage": (
                (
                    (len(original_content) - len(cleaned_content))
                    / len(original_content)
                    * 100
                )
                if original_content
                else 0
            ),
            "config": self.config.to_dict(),
        }

        try:
            with open(self.metadata_file_path, "w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")

    def get_transcription_info(self) -> Dict[str, Any]:
        """Get comprehensive information about transcription files."""
        info = {
            "base_filename": self.base_filename,
            "unified_file_path": str(self.unified_file_path),
            "cleaned_file_path": str(self.cleaned_file_path),
            "metadata_file_path": str(self.metadata_file_path),
            "output_directory": str(self.output_directory),
        }

        # Check all files
        for file_type, file_path in [
            ("original", self.unified_file_path),
            ("cleaned", self.cleaned_file_path),
            ("metadata", self.metadata_file_path),
        ]:
            info[f"{file_type}_exists"] = file_path.exists()
            info[f"{file_type}_size"] = 0
            info[f"{file_type}_characters"] = 0
            info[f"{file_type}_words"] = 0

            if info[f"{file_type}_exists"]:
                try:
                    info[f"{file_type}_size"] = file_path.stat().st_size
                    if file_type != "metadata":
                        with open(file_path, "r", encoding="utf-8") as file:
                            content = file.read()
                            info[f"{file_type}_characters"] = len(content)
                            info[f"{file_type}_words"] = len(content.split())
                except Exception as e:
                    self.logger.error(
                        f"Error reading {file_type} file info: {e}"
                    )

        return info

    def load_transcription(self, use_cleaned: bool = True) -> str:
        """Load transcription from file."""
        file_path = (
            self.cleaned_file_path if use_cleaned else self.unified_file_path
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Transcription file not found: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            self.logger.error(f"Failed to load transcription: {e}")
            raise

    def export_to_formats(
        self, content: str, formats: list = None
    ) -> Dict[str, str]:
        """Export transcription to multiple formats."""
        if formats is None:
            formats = ["txt", "md", "json"]

        exported_files = {}

        for format_type in formats:
            try:
                if format_type == "txt":
                    file_path = (
                        self.output_directory
                        / f"{self.base_filename}_transcription.txt"
                    )
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(content)
                    exported_files["txt"] = str(file_path)

                elif format_type == "md":
                    file_path = (
                        self.output_directory
                        / f"{self.base_filename}_transcription.md"
                    )
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(
                            f"# Transcription: {self.base_filename}\n\n"
                        )
                        file.write(content)
                    exported_files["md"] = str(file_path)

                elif format_type == "json":
                    import json

                    file_path = (
                        self.output_directory
                        / f"{self.base_filename}_transcription.json"
                    )
                    data = {
                        "filename": self.base_filename,
                        "content": content,
                        "word_count": len(content.split()),
                        "character_count": len(content),
                    }
                    with open(file_path, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=2, ensure_ascii=False)
                    exported_files["json"] = str(file_path)

            except Exception as e:
                self.logger.error(f"Failed to export to {format_type}: {e}")

        return exported_files

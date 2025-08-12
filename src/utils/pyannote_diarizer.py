"""
Advanced Speaker Diarization using pyannote.audio.
Provides much better accuracy than the basic MFCC approach.
"""

import logging
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

try:
    from pyannote.audio import Pipeline
    from pyannote.audio.core.io import Audio

    PYAANOTE_AVAILABLE = True
except ImportError:
    PYAANOTE_AVAILABLE = False
    Pipeline = None
    Audio = None

from ..core.config import TranscriptionConfig


@dataclass
class SpeakerSegment:
    """Represents a segment of audio from a specific speaker."""

    start_time: float
    end_time: float
    speaker_id: str
    audio_data: np.ndarray
    confidence: float
    features: Optional[np.ndarray] = None


class PyannoteDiarizer:
    """Advanced speaker diarization using pyannote.audio."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Initialize the pyannote pipeline."""
        if not PYAANOTE_AVAILABLE:
            raise ImportError(
                "pyannote.audio not available. Install with: pip install pyannote.audio==3.1.1"
            )

        try:
            # Use local cache if available, otherwise download
            model_name = "pyannote/speaker-diarization-3.1"
            self.logger.info(
                f"Loading pyannote diarization model: {model_name}"
            )

            # Try to load from cache first
            cache_dir = os.path.expanduser("~/.cache/huggingface")
            if os.path.exists(cache_dir):
                self.logger.info("Using cached pyannote model")

            self.pipeline = Pipeline.from_pretrained(model_name)
            self.logger.info(
                "✅ Pyannote diarization pipeline loaded successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to load pyannote pipeline: {e}")
            raise

    def diarize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Perform speaker diarization using pyannote.audio."""
        try:
            self.logger.info("Starting pyannote speaker diarization...")

            # Save audio to temporary file (pyannote expects file path)
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                # Ensure audio is float32 and in [-1, 1] range
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Normalize if needed
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

                # Save as WAV
                sf.write(tmp_file.name, audio_data, sample_rate)
                tmp_path = tmp_file.name

            try:
                # Run diarization
                diarization = self.pipeline(
                    {"audio": tmp_path},
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

                # Convert to our format
                segments = []
                for turn, _, speaker in diarization.itertracks(
                    yield_label=True
                ):
                    start_time = float(turn.start)
                    end_time = float(turn.end)

                    # Extract audio segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_audio = audio_data[start_sample:end_sample]

                    segment = SpeakerSegment(
                        start_time=start_time,
                        end_time=end_time,
                        speaker_id=speaker,
                        audio_data=segment_audio,
                        confidence=0.9,  # pyannote doesn't provide confidence scores
                    )
                    segments.append(segment)

                self.logger.info(
                    f"Diarization complete. Found {len(set(s.speaker_id for s in segments))} speakers"
                )
                return segments

            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            self.logger.error(f"Pyannote diarization failed: {e}")
            raise

    def merge_similar_speakers(
        self, segments: List[SpeakerSegment], gap_threshold: float = 2.0
    ) -> List[SpeakerSegment]:
        """Merge consecutive segments from the same speaker with small gaps."""
        if not segments:
            return segments

        # Group by speaker
        speaker_groups = {}
        for segment in segments:
            if segment.speaker_id not in speaker_groups:
                speaker_groups[segment.speaker_id] = []
            speaker_groups[segment.speaker_id].append(segment)

        # Merge consecutive segments from same speaker
        merged_segments = []

        for speaker_id, speaker_segments in speaker_groups.items():
            # Sort by start time
            speaker_segments.sort(key=lambda x: x.start_time)

            current_segment = None

            for segment in speaker_segments:
                if current_segment is None:
                    current_segment = segment
                else:
                    # Check if segments are close enough to merge
                    time_gap = segment.start_time - current_segment.end_time

                    if time_gap < gap_threshold:
                        # Merge audio data
                        gap_samples = int(
                            time_gap * self.config.target_sample_rate
                        )
                        gap_audio = np.zeros(gap_samples)
                        merged_audio = np.concatenate(
                            [
                                current_segment.audio_data,
                                gap_audio,
                                segment.audio_data,
                            ]
                        )

                        current_segment = SpeakerSegment(
                            start_time=current_segment.start_time,
                            end_time=segment.end_time,
                            speaker_id=speaker_id,
                            audio_data=merged_audio,
                            confidence=min(
                                current_segment.confidence, segment.confidence
                            ),
                        )
                    else:
                        # Add current segment and start new one
                        merged_segments.append(current_segment)
                        current_segment = segment

            # Add last segment
            if current_segment is not None:
                merged_segments.append(current_segment)

        # Sort by start time
        merged_segments.sort(key=lambda x: x.start_time)

        return merged_segments


def create_pyannote_diarizer(config: TranscriptionConfig) -> PyannoteDiarizer:
    """Create pyannote diarizer instance."""
    return PyannoteDiarizer(config)

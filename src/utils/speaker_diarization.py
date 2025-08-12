"""
Speaker Diarization for Persian Audio.
Separates different speakers to improve transcription quality.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import librosa
from scipy import signal
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

from ..core.config import TranscriptionConfig


@dataclass
class SpeakerSegment:
    """Represents a segment of audio from a specific speaker."""

    start_time: float
    end_time: float
    speaker_id: int
    audio_data: np.ndarray
    confidence: float
    features: Optional[np.ndarray] = None


class SpeakerDiarizer:
    """Advanced speaker diarization system for Persian audio."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.min_speaker_duration = (
            0.5  # Reduced from 2.0s to 0.5s for better detection
        )
        self.max_speakers = 8  # Maximum number of speakers to detect

    def diarize_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Perform speaker diarization on audio data.

        Optional hints can be provided for speaker counts.
        """
        try:
            self.logger.info("Starting speaker diarization...")

            # Step 1: Extract audio features
            features = self._extract_audio_features(audio_data, sample_rate)

            # Step 2: Detect speech segments
            speech_segments = self._detect_speech_segments(
                audio_data, sample_rate
            )

            # Step 3: Extract speaker embeddings
            speaker_embeddings = self._extract_speaker_embeddings(
                audio_data, sample_rate, speech_segments
            )

            # Step 4: Cluster speakers
            speaker_clusters = self._cluster_speakers(
                speaker_embeddings,
                forced_n=num_speakers,
                min_n=min_speakers,
                max_n=max_speakers,
            )

            # Step 5: Create speaker segments
            diarized_segments = self._create_speaker_segments(
                audio_data, sample_rate, speech_segments, speaker_clusters
            )

            self.logger.info(
                f"Diarization complete. Found {len(set(s.speaker_id for s in diarized_segments))} speakers"
            )
            return diarized_segments

        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            # Fallback: return single speaker segment
            return [
                SpeakerSegment(
                    start_time=0.0,
                    end_time=len(audio_data) / sample_rate,
                    speaker_id=0,
                    audio_data=audio_data,
                    confidence=1.0,
                )
            ]

    def _extract_audio_features(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Extract MFCC features for speaker analysis."""
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_data, sr=sample_rate, n_mfcc=13, hop_length=512, n_fft=2048
        )

        # Add delta and delta-delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Combine features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        return features.T  # Transpose to get time as first dimension

    def _detect_speech_segments(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> List[Tuple[float, float]]:
        """Detect segments containing speech."""
        # Energy-based voice activity detection
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Calculate energy
        energy = []
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i : i + frame_length]
            energy.append(np.sum(frame**2))

        energy = np.array(energy)

        # Normalize energy
        energy = (energy - np.mean(energy)) / np.std(energy)

        # Threshold for speech detection - more sensitive
        threshold = np.percentile(
            energy, 40
        )  # Reduced from 70 to 40 for more sensitive detection
        speech_frames = energy > threshold

        # Find speech segments
        segments = []
        start_frame = None

        for i, is_speech in enumerate(speech_frames):
            if is_speech and start_frame is None:
                start_frame = i
            elif not is_speech and start_frame is not None:
                end_frame = i
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate

                # Only keep segments longer than minimum duration
                if end_time - start_time >= self.min_speaker_duration:
                    segments.append((start_time, end_time))

                start_frame = None

        # Handle case where speech continues to end
        if start_frame is not None:
            end_time = len(audio_data) / sample_rate
            start_time = start_frame * hop_length / sample_rate
            if end_time - start_time >= self.min_speaker_duration:
                segments.append((start_time, end_time))

        return segments

    def _extract_speaker_embeddings(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speech_segments: List[Tuple[float, float]],
    ) -> List[np.ndarray]:
        """Extract speaker embeddings from speech segments."""
        embeddings = []

        for start_time, end_time in speech_segments:
            # Extract segment audio
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            if len(segment_audio) < sample_rate:  # Skip very short segments
                continue

            # Extract features for this segment
            features = self._extract_audio_features(segment_audio, sample_rate)

            # Calculate mean feature vector for the segment
            if len(features) > 0:
                mean_features = np.mean(features, axis=0)
                embeddings.append(mean_features)

        return embeddings

    def _cluster_speakers(
        self,
        embeddings: List[np.ndarray],
        forced_n: Optional[int] = None,
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
    ) -> List[int]:
        """Cluster speaker embeddings to identify different speakers."""
        if len(embeddings) < 2:
            return [0] * len(embeddings)

        # Convert to numpy array
        X = np.array(embeddings)

        # Forced number of clusters if provided
        if forced_n is not None and forced_n >= 1:
            n = min(max(1, forced_n), max(1, len(embeddings)))
            if n == 1:
                return [0] * len(embeddings)
            clustering = AgglomerativeClustering(n_clusters=n)
            return clustering.fit_predict(X).tolist()

        # Determine optimal number of clusters
        upper_bound = self.max_speakers
        if max_n is not None:
            upper_bound = min(upper_bound, max_n)
        max_clusters = min(upper_bound, len(embeddings) - 1)

        lower_bound = 1
        if min_n is not None:
            lower_bound = max(lower_bound, min_n)

        best_n_clusters = lower_bound
        best_score = -1

        for n_clusters in range(lower_bound, max_clusters + 1):
            if n_clusters == 1:
                score = 0
            else:
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = clustering.fit_predict(X)

                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(X, cluster_labels)
                else:
                    score = 0

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters

        # Perform final clustering
        if best_n_clusters <= 1:
            return [0] * len(embeddings)

        clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
        cluster_labels = clustering.fit_predict(X)

        return cluster_labels.tolist()

    def _create_speaker_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        speech_segments: List[Tuple[float, float]],
        speaker_clusters: List[int],
    ) -> List[SpeakerSegment]:
        """Create speaker segments from clustering results."""
        segments = []

        for i, (start_time, end_time) in enumerate(speech_segments):
            if i >= len(speaker_clusters):
                continue

            # Extract segment audio
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]

            # Create speaker segment
            segment = SpeakerSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_clusters[i],
                audio_data=segment_audio,
                confidence=0.8,  # Placeholder confidence
            )

            segments.append(segment)

        return segments

    def merge_similar_speakers(
        self, segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """Merge segments from similar speakers."""
        if not segments:
            return segments

        # Group segments by speaker
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

                    if (
                        time_gap < 3.0
                    ):  # Increased gap tolerance from 1.0s to 3.0s
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


def create_speaker_diarizer(config: TranscriptionConfig) -> SpeakerDiarizer:
    """Create speaker diarizer instance."""
    return SpeakerDiarizer(config)

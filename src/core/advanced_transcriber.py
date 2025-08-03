"""
Advanced Transcriber with 95% Quality Target.
Integrates all advanced features for maximum transcription quality.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path

from .config import TranscriptionConfig
from .transcriber import OptimizedWhisperTranscriber
from ..utils.enhanced_audio_preprocessor import EnhancedAudioPreprocessor
from ..utils.persian_text_postprocessor import PersianTextPostProcessor
from ..utils.advanced_model_ensemble import AdvancedModelEnsemble
from ..utils.speaker_diarization import SpeakerDiarizer, SpeakerSegment
from ..utils.quality_assessor import QualityAssessor, QualityMetrics


class AdvancedTranscriber:
    """Advanced transcriber with 95% quality target."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.audio_preprocessor = EnhancedAudioPreprocessor(config)
        self.text_postprocessor = PersianTextPostProcessor(config)
        
        # Initialize advanced components if enabled
        self.model_ensemble = None
        self.speaker_diarizer = None
        self.quality_assessor = None
        
        if config.enable_model_ensemble:
            self.model_ensemble = AdvancedModelEnsemble(config)
        
        if config.enable_speaker_diarization:
            self.speaker_diarizer = SpeakerDiarizer(config)
        
        if config.enable_quality_assessment:
            self.quality_assessor = QualityAssessor(config)
        
        # Fallback transcriber
        self.fallback_transcriber = OptimizedWhisperTranscriber(config)
    
    def transcribe_with_quality_optimization(self, audio_file: str) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio with iterative quality optimization."""
        self.logger.info("Starting advanced transcription with quality optimization...")
        
        # Step 1: Enhanced audio preprocessing
        self.logger.info("Step 1: Enhanced audio preprocessing...")
        processed_audio, audio_metadata = self.audio_preprocessor.preprocess_audio(audio_file)
        
        # Step 2: Speaker diarization (if enabled)
        speaker_segments = None
        if self.config.enable_speaker_diarization and self.speaker_diarizer:
            self.logger.info("Step 2: Speaker diarization...")
            speaker_segments = self.speaker_diarizer.diarize_audio(
                processed_audio, self.config.target_sample_rate
            )
            speaker_segments = self.speaker_diarizer.merge_similar_speakers(speaker_segments)
        
        # Step 3: Iterative transcription with quality optimization
        best_transcription = ""
        best_quality_score = 0.0
        best_metadata = {}
        current_config = self.config
        
        for iteration in range(self.config.max_optimization_iterations):
            self.logger.info(f"Step 3.{iteration + 1}: Transcription iteration {iteration + 1}")
            
            # Transcribe with current configuration
            transcription, metadata = self._transcribe_audio_segments(
                processed_audio, speaker_segments, current_config
            )
            
            # Assess quality
            if self.quality_assessor:
                quality_metrics = self.quality_assessor.assess_transcription_quality(
                    transcription, metadata
                )
                
                self.logger.info(f"Iteration {iteration + 1} quality score: {quality_metrics.overall_score:.3f}")
                
                # Update best result if improved
                if quality_metrics.overall_score > best_quality_score:
                    best_transcription = transcription
                    best_quality_score = quality_metrics.overall_score
                    best_metadata = metadata
                    best_metadata['quality_metrics'] = quality_metrics
                
                # Check if target quality achieved
                if quality_metrics.overall_score >= self.config.target_quality_threshold:
                    self.logger.info(f"Target quality ({self.config.target_quality_threshold}) achieved!")
                    break
                
                # Auto-tune parameters for next iteration
                if self.config.enable_auto_tuning and iteration < self.config.max_optimization_iterations - 1:
                    current_config = self.quality_assessor.auto_tune_parameters(
                        quality_metrics, current_config
                    )
            else:
                # No quality assessment, use current result
                best_transcription = transcription
                best_metadata = metadata
                break
        
        # Step 4: Final text post-processing
        if self.config.enable_text_postprocessing:
            self.logger.info("Step 4: Final text post-processing...")
            best_transcription, post_metadata = self.text_postprocessor.post_process_text(
                best_transcription, best_metadata
            )
            best_metadata.update(post_metadata)
        
        # Step 5: Final quality assessment
        if self.quality_assessor and best_metadata.get('quality_metrics'):
            final_metrics = self.quality_assessor.assess_transcription_quality(
                best_transcription, best_metadata
            )
            best_metadata['final_quality_metrics'] = final_metrics
            
            # Generate quality report
            quality_report = self.quality_assessor.get_quality_report(final_metrics)
            self.logger.info(quality_report)
        
        self.logger.info("Advanced transcription complete!")
        return best_transcription, best_metadata
    
    def _transcribe_audio_segments(self, audio_data: np.ndarray, 
                                 speaker_segments: Optional[List[SpeakerSegment]], 
                                 config: TranscriptionConfig) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio segments with speaker information."""
        if speaker_segments and len(speaker_segments) > 1:
            return self._transcribe_multi_speaker(audio_data, speaker_segments, config)
        else:
            return self._transcribe_single_speaker(audio_data, config)
    
    def _transcribe_multi_speaker(self, audio_data: np.ndarray, 
                                speaker_segments: List[SpeakerSegment], 
                                config: TranscriptionConfig) -> Tuple[str, Dict[str, Any]]:
        """Transcribe multi-speaker audio with speaker labels."""
        transcriptions = []
        metadata = {
            'speaker_count': len(set(s.speaker_id for s in speaker_segments)),
            'total_segments': len(speaker_segments),
            'individual_segments': []
        }
        
        for i, segment in enumerate(speaker_segments):
            self.logger.info(f"Transcribing speaker {segment.speaker_id} segment {i + 1}/{len(speaker_segments)}")
            
            # Transcribe segment
            if self.model_ensemble:
                segment_text, segment_metadata = self.model_ensemble.transcribe_ensemble(segment.audio_data)
            else:
                segment_text, segment_metadata = self.fallback_transcriber.transcribe_chunk(segment.audio_data)
            
            # Add speaker label
            speaker_label = f"[Speaker {segment.speaker_id}]: "
            segment_text = speaker_label + segment_text
            
            transcriptions.append(segment_text)
            
            # Store segment metadata
            segment_info = {
                'speaker_id': segment.speaker_id,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'confidence': segment.confidence,
                'text': segment_text,
                'metadata': segment_metadata
            }
            metadata['individual_segments'].append(segment_info)
        
        # Combine transcriptions
        full_transcription = "\n".join(transcriptions)
        
        return full_transcription, metadata
    
    def _transcribe_single_speaker(self, audio_data: np.ndarray, 
                                 config: TranscriptionConfig) -> Tuple[str, Dict[str, Any]]:
        """Transcribe single-speaker audio."""
        # Use model ensemble if available
        if self.model_ensemble:
            transcription, metadata = self.model_ensemble.transcribe_ensemble(audio_data)
        else:
            transcription, metadata = self.fallback_transcriber.transcribe_chunk(audio_data)
        
        metadata['speaker_count'] = 1
        metadata['total_segments'] = 1
        
        return transcription, metadata
    
    def transcribe_file(self, audio_file: str, output_file: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio file with full quality optimization."""
        start_time = time.time()
        
        try:
            # Perform advanced transcription
            transcription, metadata = self.transcribe_with_quality_optimization(audio_file)
            
            # Add timing information
            processing_time = time.time() - start_time
            metadata['processing_time'] = processing_time
            metadata['audio_file'] = audio_file
            
            # Save output if specified
            if output_file:
                self._save_transcription(transcription, metadata, output_file)
            
            self.logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            return transcription, metadata
            
        except Exception as e:
            self.logger.error(f"Advanced transcription failed: {e}")
            raise
    
    def _save_transcription(self, transcription: str, metadata: Dict[str, Any], output_file: str):
        """Save transcription and metadata to file."""
        output_path = Path(output_file)
        
        # Save main transcription
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Transcription saved to {output_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")


def create_advanced_transcriber(config: TranscriptionConfig) -> AdvancedTranscriber:
    """Create advanced transcriber instance."""
    return AdvancedTranscriber(config) 
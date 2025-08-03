"""
Advanced Model Ensemble for Persian Speech Recognition.
Combines multiple models for higher transcription accuracy.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    AutoProcessor, AutoModelForCTC
)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.config import TranscriptionConfig


class AdvancedModelEnsemble:
    """Advanced ensemble of multiple Persian speech recognition models."""
    
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.processors = {}
        self.weights = {}
        self._setup_models()
    
    def _setup_models(self):
        """Initialize multiple models for ensemble."""
        self._setup_hf_download_retry()
        
        # Primary model: Persian Whisper
        self._load_model("whisper", "nezamisafa/whisper-persian-v4", 0.4)
        
        # Secondary model: Wav2Vec2 Persian
        self._load_model("wav2vec2", "m3hrdadfi/wav2vec2-large-xlsr-persian-v3", 0.3)
        
        # Tertiary model: NVIDIA FastConformer (if available)
        try:
            self._load_model("fastconformer", "nvidia/stt_fa_fastconformer_hybrid_large", 0.3)
        except Exception as e:
            self.logger.warning(f"Could not load FastConformer model: {e}")
            # Adjust weights if FastConformer is not available
            self.weights["whisper"] = 0.6
            self.weights["wav2vec2"] = 0.4
    
    def _setup_hf_download_retry(self):
        """Setup retry strategy for Hugging Face downloads."""
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        self._hf_session = requests.Session()
        self._hf_session.mount("https://", adapter)
        self._hf_session.mount("http://", adapter)
        self._hf_session.timeout = (30, 300)
    
    def _load_model(self, name: str, model_id: str, weight: float):
        """Load a specific model."""
        try:
            self.logger.info(f"Loading {name} model: {model_id}")
            
            # Configure Hugging Face Hub
            import huggingface_hub
            try:
                huggingface_hub.set_http_backend(self._hf_session)
            except AttributeError:
                self.logger.info("Using default Hugging Face Hub HTTP backend")
            
            if "whisper" in model_id.lower():
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    local_files_only=False,
                    resume_download=True,
                )
                processor = WhisperProcessor.from_pretrained(
                    model_id,
                    local_files_only=False,
                    resume_download=True,
                )
            elif "wav2vec2" in model_id.lower():
                model = Wav2Vec2ForCTC.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    local_files_only=False,
                    resume_download=True,
                )
                processor = Wav2Vec2Processor.from_pretrained(
                    model_id,
                    local_files_only=False,
                    resume_download=True,
                )
            else:
                # Generic CTC model
                model = AutoModelForCTC.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                    device_map="auto" if self.config.device == "cuda" else None,
                    local_files_only=False,
                    resume_download=True,
                )
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    local_files_only=False,
                    resume_download=True,
                )
            
            self.models[name] = model
            self.processors[name] = processor
            self.weights[name] = weight
            
            self.logger.info(f"Successfully loaded {name} model")
            
        except Exception as e:
            self.logger.error(f"Failed to load {name} model: {e}")
            raise
    
    def transcribe_ensemble(self, audio_data: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """Transcribe audio using ensemble of models."""
        results = {}
        confidences = {}
        
        # Transcribe with each model
        for name, model in self.models.items():
            try:
                if "whisper" in name:
                    text, confidence = self._transcribe_whisper(name, audio_data)
                elif "wav2vec2" in name:
                    text, confidence = self._transcribe_wav2vec2(name, audio_data)
                else:
                    text, confidence = self._transcribe_generic(name, audio_data)
                
                results[name] = text
                confidences[name] = confidence
                
            except Exception as e:
                self.logger.error(f"Error transcribing with {name}: {e}")
                results[name] = ""
                confidences[name] = 0.0
        
        # Ensemble the results
        final_text, ensemble_confidence = self._ensemble_results(results, confidences)
        
        metadata = {
            'ensemble_confidence': ensemble_confidence,
            'individual_results': results,
            'individual_confidences': confidences,
            'weights': self.weights
        }
        
        return final_text, metadata
    
    def _transcribe_whisper(self, name: str, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe using Whisper model."""
        model = self.models[name]
        processor = self.processors[name]
        
        # Prepare input
        inputs = processor(
            audio_data, 
            sampling_rate=self.config.target_sample_rate, 
            return_tensors="pt"
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_features"],
                language="fa",
                task="transcribe",
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                num_beams=5,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode
        transcription = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        
        # Calculate confidence (simplified)
        confidence = 0.8  # Placeholder - would need logprobs for accurate confidence
        
        return transcription, confidence
    
    def _transcribe_wav2vec2(self, name: str, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe using Wav2Vec2 model."""
        model = self.models[name]
        processor = self.processors[name]
        
        # Prepare input
        inputs = processor(
            audio_data, 
            sampling_rate=self.config.target_sample_rate, 
            return_tensors="pt"
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0].mean().item()
        
        return transcription, confidence
    
    def _transcribe_generic(self, name: str, audio_data: np.ndarray) -> Tuple[str, float]:
        """Transcribe using generic CTC model."""
        model = self.models[name]
        processor = self.processors[name]
        
        # Prepare input
        inputs = processor(
            audio_data, 
            sampling_rate=self.config.target_sample_rate, 
            return_tensors="pt"
        )
        
        # Move to device
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0].mean().item()
        
        return transcription, confidence
    
    def _ensemble_results(self, results: Dict[str, str], confidences: Dict[str, float]) -> Tuple[str, float]:
        """Combine results from multiple models."""
        # Filter out empty results
        valid_results = {k: v for k, v in results.items() if v.strip()}
        
        if not valid_results:
            return "", 0.0
        
        if len(valid_results) == 1:
            name = list(valid_results.keys())[0]
            return valid_results[name], confidences[name]
        
        # Weighted voting based on confidence and model weights
        weighted_scores = {}
        
        for name, text in valid_results.items():
            weight = self.weights.get(name, 0.1)
            confidence = confidences.get(name, 0.5)
            combined_score = weight * confidence
            
            # Simple text similarity scoring
            if name in weighted_scores:
                weighted_scores[name] += combined_score
            else:
                weighted_scores[name] = combined_score
        
        # Select best result
        best_model = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        best_text = valid_results[best_model]
        ensemble_confidence = weighted_scores[best_model]
        
        return best_text, ensemble_confidence


def create_advanced_ensemble(config: TranscriptionConfig) -> AdvancedModelEnsemble:
    """Create advanced model ensemble."""
    return AdvancedModelEnsemble(config) 
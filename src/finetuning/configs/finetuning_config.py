"""
Fine-tuning configuration for Farsi voice transcription.
Comprehensive configuration management for all aspects of fine-tuning.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Configuration for the Whisper model."""
    
    # Base model settings
    base_model_name: str = "openai/whisper-base"
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "fa"  # Persian/Farsi
    task: str = "transcribe"
    
    # Model architecture
    use_peft: bool = True  # Parameter Efficient Fine-Tuning
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Quantization
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Model loading
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    
    # Farsi-specific settings
    enable_persian_optimization: bool = True
    persian_tokenizer_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training settings
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: Optional[int] = None
    
    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    
    # Training control
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_wer"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Farsi-specific training
    enable_persian_curriculum: bool = True
    persian_difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    train_data_path: str = ""
    validation_data_path: str = ""
    test_data_path: str = ""
    output_dir: str = ""
    
    # Data processing
    max_input_length: float = 30.0  # seconds
    min_input_length: float = 1.0   # seconds
    target_sample_rate: int = 16000
    audio_format: str = "wav"
    
    # Text processing
    max_text_length: int = 500
    min_text_length: int = 3
    normalize_text: bool = True
    remove_punctuation: bool = False
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_factor: int = 3
    augmentation_types: List[str] = field(default_factory=lambda: ["pitch", "speed", "noise"])
    
    # Quality control
    quality_threshold: float = 0.7
    min_audio_quality: float = 0.8
    min_text_quality: float = 0.9


@dataclass
class FineTuningConfig:
    """Main configuration class for fine-tuning."""
    
    # Configuration sections
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # General settings
    project_name: str = "farsi-whisper-finetuning"
    run_name: str = "farsi-voice-1"
    seed: int = 42
    dataloader_num_workers: int = 4
    
    # Logging and monitoring
    logging_dir: str = "./logs"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_id: Optional[str] = None
    
    # Checkpointing
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    save_safetensors: bool = True
    
    # Hardware optimization
    ddp_find_unused_parameters: bool = False
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        """Validate and optimize configuration after initialization."""
        self._validate_config()
        self._optimize_for_hardware()
        self._set_default_paths()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate model settings
        if self.model.model_size not in ["tiny", "base", "small", "medium", "large"]:
            raise ValueError(f"Invalid model size: {self.model.model_size}")
        
        # Validate training settings
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.training.num_train_epochs <= 0:
            raise ValueError("Number of training epochs must be positive")
        
        # Validate data settings
        if not self.data.train_data_path:
            raise ValueError("Training data path must be specified")
        
        if not self.data.validation_data_path:
            raise ValueError("Validation data path must be specified")
    
    def _optimize_for_hardware(self):
        """Optimize configuration based on available hardware."""
        # GPU memory optimization
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory < 8:  # Less than 8GB
                self.training.per_device_train_batch_size = 2
                self.training.gradient_accumulation_steps = 8
                self.training.fp16 = True
                self.model.use_4bit = True
                
            elif gpu_memory < 16:  # Less than 16GB
                self.training.per_device_train_batch_size = 4
                self.training.gradient_accumulation_steps = 4
                self.training.fp16 = True
                
            else:  # 16GB or more
                self.training.per_device_train_batch_size = 8
                self.training.gradient_accumulation_steps = 2
                self.training.fp16 = False
                self.training.bf16 = True
        
        # CPU optimization
        else:
            self.training.per_device_train_batch_size = 1
            self.training.gradient_accumulation_steps = 16
            self.training.fp16 = False
            self.training.bf16 = False
            self.dataloader_num_workers = min(4, os.cpu_count() or 2)
    
    def _set_default_paths(self):
        """Set default paths if not specified."""
        if not self.data.output_dir:
            self.data.output_dir = f"./output/{self.project_name}"
        
        if not self.logging_dir:
            self.logging_dir = f"./logs/{self.project_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "logging_dir": self.logging_dir,
            "report_to": self.report_to,
            "run_id": self.run_id,
            "save_strategy": self.save_strategy,
            "evaluation_strategy": self.evaluation_strategy,
            "save_safetensors": self.save_safetensors,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "remove_unused_columns": self.remove_unused_columns
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FineTuningConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ["model", "training", "data"]}
        
        # Create main config
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **main_config
        )
    
    def save_config(self, file_path: str):
        """Save configuration to file."""
        import json
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, file_path: str) -> "FineTuningConfig":
        """Load configuration from file."""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def clone(self) -> "FineTuningConfig":
        """Create a copy of the configuration."""
        return FineTuningConfig.from_dict(self.to_dict())
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.data, key):
                setattr(self.data, key, value)
    
    def get_huggingface_training_args(self) -> Dict[str, Any]:
        """Get configuration in HuggingFace TrainingArguments format."""
        return {
            "output_dir": self.data.output_dir,
            "num_train_epochs": self.training.num_train_epochs,
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "warmup_steps": self.training.warmup_steps,
            "max_steps": self.training.max_steps,
            "optim": self.training.optim,
            "lr_scheduler_type": self.training.lr_scheduler_type,
            "gradient_checkpointing": self.training.gradient_checkpointing,
            "fp16": self.training.fp16,
            "bf16": self.training.bf16,
            "save_steps": self.training.save_steps,
            "eval_steps": self.training.eval_steps,
            "logging_steps": self.training.logging_steps,
            "save_total_limit": self.training.save_total_limit,
            "load_best_model_at_end": self.training.load_best_model_at_end,
            "metric_for_best_model": self.training.metric_for_best_model,
            "greater_is_better": self.training.greater_is_better,
            "logging_dir": self.logging_dir,
            "report_to": self.report_to,
            "save_strategy": self.save_strategy,
            "evaluation_strategy": self.evaluation_strategy,
            "save_safetensors": self.save_safetensors,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "remove_unused_columns": self.remove_unused_columns,
            "ddp_find_unused_parameters": self.ddp_find_unused_parameters,
            "seed": self.seed
        }

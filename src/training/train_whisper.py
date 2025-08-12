"""
Fine-tuning script for Whisper on Persian/Farsi datasets.

Supports full fine-tuning or parameter-efficient fine-tuning (LoRA) using PEFT.
Minimal defaults chosen for stability; override via CLI.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from datasets import DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from .data_prep import DataPrepConfig, load_tabular, load_jsonl


@dataclass
class TrainConfig:
    model_name: str = "openai/whisper-small"
    language: str = "fa"
    task: str = "transcribe"
    sampling_rate: int = 16000
    max_steps: int = 2000
    lr: float = 1e-4
    batch_size: int = 8
    grad_accum: int = 2
    eval_steps: int = 200
    save_steps: int = 200
    output_dir: str = "./checkpoints"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    fp16: bool = True
    bf16: bool = False
    warmup_steps: int = 100
    logging_steps: int = 50
    freeze_encoder: bool = True


def prepare_datasets(
    data_path: str,
    data_format: str,
    cfg: DataPrepConfig,
) -> DatasetDict:
    if data_format == "csv" or data_format == "tsv":
        return load_tabular(data_path, cfg)
    elif data_format == "jsonl":
        return load_jsonl(data_path, cfg)
    else:
        raise ValueError("Unsupported data format. Use csv, tsv, or jsonl.")


def add_special_tokens(
    processor: WhisperProcessor, lang: str, task: str
) -> None:
    processor.tokenizer.set_prefix_tokens(language=lang, task=task)


def preprocess_batch(
    batch,
    processor: WhisperProcessor,
    cfg: TrainConfig,
    dp_cfg: DataPrepConfig,
):
    audio = batch[dp_cfg.path_column]
    text = batch[dp_cfg.text_column]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels = processor(text, return_tensors="pt").input_ids
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = labels[0]
    return batch


def build_model_and_processor(cfg: TrainConfig):
    processor = WhisperProcessor.from_pretrained(cfg.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)

    # Set generation language/task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.language = cfg.language
    model.config.task = cfg.task

    if cfg.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if cfg.use_lora and PEFT_AVAILABLE:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper for Persian/Farsi"
    )
    parser.add_argument(
        "data_path", help="Path to dataset file (csv/tsv/jsonl)"
    )
    parser.add_argument(
        "--data-format", choices=["csv", "tsv", "jsonl"], default="csv"
    )
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--language", default="fa")
    parser.add_argument(
        "--task", default="transcribe", choices=["transcribe", "translate"]
    )
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        model_name=args.model,
        language=args.language,
        task=args.task,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_lora=not args.no_lora,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    dp_cfg = DataPrepConfig(sampling_rate=16000)
    dsd = prepare_datasets(args.data_path, args.data_format, dp_cfg)

    model, processor = build_model_and_processor(train_cfg)
    add_special_tokens(processor, train_cfg.language, train_cfg.task)

    # Map preprocessing
    def _map_fn(batch):
        return preprocess_batch(batch, processor, train_cfg, dp_cfg)

    processed = DatasetDict()
    for split in dsd.keys():
        processed[split] = dsd[split].map(
            _map_fn,
            remove_columns=[
                c
                for c in dsd[split].column_names
                if c not in (dp_cfg.text_column, dp_cfg.path_column)
            ],
            batched=False,
        )

    data_collator = DataCollatorForSeq2Seq(processor.tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_cfg.output_dir,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=max(1, train_cfg.batch_size // 2),
        gradient_accumulation_steps=train_cfg.grad_accum,
        learning_rate=train_cfg.lr,
        logging_steps=train_cfg.logging_steps,
        evaluation_strategy="steps",
        eval_steps=train_cfg.eval_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=2,
        num_train_epochs=1,
        max_steps=train_cfg.max_steps,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        predict_with_generate=True,
        report_to=["none"],
        gradient_checkpointing=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed.get("train"),
        eval_dataset=processed.get("validation"),
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(train_cfg.output_dir)
    processor.save_pretrained(os.path.join(train_cfg.output_dir, "processor"))


if __name__ == "__main__":
    main()

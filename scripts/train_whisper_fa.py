#!/usr/bin/env python3
"""
Fine-tune Whisper (large-v3) on Persian/Farsi manifests.

Manifest format (TSV or CSV, tab recommended):
  audio_path\ttext\tdataset\tsplit\tuid
Only `audio_path` and `text` are required.

Example usage:
  python3 scripts/train_whisper_fa.py \
    --train-manifest data/raw/common_voice_fa/train.csv \
    --eval-manifest data/raw/common_voice_fa/validation.csv \
    --output-dir checkpoints/whisper-fa \
    --model-id openai/whisper-large-v3 \
    --lora --batch-size 8 --gradient-accumulation 4 --num-epochs 3

Dry run (no data needed):
  python3 scripts/train_whisper_fa.py --dry-run --model-id openai/whisper-large-v3
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import Dataset, Audio
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


@dataclass
class TrainConfig:
    train_manifest: Optional[Path]
    eval_manifest: Optional[Path]
    output_dir: Path
    model_id: str = "openai/whisper-large-v3"
    language: str = "fa"
    task: str = "transcribe"
    sampling_rate: int = 16000
    max_duration_s: Optional[float] = None
    min_duration_s: float = 0.0
    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = False
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    seed: int = 42
    logging_steps: int = 25
    eval_steps: int = 400
    save_steps: int = 800
    save_total_limit: int = 2
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    dry_run: bool = False


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Fine-tune Whisper on Persian manifests")
    p.add_argument("--train-manifest", type=Path, default=None)
    p.add_argument("--eval-manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/whisper-fa"))
    p.add_argument("--model-id", type=str, default="openai/whisper-large-v3")
    p.add_argument("--language", type=str, default="fa")
    p.add_argument("--task", type=str, default="transcribe")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--max-duration-s", type=float, default=None)
    p.add_argument("--min-duration-s", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gradient-accumulation", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--eval-steps", type=int, default=400)
    p.add_argument("--save-steps", type=int, default=800)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-token", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    return TrainConfig(
        train_manifest=args.train_manifest,
        eval_manifest=args.eval_manifest,
        output_dir=args.output_dir if args.output_dir.is_absolute() else Path(os.getcwd()) / args.output_dir,
        model_id=args.model_id,
        language=args.language,
        task=args.task,
        sampling_rate=args.sr,
        max_duration_s=args.max_duration_s,
        min_duration_s=args.min_duration_s,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        dry_run=args.dry_run,
    )


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".tab"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    required = {"audio_path", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {missing}")
    return df


def build_dataset(df: pd.DataFrame, sr: int) -> Dataset:
    # Ensure absolute paths
    df = df.copy()
    df["audio_path"] = df["audio_path"].apply(lambda p: str(Path(p).resolve()))

    def load_row(row):
        audio_path = row["audio_path"]
        text = str(row["text"]) if not pd.isna(row["text"]) else ""
        audio, rate = librosa.load(audio_path, sr=sr, mono=True)
        return {"audio": {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": sr}, "text": text}

    dataset = Dataset.from_pandas(df[["audio_path", "text"]])
    dataset = dataset.map(lambda r: load_row(r), remove_columns=["audio_path"], num_proc=1)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
    return dataset


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_inputs = [{"input_features": f["input_features"] if "input_features" in f else f["input_values"]} for f in features]
        label_inputs = [f["labels"] for f in features]
        batch = self.processor.feature_extractor.pad(audio_inputs, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_inputs},
            padding=True,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main() -> int:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {cfg.model_id}\nDevice: {device}\nOutput: {cfg.output_dir}")

    # Processor setup
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(cfg.model_id, language=cfg.language, task=cfg.task)
    feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
    tokenizer: WhisperTokenizer = processor.tokenizer

    # Forced decoder ids for fa transcription
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)

    if cfg.dry_run:
        print("Dry run: loaded processor and prepared config. Skipping model and training.")
        return 0

    # Model load
    model_kwargs: Dict = {"device_map": "auto" if device == "cuda" else None}
    if cfg.load_in_8bit:
        model_kwargs.update({"load_in_8bit": True})
    if cfg.load_in_4bit:
        model_kwargs.update({"load_in_4bit": True})
    model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(cfg.model_id, **model_kwargs)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    if cfg.lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT not installed but --lora was set. Install peft and bitsandbytes.")
        if cfg.load_in_8bit or cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if cfg.train_manifest is None or not cfg.train_manifest.exists():
        raise FileNotFoundError("--train-manifest is required and must exist")

    train_df = read_manifest(cfg.train_manifest)
    if cfg.max_duration_s is not None:
        # Duration filter requires reading audio durations; approximate via librosa.get_duration
        durations = train_df["audio_path"].apply(lambda p: librosa.get_duration(filename=str(Path(p).resolve())))
        mask = (durations >= cfg.min_duration_s) & (durations <= cfg.max_duration_s)
        train_df = train_df.loc[mask]

    eval_df = None
    if cfg.eval_manifest is not None and cfg.eval_manifest.exists():
        eval_df = read_manifest(cfg.eval_manifest)

    train_ds = build_dataset(train_df, cfg.sampling_rate)
    eval_ds = build_dataset(eval_df, cfg.sampling_rate) if eval_df is not None else None

    def prepare_batch(batch):
        audio = batch["audio"]
        input_features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        with processor.as_target_processor():
            labels = tokenizer(batch["text"]).input_ids
        return {"input_features": input_features, "labels": labels}

    train_ds = train_ds.map(prepare_batch, remove_columns=train_ds.column_names, num_proc=1)
    if eval_ds is not None:
        eval_ds = eval_ds.map(prepare_batch, remove_columns=eval_ds.column_names, num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * wer_metric.compute(predictions=preds, references=refs)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir.as_posix(),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_epochs,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps if eval_ds is not None else None,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        predict_with_generate=True,
        push_to_hub=cfg.push_to_hub,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir.as_posix())
    processor.save_pretrained(cfg.output_dir.as_posix())

    print("Training complete. Checkpoints saved to:", cfg.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




"""
Data preparation helpers for fine-tuning Whisper on Persian/Farsi datasets.

Supported input formats:
- TSV/CSV with columns: `audio_path`, `text` (and optional `speaker_id`, `split`)
- JSONL with fields: `audio_path`, `text`, `speaker_id`, `split`

Outputs a Hugging Face `DatasetDict` with train/validation splits and
audio columns decoded using `datasets.Audio` for on-the-fly resampling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
from datasets import Dataset, DatasetDict, Audio


@dataclass
class DataPrepConfig:
    sampling_rate: int = 16000
    train_ratio: float = 0.98
    text_column: str = "text"
    path_column: str = "audio_path"
    split_column: str = "split"  # optional; values like train/validation/test


def _ensure_columns(df: pd.DataFrame, path_col: str, text_col: str) -> None:
    missing = [c for c in (path_col, text_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_tabular(
    file_path: str,
    config: Optional[DataPrepConfig] = None,
) -> DatasetDict:
    config = config or DataPrepConfig()
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in {".csv", ".tsv"}:
        raise ValueError("Expected a .csv or .tsv file")

    df = pd.read_csv(file_path, sep="\t" if ext == ".tsv" else ",")
    _ensure_columns(df, config.path_column, config.text_column)

    if config.split_column in df.columns:
        splits = {}
        for split_name in sorted(df[config.split_column].dropna().unique()):
            part = df[df[config.split_column] == split_name]
            ds = Dataset.from_pandas(part.reset_index(drop=True))
            splits[split_name] = ds
        dsd = DatasetDict(splits)
    else:
        # simple split
        n = len(df)
        n_train = max(1, int(n * config.train_ratio))
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:].reset_index(drop=True)
        dsd = DatasetDict(
            train=Dataset.from_pandas(train_df),
            validation=Dataset.from_pandas(val_df),
        )

    # Cast audio
    for k in dsd.keys():
        dsd[k] = dsd[k].cast_column(config.path_column, Audio(sampling_rate=config.sampling_rate))
    return dsd


def load_jsonl(
    file_path: str,
    config: Optional[DataPrepConfig] = None,
) -> DatasetDict:
    config = config or DataPrepConfig()
    import json

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    _ensure_columns(df, config.path_column, config.text_column)

    if config.split_column in df.columns:
        splits = {}
        for split_name in sorted(df[config.split_column].dropna().unique()):
            part = df[df[config.split_column] == split_name]
            ds = Dataset.from_pandas(part.reset_index(drop=True))
            splits[split_name] = ds
        dsd = DatasetDict(splits)
    else:
        n = len(df)
        n_train = max(1, int(n * config.train_ratio))
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:].reset_index(drop=True)
        dsd = DatasetDict(
            train=Dataset.from_pandas(train_df),
            validation=Dataset.from_pandas(val_df),
        )

    for k in dsd.keys():
        dsd[k] = dsd[k].cast_column(config.path_column, Audio(sampling_rate=config.sampling_rate))
    return dsd


def prepare_commonvoice_like(
    root_dir: str,
    train_tsv: str,
    dev_tsv: Optional[str] = None,
    config: Optional[DataPrepConfig] = None,
) -> DatasetDict:
    config = config or DataPrepConfig()
    train_path = os.path.join(root_dir, train_tsv)
    dev_path = os.path.join(root_dir, dev_tsv) if dev_tsv else None

    train_df = pd.read_csv(train_path, sep="\t")
    _ensure_columns(train_df, config.path_column, config.text_column)
    train_df[config.path_column] = train_df[config.path_column].apply(lambda p: os.path.join(root_dir, p))

    dsd = DatasetDict(train=Dataset.from_pandas(train_df.reset_index(drop=True)))
    if dev_path and os.path.exists(dev_path):
        dev_df = pd.read_csv(dev_path, sep="\t")
        _ensure_columns(dev_df, config.path_column, config.text_column)
        dev_df[config.path_column] = dev_df[config.path_column].apply(lambda p: os.path.join(root_dir, p))
        dsd["validation"] = Dataset.from_pandas(dev_df.reset_index(drop=True))

    for k in dsd.keys():
        dsd[k] = dsd[k].cast_column(config.path_column, Audio(sampling_rate=config.sampling_rate))
    return dsd



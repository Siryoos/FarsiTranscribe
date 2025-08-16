#!/usr/bin/env python3
"""
Download and materialize Persian/Farsi ASR datasets locally.

Supported datasets:
- common_voice_fa: mozilla-foundation/common_voice_17_0 (config: fa)
- fleurs_fa_ir: google/fleurs (config: fa_ir)

Outputs:
- Audio saved as 16kHz mono WAV under data/raw/<dataset>/audio/<split>/<id>.wav
- Manifest CSV per split under data/raw/<dataset>/<split>.csv with columns:
  [audio_path, text, dataset, split, uid]
- Updates data/raw/dataset_info.json with simple bookkeeping

Usage examples:
  python scripts/download_farsi_asr.py --datasets common_voice_fa fleurs_fa_ir \
      --output-dir data/raw --max-per-split 100

  python scripts/download_farsi_asr.py --datasets common_voice_fa --full
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import soundfile as sf
from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import snapshot_download
import librosa
from tqdm import tqdm


DATASET_REGISTRY = {
    "common_voice_fa": {
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "config": "fa",
        "splits": ["train", "validation", "test"],
        "text_field": "sentence",
        "audio_field": "audio",
        "id_field": "client_id",  # fallback to index if missing
        "license": "CC0-1.0",
        "citation": "Ardila et al. (2020). Common Voice: A Massively-Multilingual Speech Corpus.",
        "estimated_size_mb": 2500,
    },
    "fleurs_fa_ir": {
        "hf_id": "google/fleurs",
        "config": "fa_ir",
        "splits": ["train", "validation", "test"],
        # In FLEURS, the transcription columns are often `raw_transcription` and `transcription`.
        # Prefer `raw_transcription` if present, else fall back to `transcription`.
        "text_field": ["raw_transcription", "transcription"],
        "audio_field": "audio",
        "id_field": "id",  # numeric id
        "license": "CC-BY-4.0",
        "citation": "Conneau et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.",
        "estimated_size_mb": 600,  # rough per-language estimate
    },
}


@dataclass
class DownloadConfig:
    dataset_keys: List[str]
    output_dir: Path
    sampling_rate: int = 16000
    mono: bool = True
    max_per_split: Optional[int] = None
    full: bool = False
    hf_token: Optional[str] = None
    use_auth: bool = False
    force_redownload: bool = False
    cache_dir: Optional[Path] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_text(example: dict, field_spec) -> Optional[str]:
    if isinstance(field_spec, list):
        for f in field_spec:
            if f in example and example[f] is not None and str(example[f]).strip():
                return str(example[f]).strip()
        return None
    return str(example.get(field_spec, "")).strip() or None


def materialize_split(
    ds_name: str,
    split_name: str,
    dataset_dict: DatasetDict,
    text_field_spec,
    audio_field: str,
    id_field: Optional[str],
    out_root: Path,
    sampling_rate: int,
    mono: bool,
    max_records: Optional[int] = None,
) -> Tuple[int, int]:
    """Save audio to WAV and write manifest CSV. Returns (num_ok, num_skipped)."""
    if split_name not in dataset_dict:
        return 0, 0

    split = dataset_dict[split_name]
    split = split.cast_column(audio_field, Audio(sampling_rate=sampling_rate))

    audio_dir = out_root / "audio" / split_name
    ensure_dir(audio_dir)

    manifest_path = out_root / f"{split_name}.csv"
    written = 0
    skipped = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        mf.write("audio_path\ttext\tdataset\tsplit\tuid\n")

        iterable: Iterable = split
        if max_records is not None:
            iterable = iterable.select(range(min(max_records, len(split))))

        for idx, example in enumerate(tqdm(iterable, desc=f"{ds_name}:{split_name}")):
            try:
                audio = example[audio_field]
                text = select_text(example, text_field_spec)
                if not text:
                    skipped += 1
                    continue

                uid = None
                if id_field and id_field in example:
                    uid = str(example[id_field])
                if not uid:
                    uid = str(example.get("id", idx))

                # audio["array"] is float32 mono; if stereo is present, average to mono
                data = audio["array"]
                sr = audio["sampling_rate"]
                # Resampling is handled by datasets cast_column, but validate
                if sr != sampling_rate:
                    skipped += 1
                    continue

                # Force mono if necessary
                if data.ndim == 2:
                    # average channels
                    data = data.mean(axis=1)

                out_wav = audio_dir / f"{uid}.wav"
                sf.write(out_wav.as_posix(), data, sampling_rate, subtype="PCM_16")

                mf.write(
                    f"{out_wav.as_posix()}\t{text}\t{ds_name}\t{split_name}\t{uid}\n"
                )
                written += 1
            except Exception:
                skipped += 1
                continue

    return written, skipped


def update_dataset_info(info_path: Path, updates: Dict[str, dict]) -> None:
    record = {
        "download_date": None,
        "total_datasets": 0,
        "datasets": {},
    }
    if info_path.exists():
        try:
            record = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    from datetime import datetime

    record["download_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record.setdefault("datasets", {})

    for k, v in updates.items():
        record["datasets"][k] = v

    record["total_datasets"] = len(record["datasets"])
    ensure_dir(info_path.parent)
    info_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def download_dataset(cfg: DownloadConfig, key: str) -> None:
    meta = DATASET_REGISTRY[key]
    ds_id = meta["hf_id"]
    config = meta.get("config")
    splits = meta["splits"]
    text_field = meta["text_field"]
    audio_field = meta["audio_field"]
    id_field = meta.get("id_field")

    out_root = cfg.output_dir / key
    ensure_dir(out_root)

    print(f"\n==> Loading {key}: {ds_id} (config={config})")
    dataset_dict = None
    load_kwargs = {"trust_remote_code": True}
    # Prefer explicit token if provided; else use cached auth if requested; else anonymous
    if cfg.hf_token:
        # Explicit token string provided or via env var
        load_kwargs["token"] = cfg.hf_token
    elif cfg.use_auth:
        # Use cached login (huggingface-cli login)
        load_kwargs["token"] = True

    try:
        if cfg.force_redownload:
            load_kwargs["download_mode"] = "force_redownload"
        if cfg.cache_dir is not None:
            load_kwargs["cache_dir"] = cfg.cache_dir.as_posix()
        dataset_dict = load_dataset(ds_id, config, **load_kwargs)
    except Exception as e:
        msg = str(e)
        # If common voice fails due to gating or remote code policy, fallback to snapshot
        if key == "common_voice_fa":
            print("\nℹ️ Falling back to snapshot download for Common Voice (fa)...")
            download_common_voice_via_snapshot(cfg, key, meta)
            return
        # Helpful guidance for common 403 cases
        if "403" in msg or "Forbidden" in msg or "gated" in msg:
            print("\n❌ Access forbidden (403) or gated dataset.")
            print(
                "Visit the dataset page to request access, then set a token:\n"
                "  export HUGGINGFACE_HUB_TOKEN=hf_...\n"
                "  huggingface-cli login\n"
                "Re-run with --use-auth or --hf-token.\n"
                "Note: This dataset requires executing remote code; we pass trust_remote_code=True."
            )
        raise

    totals = {"written": 0, "skipped": 0}
    for split in splits:
        w, s = materialize_split(
            ds_name=key,
            split_name=split,
            dataset_dict=dataset_dict,
            text_field_spec=text_field,
            audio_field=audio_field,
            id_field=id_field,
            out_root=out_root,
            sampling_rate=cfg.sampling_rate,
            mono=cfg.mono,
            max_records=None if cfg.full else cfg.max_per_split,
        )
        totals["written"] += w
        totals["skipped"] += s

    # Update dataset info
    info_updates = {
        key: {
            "name": key,
            "description": f"Materialized from {ds_id} ({config})",
            "license": meta.get("license"),
            "citation": meta.get("citation"),
            "downloaded": True,
            "path": out_root.as_posix(),
            "size_mb": meta.get("estimated_size_mb"),
            "num_items": totals["written"],
        }
    }
    update_dataset_info(cfg.output_dir / "dataset_info.json", info_updates)

    print(
        f"Completed {key}: {totals['written']} items written, {totals['skipped']} skipped."
    )


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(description="Download Persian ASR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        required=True,
        help="Datasets to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory to place raw datasets",
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Limit number of records per split (useful for sanity checks)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore --max-per-split and download full splits",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sampling rate for WAV output",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token to use for gated or rate-limited datasets",
    )
    parser.add_argument(
        "--use-auth",
        action="store_true",
        help="Use cached Hugging Face login (huggingface-cli login)",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download to bypass possibly corrupted cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hf_cache"),
        help="Custom cache directory to avoid user-level cache corruption",
    )

    args = parser.parse_args()
    return DownloadConfig(
        dataset_keys=args.datasets,
        output_dir=args.output_dir if args.output_dir.is_absolute() else Path(os.getcwd()) / args.output_dir,
        sampling_rate=args.sr,
        mono=True,
        max_per_split=args.max_per_split,
        full=args.full,
        hf_token=args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        use_auth=args.use_auth,
        force_redownload=args.force_redownload,
        cache_dir=(args.cache_dir if args.cache_dir.is_absolute() else Path(os.getcwd()) / args.cache_dir),
    )


def download_common_voice_via_snapshot(cfg: DownloadConfig, key: str, meta: dict) -> None:
    """Fallback path to download Common Voice fa via snapshot and materialize locally."""
    repo_id = meta["hf_id"]
    lang = meta.get("config", "fa")
    token = cfg.hf_token if cfg.hf_token else (True if cfg.use_auth else None)

    print(f"Downloading snapshot of {repo_id} (language: {lang})... This can be large.")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[
            f"{lang}/*",
            f"{lang}/**",
            "fa/*",
            "fa/**",
            "README*",
            "LICENSE*",
        ],
        revision="main",
        token=token,
        local_dir_use_symlinks=False,
    )

    lang_dir = Path(local_dir) / lang
    # Map possible split files
    split_files = {
        "train": ["train.tsv"],
        "validation": ["validation.tsv", "dev.tsv", "validated.tsv"],
        "test": ["test.tsv"],
    }

    out_root = cfg.output_dir / key
    ensure_dir(out_root)

    totals = {"written": 0, "skipped": 0}
    for split, candidates in split_files.items():
        tsv_path = None
        for cand in candidates:
            candidate_path = lang_dir / cand
            if candidate_path.exists():
                tsv_path = candidate_path
                break
        if tsv_path is None:
            continue

        audio_dir = out_root / "audio" / split
        ensure_dir(audio_dir)
        manifest_path = out_root / f"{split}.csv"
        written = 0
        skipped = 0

        with tsv_path.open("r", encoding="utf-8") as f_in, manifest_path.open(
            "w", encoding="utf-8"
        ) as mf:
            reader = csv.DictReader(f_in, delimiter="\t")
            mf.write("audio_path\ttext\tdataset\tsplit\tuid\n")

            for idx, row in enumerate(reader):
                if not cfg.full and cfg.max_per_split is not None and idx >= cfg.max_per_split:
                    break
                try:
                    path_rel = row.get("path")
                    text = (row.get("sentence") or "").strip()
                    uid = (row.get("client_id") or row.get("id") or str(idx)).strip()
                    if not path_rel or not text:
                        skipped += 1
                        continue

                    mp3_path = lang_dir / "clips" / path_rel
                    if not mp3_path.exists():
                        skipped += 1
                        continue

                    # Load and resample to target SR mono
                    data, sr = librosa.load(mp3_path.as_posix(), sr=cfg.sampling_rate, mono=True)
                    out_wav = audio_dir / f"{uid}.wav"
                    sf.write(out_wav.as_posix(), data, cfg.sampling_rate, subtype="PCM_16")

                    mf.write(
                        f"{out_wav.as_posix()}\t{text}\t{key}\t{split}\t{uid}\n"
                    )
                    written += 1
                except Exception:
                    skipped += 1
                    continue

        totals["written"] += written
        totals["skipped"] += skipped

    info_updates = {
        key: {
            "name": key,
            "description": f"Materialized from snapshot {repo_id} ({lang})",
            "license": meta.get("license"),
            "citation": meta.get("citation"),
            "downloaded": True,
            "path": (cfg.output_dir / key).as_posix(),
            "size_mb": meta.get("estimated_size_mb"),
            "num_items": totals["written"],
        }
    }
    update_dataset_info(cfg.output_dir / "dataset_info.json", info_updates)
    print(
        f"Completed {key} via snapshot: {totals['written']} items written, {totals['skipped']} skipped."
    )


def main() -> int:
    cfg = parse_args()

    for key in cfg.dataset_keys:
        download_dataset(cfg, key)

    print("\nAll requested datasets processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



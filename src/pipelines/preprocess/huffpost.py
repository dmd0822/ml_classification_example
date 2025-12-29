"""HuffPost dataset preprocessing.

This module reads the raw HuffPost JSONL dataset, derives a combined `text`
column using the configured separator, applies basic cleaning rules, and writes
train/valid/test splits as deterministic artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipelines.common.text import combine_text, normalize_whitespace


@dataclass(frozen=True)
class PreprocessOutputs:
    """Paths to preprocessing outputs."""

    full_path: Path
    train_path: Path
    valid_path: Path
    test_path: Path
    manifest_path: Path


"""NOTE: Text assembly helpers live in `src.pipelines.common.text`.

They are imported here to guarantee training/inference parity.
"""


def _ensure_required_columns(df: pd.DataFrame, required: Tuple[str, ...]) -> None:
    """Fail fast if expected columns are missing."""

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_text_column(df: pd.DataFrame, schema_cfg: Mapping[str, Any], text_cfg: Mapping[str, Any]) -> pd.Series:
    """Create the derived `text` column from headline and short description."""

    headline_col = str(schema_cfg["headline_col"])
    description_col = str(schema_cfg["description_col"])

    headline = df[headline_col].fillna(text_cfg.get("null_headline", "")).astype("string")
    desc = df[description_col].fillna(text_cfg.get("null_description", "")).astype("string")

    separator = str(text_cfg.get("separator", " "))
    normalize = bool(text_cfg.get("normalize_whitespace", True))
    strip = bool(text_cfg.get("strip", True))

    # Use vectorized concat for speed, then apply normalization deterministically.
    combined = headline.str.cat(desc, sep=separator).fillna("")

    if normalize:
        combined = combined.map(lambda s: normalize_whitespace(str(s)))

    if strip:
        combined = combined.str.strip()

    return combined


def clean_huffpost_dataframe(
    df_raw: pd.DataFrame,
    *,
    schema_cfg: Mapping[str, Any],
    text_cfg: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply minimal cleaning rules and derive `text`.

    Returns
    -------
    (df_clean, stats)
        df_clean is ready for splitting and feature extraction.
        stats contains counts of dropped rows for auditability.
    """

    label_col = str(schema_cfg["label_col"])
    headline_col = str(schema_cfg["headline_col"])
    description_col = str(schema_cfg["description_col"])
    text_col = str(schema_cfg["text_col"])

    _ensure_required_columns(df_raw, (label_col, headline_col, description_col))

    df = df_raw.copy()

    # Add a stable row id so splits/predictions can be joined later.
    df.insert(0, "row_id", pd.RangeIndex(start=0, stop=len(df), step=1, name="row_id"))

    # Ensure string dtypes for text columns; keep other columns as-is.
    for col in (label_col, headline_col, description_col):
        df[col] = df[col].astype("string")

    # Derive combined text.
    df[text_col] = build_text_column(df, schema_cfg, text_cfg)

    dropped_missing_category = 0
    dropped_empty_text = 0
    dropped_rare_labels = 0
    dropped_duplicates = 0

    # Drop rows with missing/empty label if configured.
    if bool(preprocess_cfg.get("drop_missing_category", True)):
        before = len(df)
        df = df[df[label_col].notna() & (df[label_col].astype("string").str.strip() != "")].copy()
        dropped_missing_category = before - len(df)

    # Drop empty/too-short text if configured.
    if bool(text_cfg.get("drop_empty_text", True)):
        min_len = int(text_cfg.get("min_text_length", 1))
        before = len(df)
        df = df[df[text_col].fillna("").astype("string").str.len() >= min_len].copy()
        dropped_empty_text = before - len(df)

    # Optional: de-duplicate on combined text.
    if bool(preprocess_cfg.get("dedupe_on_text", False)):
        before = len(df)
        df = df.drop_duplicates(subset=[text_col], keep="first").copy()
        dropped_duplicates = before - len(df)

    # Ensure stratified splitting can succeed by dropping labels with too few samples.
    min_count = int(preprocess_cfg.get("drop_rare_labels_min_count", 2))
    if min_count > 1:
        counts = df[label_col].value_counts(dropna=False)
        rare_labels = counts[counts < min_count].index.tolist()
        if rare_labels:
            before = len(df)
            df = df[~df[label_col].isin(rare_labels)].copy()
            dropped_rare_labels = before - len(df)

    stats = {
        "rows_in": int(len(df_raw)),
        "rows_out": int(len(df)),
        "dropped_missing_category": int(dropped_missing_category),
        "dropped_empty_text": int(dropped_empty_text),
        "dropped_duplicates": int(dropped_duplicates),
        "dropped_rare_labels": int(dropped_rare_labels),
        "label_cardinality_out": int(df[label_col].nunique(dropna=True)),
    }

    return df, stats


def make_splits(
    df: pd.DataFrame,
    *,
    label_col: str,
    random_seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    stratified: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic train/valid/test splits."""

    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total}).")

    stratify_y = df[label_col] if stratified else None

    # First split off test.
    train_valid, test = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_y,
    )

    # Then split train vs valid from the remainder.
    remaining = 1.0 - test_ratio
    valid_size_relative = valid_ratio / remaining
    stratify_tv = train_valid[label_col] if stratified else None

    train, valid = train_test_split(
        train_valid,
        test_size=valid_size_relative,
        random_state=random_seed,
        stratify=stratify_tv,
    )

    return train, valid, test


def _write_csv(df: pd.DataFrame, path: Path, *, force: bool) -> None:
    """Write a CSV file safely."""

    # Keep outputs reproducible and avoid accidental overwrites.
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists: {path}. Use --force to overwrite.")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def preprocess_huffpost(config: Mapping[str, Any], *, force: bool = False) -> PreprocessOutputs:
    """Run the full HuffPost preprocessing pipeline driven by config."""

    schema_cfg = config["schema"]
    paths_cfg = config["paths"]
    text_cfg = config["text_assembly"]
    preprocess_cfg = config.get("preprocess", {})
    split_cfg = config["split"]

    raw_path = Path(str(paths_cfg["raw_dir"])) / str(paths_cfg["raw_filename"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df_raw = pd.read_json(raw_path, lines=True)

    df_clean, stats = clean_huffpost_dataframe(
        df_raw,
        schema_cfg=schema_cfg,
        text_cfg=text_cfg,
        preprocess_cfg=preprocess_cfg,
    )

    label_col = str(schema_cfg["label_col"])
    stratified = str(split_cfg.get("strategy", "stratified")).lower() == "stratified"

    train, valid, test = make_splits(
        df_clean,
        label_col=label_col,
        random_seed=int(split_cfg["random_seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        valid_ratio=float(split_cfg["valid_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        stratified=stratified,
    )

    out_dir = Path(str(paths_cfg["preprocessed_dir"]))
    full_path = out_dir / str(paths_cfg["preprocessed_full_filename"])
    train_path = out_dir / str(paths_cfg["preprocessed_train_filename"])
    valid_path = out_dir / str(paths_cfg["preprocessed_valid_filename"])
    test_path = out_dir / str(paths_cfg["preprocessed_test_filename"])
    manifest_path = out_dir / "preprocess.manifest.json"

    _write_csv(df_clean, full_path, force=force)
    _write_csv(train, train_path, force=force)
    _write_csv(valid, valid_path, force=force)
    _write_csv(test, test_path, force=force)

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"raw_path": str(raw_path.as_posix())},
        "outputs": {
            "full": str(full_path.as_posix()),
            "train": str(train_path.as_posix()),
            "valid": str(valid_path.as_posix()),
            "test": str(test_path.as_posix()),
        },
        "stats": stats,
        "split": {
            "strategy": split_cfg.get("strategy"),
            "train_ratio": split_cfg.get("train_ratio"),
            "valid_ratio": split_cfg.get("valid_ratio"),
            "test_ratio": split_cfg.get("test_ratio"),
            "random_seed": split_cfg.get("random_seed"),
        },
        "schema": {
            "label_col": schema_cfg.get("label_col"),
            "headline_col": schema_cfg.get("headline_col"),
            "description_col": schema_cfg.get("description_col"),
            "text_col": schema_cfg.get("text_col"),
        },
        "text_assembly": {
            "separator": text_cfg.get("separator"),
            "strip": text_cfg.get("strip"),
            "normalize_whitespace": text_cfg.get("normalize_whitespace"),
            "drop_empty_text": text_cfg.get("drop_empty_text"),
            "min_text_length": text_cfg.get("min_text_length"),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return PreprocessOutputs(
        full_path=full_path,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        manifest_path=manifest_path,
    )

"""TF-IDF feature extraction for HuffPost classification.

This pipeline:
- Reads preprocessed train/valid/test CSVs.
- Fits a TF-IDF vectorizer on *train only* (to avoid leakage).
- Transforms train/valid/test into sparse feature matrices.
- Encodes labels into integer IDs via LabelEncoder.
- Writes artifacts (vectorizer, encoder, X/y splits, manifest).

Artifacts are written under `data/03-features/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class FeatureOutputs:
    """Paths to feature extraction outputs."""

    vectorizer_path: Path
    label_encoder_path: Path
    x_train_path: Path
    x_valid_path: Path
    x_test_path: Path
    y_train_path: Path
    y_valid_path: Path
    y_test_path: Path
    manifest_path: Path


def _parse_ngram_range(value: Any) -> Tuple[int, int]:
    """Parse ngram_range from config."""

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError("features.tfidf.ngram_range must be a 2-item list")


def build_tfidf_vectorizer(tfidf_cfg: Mapping[str, Any]) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer from config."""

    ngram_range = _parse_ngram_range(tfidf_cfg.get("ngram_range", [1, 1]))

    dtype_cfg = str(tfidf_cfg.get("dtype", "float32")).lower()
    dtype = np.float32 if dtype_cfg == "float32" else np.float64

    return TfidfVectorizer(
        max_features=int(tfidf_cfg.get("max_features", 50000)),
        ngram_range=ngram_range,
        min_df=tfidf_cfg.get("min_df", 1),
        max_df=tfidf_cfg.get("max_df", 1.0),
        lowercase=bool(tfidf_cfg.get("lowercase", True)),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
        dtype=dtype,
    )


def load_preprocessed_splits(paths_cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed CSV splits (train/valid/test)."""

    pre_dir = Path(str(paths_cfg["preprocessed_dir"]))
    train_path = pre_dir / str(paths_cfg["preprocessed_train_filename"])
    valid_path = pre_dir / str(paths_cfg["preprocessed_valid_filename"])
    test_path = pre_dir / str(paths_cfg["preprocessed_test_filename"])

    if not train_path.exists():
        raise FileNotFoundError(f"Missing preprocessed train split: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing preprocessed valid split: {valid_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing preprocessed test split: {test_path}")

    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_valid, df_test


def _ensure_columns(df: pd.DataFrame, required: Tuple[str, ...], *, split_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {missing}")


def extract_tfidf_features_huffpost(config: Mapping[str, Any], *, force: bool = False) -> FeatureOutputs:
    """Run TF-IDF feature extraction driven by config."""

    schema_cfg = config["schema"]
    paths_cfg = config["paths"]

    features_cfg = config.get("features", {})
    if str(features_cfg.get("type", "tfidf")).lower() != "tfidf":
        raise ValueError("This pipeline only supports features.type == 'tfidf'.")

    tfidf_cfg = features_cfg.get("tfidf", {})

    label_col = str(schema_cfg["label_col"])
    text_col = str(schema_cfg["text_col"])

    df_train, df_valid, df_test = load_preprocessed_splits(paths_cfg)

    required = (label_col, text_col)
    _ensure_columns(df_train, required, split_name="train")
    _ensure_columns(df_valid, required, split_name="valid")
    _ensure_columns(df_test, required, split_name="test")

    # Fit on train only (no leakage).
    vectorizer = build_tfidf_vectorizer(tfidf_cfg)

    train_text = df_train[text_col].fillna("").astype("string")
    valid_text = df_valid[text_col].fillna("").astype("string")
    test_text = df_test[text_col].fillna("").astype("string")

    x_train = vectorizer.fit_transform(train_text)
    x_valid = vectorizer.transform(valid_text)
    x_test = vectorizer.transform(test_text)

    # Label encoding: fit on train labels only.
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(
        df_train[label_col].fillna("").astype("string")
    )
    y_valid = label_encoder.transform(
        df_valid[label_col].fillna("").astype("string")
    )
    y_test = label_encoder.transform(
        df_test[label_col].fillna("").astype("string")
    )

    out_dir = Path(str(paths_cfg["features_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = out_dir / str(paths_cfg["features_vectorizer_filename"])
    label_encoder_path = out_dir / str(paths_cfg["features_label_encoder_filename"])
    x_train_path = out_dir / str(paths_cfg["features_x_train_filename"])
    x_valid_path = out_dir / str(paths_cfg["features_x_valid_filename"])
    x_test_path = out_dir / str(paths_cfg["features_x_test_filename"])
    y_train_path = out_dir / str(paths_cfg["features_y_train_filename"])
    y_valid_path = out_dir / str(paths_cfg["features_y_valid_filename"])
    y_test_path = out_dir / str(paths_cfg["features_y_test_filename"])
    manifest_path = out_dir / "features.manifest.json"

    # Avoid accidental overwrites unless force=True.
    outputs = [
        vectorizer_path,
        label_encoder_path,
        x_train_path,
        x_valid_path,
        x_test_path,
        y_train_path,
        y_valid_path,
        y_test_path,
        manifest_path,
    ]
    if not force:
        existing = [p for p in outputs if p.exists()]
        if existing:
            raise FileExistsError(
                "Feature outputs already exist. Use --force to overwrite: "
                + ", ".join(str(p) for p in existing)
            )

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, label_encoder_path)

    sp.save_npz(x_train_path, x_train)
    sp.save_npz(x_valid_path, x_valid)
    sp.save_npz(x_test_path, x_test)

    np.save(y_train_path, y_train)
    np.save(y_valid_path, y_valid)
    np.save(y_test_path, y_test)

    manifest: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preprocessed_dir": str(Path(str(paths_cfg["preprocessed_dir"])).as_posix()),
        },
        "outputs": {
            "features_dir": str(out_dir.as_posix()),
            "vectorizer": str(vectorizer_path.as_posix()),
            "label_encoder": str(label_encoder_path.as_posix()),
            "x_train": str(x_train_path.as_posix()),
            "x_valid": str(x_valid_path.as_posix()),
            "x_test": str(x_test_path.as_posix()),
            "y_train": str(y_train_path.as_posix()),
            "y_valid": str(y_valid_path.as_posix()),
            "y_test": str(y_test_path.as_posix()),
        },
        "schema": {
            "label_col": label_col,
            "text_col": text_col,
        },
        "tfidf": {
            "max_features": tfidf_cfg.get("max_features"),
            "ngram_range": list(_parse_ngram_range(tfidf_cfg.get("ngram_range", [1, 1]))),
            "min_df": tfidf_cfg.get("min_df"),
            "max_df": tfidf_cfg.get("max_df"),
            "lowercase": tfidf_cfg.get("lowercase"),
            "sublinear_tf": tfidf_cfg.get("sublinear_tf"),
            "dtype": tfidf_cfg.get("dtype"),
        },
        "stats": {
            "n_train": int(x_train.shape[0]),
            "n_valid": int(x_valid.shape[0]),
            "n_test": int(x_test.shape[0]),
            "n_features": int(x_train.shape[1]),
            "train_nnz": int(x_train.nnz),
        },
        "labels": {
            "num_classes": int(len(label_encoder.classes_)),
            "classes": [str(c) for c in label_encoder.classes_.tolist()],
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return FeatureOutputs(
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path,
        x_train_path=x_train_path,
        x_valid_path=x_valid_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_valid_path=y_valid_path,
        y_test_path=y_test_path,
        manifest_path=manifest_path,
    )

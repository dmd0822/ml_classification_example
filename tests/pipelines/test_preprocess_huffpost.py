"""Tests for HuffPost preprocessing helpers."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.pipelines.preprocess.huffpost import (
    clean_huffpost_dataframe,
    combine_text,
    make_splits,
    normalize_whitespace,
)


def test_normalize_whitespace_collapses_runs() -> None:
    """normalize_whitespace() should collapse whitespace into single spaces."""

    assert normalize_whitespace("a  b\n\tc") == "a b c"


def test_combine_text_uses_sep_and_trims() -> None:
    """combine_text() should join headline and description with [SEP] and strip."""

    out = combine_text(
        " hello ",
        " world ",
        separator="[SEP]",
        normalize=True,
        strip=True,
    )
    assert out == "hello [SEP] world"


def test_clean_drops_missing_category_and_empty_text() -> None:
    """Cleaning should drop rows with empty label or empty combined text."""

    df_raw = pd.DataFrame(
        {
            "category": ["A", None, "", "B"],
            "headline": ["h1", "h2", "", ""],
            "short_description": ["d1", "d2", "", ""],
        }
    )

    schema_cfg: Dict[str, Any] = {
        "label_col": "category",
        "headline_col": "headline",
        "description_col": "short_description",
        "text_col": "text",
    }
    text_cfg: Dict[str, Any] = {
        "separator": "[SEP]",
        "null_headline": "",
        "null_description": "",
        "strip": True,
        "normalize_whitespace": True,
        "drop_empty_text": True,
        "min_text_length": 1,
    }
    preprocess_cfg: Dict[str, Any] = {
        "drop_missing_category": True,
        "drop_rare_labels_min_count": 1,
        "dedupe_on_text": False,
    }

    df_clean, stats = clean_huffpost_dataframe(
        df_raw,
        schema_cfg=schema_cfg,
        text_cfg=text_cfg,
        preprocess_cfg=preprocess_cfg,
    )

    assert "text" in df_clean.columns
    assert df_clean["category"].isna().sum() == 0
    assert (df_clean["category"].astype("string").str.strip() == "").sum() == 0
    assert (df_clean["text"].astype("string").str.len() < 1).sum() == 0
    assert stats["rows_in"] == 4


def test_make_splits_stratified_shapes() -> None:
    """Stratified splitting should produce three non-empty splits."""

    df = pd.DataFrame(
        {
            "row_id": list(range(30)),
            "category": ["A"] * 15 + ["B"] * 15,
            "text": [f"t{i}" for i in range(30)],
        }
    )

    train, valid, test = make_splits(
        df,
        label_col="category",
        random_seed=42,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        stratified=True,
    )

    assert len(train) > 0
    assert len(valid) > 0
    assert len(test) > 0
    assert len(train) + len(valid) + len(test) == len(df)

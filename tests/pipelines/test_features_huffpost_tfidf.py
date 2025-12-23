"""Tests for TF-IDF feature extraction helpers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.pipelines.features.huffpost_tfidf import build_tfidf_vectorizer


def test_build_vectorizer_respects_ngram_range() -> None:
    """Vectorizer should accept config-driven ngram_range."""

    vec = build_tfidf_vectorizer(
        {
            "max_features": 100,
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "sublinear_tf": True,
            "dtype": "float32",
        }
    )

    x = vec.fit_transform(["hello world", "hello there"])
    assert x.shape[0] == 2
    assert x.shape[1] >= 3


def test_vectorizer_fit_on_train_only_prevents_vocab_from_test_only_token() -> None:
    """A token only seen in test should not appear in train-fitted vocabulary."""

    tfidf_cfg: Dict[str, Any] = {
        "max_features": 1000,
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "lowercase": True,
        "sublinear_tf": False,
        "dtype": "float32",
    }

    vec = build_tfidf_vectorizer(tfidf_cfg)

    train_text = ["cat dog", "dog mouse"]
    test_text = ["ONLYINTHE_TESTSET"]

    vec.fit(train_text)

    # sklearn lowercases by default
    assert "onlyinthe_testset" not in vec.vocabulary_

    x_test = vec.transform(test_text)
    assert x_test.shape[0] == 1
    assert x_test.nnz == 0


def test_vectorizer_dtype_float32() -> None:
    """Vectorizer should produce float32 when configured."""

    vec = build_tfidf_vectorizer(
        {
            "max_features": 100,
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
            "sublinear_tf": False,
            "dtype": "float32",
        }
    )

    # scikit-learn's default token_pattern ignores 1-char tokens.
    x = vec.fit_transform(["alpha beta gamma", "alpha beta", "beta gamma"])
    assert x.dtype == np.float32

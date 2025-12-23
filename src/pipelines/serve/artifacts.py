"""Model artifact loading for inference.

This module centralizes how we load the baked-in model artifacts so the HTTP
layer can stay thin and focused on request/response handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import tensorflow as tf
from transformers import AutoTokenizer


@dataclass(frozen=True)
class LoadedArtifacts:
    """In-memory model artifacts needed for prediction."""

    model: tf.keras.Model
    tokenizer: Any
    label_encoder: Any
    max_length: int


def load_artifacts(
    *,
    keras_model_path: str | Path,
    tokenizer_dir: str | Path,
    label_encoder_path: str | Path,
    max_length: int,
) -> LoadedArtifacts:
    """Load model + tokenizer + label encoder.

    Parameters
    ----------
    keras_model_path:
        Path to a saved Keras model file (e.g. `keras_model.keras`).
    tokenizer_dir:
        Path to a directory created by `AutoTokenizer.save_pretrained()`.
    label_encoder_path:
        Path to the label encoder joblib artifact.
    max_length:
        Tokenizer max length used during training.

    Returns
    -------
    LoadedArtifacts
        In-memory artifacts ready for inference.
    """

    model_path = Path(keras_model_path)
    tok_dir = Path(tokenizer_dir)
    le_path = Path(label_encoder_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")
    if not tok_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tok_dir}")
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {le_path}")

    model = tf.keras.models.load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    label_encoder = joblib.load(le_path)

    return LoadedArtifacts(
        model=model,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=int(max_length),
    )

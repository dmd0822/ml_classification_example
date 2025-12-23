"""Prediction helpers for the DistilBERT (Keras Hub backbone) classifier."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from src.pipelines.serve.artifacts import LoadedArtifacts


def _encode_texts(tokenizer: Any, texts: Sequence[str], *, max_length: int) -> Dict[str, tf.Tensor]:
    """Tokenize texts into the input dict expected by the saved Keras model."""

    enc = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=int(max_length),
        return_tensors="tf",
    )

    token_ids = tf.cast(enc["input_ids"], tf.int32)
    padding_mask = tf.cast(enc["attention_mask"], tf.int32)
    return {"token_ids": token_ids, "padding_mask": padding_mask}


def predict_topk(
    artifacts: LoadedArtifacts,
    texts: Sequence[str],
    *,
    top_k: int = 1,
) -> List[Dict[str, Any]]:
    """Predict top-k labels (and probabilities) for each input text.

    Parameters
    ----------
    artifacts:
        LoadedArtifacts containing model, tokenizer, and label encoder.
    texts:
        One or more input texts.
    top_k:
        Number of top predictions to return per item.

    Returns
    -------
    List[Dict[str, Any]]
        Per input, returns a dict with `predicted_label` and `top_k`.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    x = _encode_texts(artifacts.tokenizer, texts, max_length=artifacts.max_length)
    logits = artifacts.model(x, training=False)

    probs = tf.nn.softmax(logits, axis=1).numpy()
    probs = np.asarray(probs)

    k = min(int(top_k), probs.shape[1])
    top_idx = np.argsort(-probs, axis=1)[:, :k]

    results: List[Dict[str, Any]] = []
    for i in range(len(texts)):
        idxs = top_idx[i]
        labels = artifacts.label_encoder.inverse_transform(idxs)
        items: List[Dict[str, Any]] = []
        for j, cls_idx in enumerate(idxs.tolist()):
            items.append(
                {
                    "class_index": int(cls_idx),
                    "label": str(labels[j]),
                    "probability": float(probs[i, cls_idx]),
                }
            )

        results.append(
            {
                "predicted_label": str(items[0]["label"]),
                "top_k": items,
            }
        )

    return results

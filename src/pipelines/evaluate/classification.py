"""Classification evaluation utilities.

Provides accuracy + macro-F1 computation and helpers for saving reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


@dataclass(frozen=True)
class ClassificationMetrics:
    """Container for common multiclass metrics."""

    accuracy: float
    macro_f1: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Compute accuracy and macro-F1."""

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
    )


def write_predictions_csv(
    path: Path,
    *,
    row_id: Optional[Iterable[int]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_label: Optional[Iterable[str]] = None,
    y_pred_label: Optional[Iterable[str]] = None,
) -> None:
    """Write a standard predictions CSV for later analysis."""

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    if row_id is not None:
        df.insert(0, "row_id", list(row_id))

    if y_true_label is not None:
        df["y_true_label"] = list(y_true_label)

    if y_pred_label is not None:
        df["y_pred_label"] = list(y_pred_label)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_metrics_json(path: Path, *, metrics: ClassificationMetrics, extra: Mapping[str, Any]) -> None:
    """Write metrics JSON with optional metadata."""

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
        },
        "extra": dict(extra),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_classification_report(
    path: Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[Iterable[str]] = None,
) -> None:
    """Write sklearn classification report to a text file."""

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(target_names) if target_names is not None else None,
        digits=4,
        zero_division=0,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report + "\n", encoding="utf-8")

"""Shared helpers for training pipelines."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def make_run_dir(predictions_dir: str, *, experiment_name: str) -> Path:
    """Create a run directory under data/04-predictions.

    Uses a UTC timestamp for easy sorting and reproducibility.
    """

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(predictions_dir) / experiment_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def require_keys(cfg: Mapping[str, Any], keys: tuple[str, ...], *, name: str) -> None:
    """Ensure required config keys exist."""

    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in {name}: {missing}")

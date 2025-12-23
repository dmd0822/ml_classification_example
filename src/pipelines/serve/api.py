"""FastAPI app for HuffPost category prediction.

This is intentionally a small HTTP surface:
- GET /health: readiness probe (artifacts loaded)
- POST /predict: category prediction from text or headline+short_description

The model artifacts are expected to be baked into the Docker image and referred
to via environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipelines.common.config import load_json_config
from src.pipelines.serve.artifacts import LoadedArtifacts, load_artifacts
from src.pipelines.serve.predict import predict_topk
from src.pipelines.serve.text import combine_text


class PredictRequest(BaseModel):
    """Request payload for /predict."""

    text: Optional[str] = Field(default=None, description="Pre-combined text.")
    headline: Optional[str] = Field(default=None, description="Headline text.")
    short_description: Optional[str] = Field(default=None, description="Short description text.")
    top_k: int = Field(default=1, ge=1, le=10, description="Number of predictions to return.")


def _env_path(name: str, default: str) -> Path:
    """Resolve a filesystem path from an environment variable."""

    return Path(os.environ.get(name, default)).expanduser().resolve()


def _load_text_config(config_path: Path) -> Dict[str, Any]:
    """Load the JSON config used for assembling text.

    We reuse the same separator/normalization choices as training.
    """

    cfg = load_json_config(config_path)
    return {
        "separator": str(cfg.get("text_assembly", {}).get("separator", " ")),
        "normalize_whitespace": bool(cfg.get("text_assembly", {}).get("normalize_whitespace", True)),
        "strip": bool(cfg.get("text_assembly", {}).get("strip", True)),
    }


def _resolve_text(req: PredictRequest, *, text_cfg: Dict[str, Any]) -> str:
    """Resolve final text based on either `text` or `headline`/`short_description`."""

    if req.text is not None and str(req.text).strip() != "":
        return str(req.text)

    headline = "" if req.headline is None else str(req.headline)
    short_description = "" if req.short_description is None else str(req.short_description)

    combined = combine_text(
        headline,
        short_description,
        separator=str(text_cfg["separator"]),
        normalize=bool(text_cfg["normalize_whitespace"]),
        strip=bool(text_cfg["strip"]),
    )

    if combined.strip() == "":
        raise ValueError("Provide either `text` or (`headline` and/or `short_description`).")

    return combined


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(title="HuffPost Category Classifier", version="0.1.0")

    config_path = _env_path("TEXT_CONFIG_PATH", "/app/config/huffpost_category_text.json")
    text_cfg = _load_text_config(config_path)

    keras_model_path = _env_path("KERAS_MODEL_PATH", "/app/artifacts/best/keras_model.keras")
    tokenizer_dir = _env_path("TOKENIZER_DIR", "/app/artifacts/best/tokenizer")
    label_encoder_path = _env_path("LABEL_ENCODER_PATH", "/app/artifacts/best/label_encoder.joblib")
    max_length = int(os.environ.get("MAX_LENGTH", "64"))

    try:
        artifacts = load_artifacts(
            keras_model_path=keras_model_path,
            tokenizer_dir=tokenizer_dir,
            label_encoder_path=label_encoder_path,
            max_length=max_length,
        )
    except Exception as exc:
        artifacts = None
        load_error = str(exc)
    else:
        load_error = None

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Readiness endpoint."""

        if artifacts is None:
            return {"status": "not_ready", "error": load_error}
        return {"status": "ok"}

    @app.post("/predict")
    def predict(req: PredictRequest) -> Dict[str, Any]:
        """Predict category for a single item."""

        if artifacts is None:
            raise HTTPException(status_code=503, detail={"error": load_error})

        try:
            text = _resolve_text(req, text_cfg=text_cfg)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        preds = predict_topk(artifacts, [text], top_k=int(req.top_k))
        return {"predictions": preds[0]}

    return app


app = create_app()

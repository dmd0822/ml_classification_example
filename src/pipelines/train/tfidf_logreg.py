"""TF-IDF + Logistic Regression experiment.

Consumes feature artifacts from `data/03-features/` and writes predictions,
metrics, and a fitted sklearn model to `data/04-predictions/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from src.pipelines.evaluate.classification import (
    compute_metrics,
    write_classification_report,
    write_metrics_json,
    write_predictions_csv,
)
from src.pipelines.train.common import make_run_dir


def _load_features(paths_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Load TF-IDF features and labels from feature dir."""

    features_dir = Path(str(paths_cfg["features_dir"]))

    x_train = sp.load_npz(features_dir / str(paths_cfg["features_x_train_filename"]))
    x_valid = sp.load_npz(features_dir / str(paths_cfg["features_x_valid_filename"]))
    x_test = sp.load_npz(features_dir / str(paths_cfg["features_x_test_filename"]))

    y_train = np.load(features_dir / str(paths_cfg["features_y_train_filename"]))
    y_valid = np.load(features_dir / str(paths_cfg["features_y_valid_filename"]))
    y_test = np.load(features_dir / str(paths_cfg["features_y_test_filename"]))

    label_encoder = joblib.load(features_dir / str(paths_cfg["features_label_encoder_filename"]))

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "label_encoder": label_encoder,
    }


def run_tfidf_logreg_experiment(config: Mapping[str, Any], *, force: bool = False) -> Path:
    """Train/evaluate a logistic regression classifier on TF-IDF features."""

    paths_cfg = config["paths"]
    exp_cfg = config.get("experiments", {}).get("tfidf_logreg", {})

    run_dir = make_run_dir(str(paths_cfg["predictions_dir"]), experiment_name="tfidf_logreg")

    artifacts = _load_features(paths_cfg)

    model = LogisticRegression(
        C=float(exp_cfg.get("C", 4.0)),
        max_iter=int(exp_cfg.get("max_iter", 200)),
        solver=str(exp_cfg.get("solver", "saga")),
        n_jobs=None,
        verbose=0,
    )

    model.fit(artifacts["x_train"], artifacts["y_train"])

    for split in ("valid", "test"):
        x = artifacts[f"x_{split}"]
        y_true = artifacts[f"y_{split}"]
        y_pred = model.predict(x)

        metrics = compute_metrics(y_true, y_pred)
        write_metrics_json(
            run_dir / f"metrics_{split}.json",
            metrics=metrics,
            extra={
                "experiment": "tfidf_logreg",
                "split": split,
                "model": "LogisticRegression",
            },
        )

        label_encoder = artifacts["label_encoder"]
        y_true_label = label_encoder.inverse_transform(y_true)
        y_pred_label = label_encoder.inverse_transform(y_pred)

        write_predictions_csv(
            run_dir / f"predictions_{split}.csv",
            row_id=None,
            y_true=y_true,
            y_pred=y_pred,
            y_true_label=y_true_label,
            y_pred_label=y_pred_label,
        )

        write_classification_report(
            run_dir / f"classification_report_{split}.txt",
            y_true=y_true,
            y_pred=y_pred,
            target_names=[str(c) for c in label_encoder.classes_.tolist()],
        )

    joblib.dump(model, run_dir / "model.joblib")

    return run_dir

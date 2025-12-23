"""Small dense neural model on TF-IDF features.

Directly training a dense network on 50k-dim TF-IDF vectors can be memory-heavy.
This pipeline fits TruncatedSVD on train TF-IDF features to produce a compact
dense representation, then trains a small Keras MLP.

Artifacts are saved to `data/04-predictions/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD

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


def _build_mlp(input_dim: int, num_classes: int, *, hidden_units: Sequence[int], dropout: float) -> tf.keras.Model:
    """Build a small MLP classifier."""

    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = inputs

    for units in hidden_units:
        x = tf.keras.layers.Dense(int(units), activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(float(dropout))(x)

    logits = tf.keras.layers.Dense(int(num_classes), activation=None)(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


def run_tfidf_dense_experiment(config: Mapping[str, Any], *, force: bool = False) -> Path:
    """Train/evaluate a small dense model on SVD-compressed TF-IDF features."""

    paths_cfg = config["paths"]
    train_cfg = config.get("training", {})
    exp_cfg = config.get("experiments", {}).get("tfidf_dense", {})

    run_dir = make_run_dir(str(paths_cfg["predictions_dir"]), experiment_name="tfidf_dense")

    artifacts = _load_features(paths_cfg)

    svd_components = int(exp_cfg.get("svd_components", 256))
    svd = TruncatedSVD(n_components=svd_components, random_state=int(train_cfg.get("random_seed", 42)))

    x_train_dense = svd.fit_transform(artifacts["x_train"])
    x_valid_dense = svd.transform(artifacts["x_valid"])
    x_test_dense = svd.transform(artifacts["x_test"])

    num_classes = int(len(artifacts["label_encoder"].classes_))

    model = _build_mlp(
        input_dim=int(x_train_dense.shape[1]),
        num_classes=num_classes,
        hidden_units=exp_cfg.get("hidden_units", [256]),
        dropout=float(exp_cfg.get("dropout", 0.2)),
    )

    lr = float(exp_cfg.get("learning_rate", 1e-3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_cfg = train_cfg.get("early_stopping", {})
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=str(early_cfg.get("monitor", "val_loss")),
            patience=int(early_cfg.get("patience", 2)),
            restore_best_weights=bool(early_cfg.get("restore_best_weights", True)),
        )
    ]

    model.fit(
        x_train_dense,
        artifacts["y_train"],
        validation_data=(x_valid_dense, artifacts["y_valid"]),
        epochs=int(train_cfg.get("max_epochs", 8)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        callbacks=callbacks,
        verbose=2,
    )

    for split, x_dense, y_true in (
        ("valid", x_valid_dense, artifacts["y_valid"]),
        ("test", x_test_dense, artifacts["y_test"]),
    ):
        logits = model.predict(x_dense, batch_size=int(train_cfg.get("batch_size", 32)), verbose=0)
        y_pred = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

        metrics = compute_metrics(y_true, y_pred)
        write_metrics_json(
            run_dir / f"metrics_{split}.json",
            metrics=metrics,
            extra={
                "experiment": "tfidf_dense",
                "split": split,
                "model": "keras_mlp",
                "svd_components": svd_components,
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

    joblib.dump(svd, run_dir / "svd.joblib")
    model.save(run_dir / "keras_model.keras", include_optimizer=False)

    return run_dir

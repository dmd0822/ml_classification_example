"""DistilBERT experiments using Keras Hub (TensorFlow).

Implements two variants:
- Frozen: backbone frozen, train classification head only.
- Unfrozen: full backbone fine-tuned.

This pipeline reads preprocessed train/valid/test CSVs, tokenizes with a
DistilBERT preprocessor, trains with EarlyStopping, and writes predictions and
metrics to `data/04-predictions/`.

Note: This can be slow on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import tensorflow_text as _tensorflow_text  # noqa: F401

    _HAS_TF_TEXT = True
except Exception:
    _HAS_TF_TEXT = False

import keras_hub

from src.pipelines.evaluate.classification import (
    compute_metrics,
    write_classification_report,
    write_metrics_json,
    write_predictions_csv,
)
from src.pipelines.train.common import make_run_dir


@dataclass(frozen=True)
class BertRunConfig:
    """Resolved training config for BERT experiments."""

    preset: str
    max_length: int
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_monitor: str
    early_patience: int
    early_restore_best: bool


def _resolve_hf_model_name(model_name: str) -> str:
    """Resolve a config model name to a HuggingFace model id.

    The config may contain either a HuggingFace id (e.g. 'distilbert-base-uncased')
    or a Keras Hub preset id (e.g. 'distil_bert_base_en_uncased').
    """

    keras_to_hf = {
        "distil_bert_base_en_uncased": "distilbert-base-uncased",
        "distil_bert_base_en": "distilbert-base-cased",
        "distil_bert_base_multi": "distilbert-base-multilingual-cased",
    }
    return keras_to_hf.get(model_name, model_name)


def _run_distilbert_experiment_tokenized_backbone(
    *,
    experiment_name: str,
    run_dir: Path,
    text_train: np.ndarray,
    y_train: np.ndarray,
    text_valid: np.ndarray,
    y_valid: np.ndarray,
    text_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: Any,
    bert_cfg: BertRunConfig,
    model_name_raw: str,
    unfrozen: bool,
    seed: int,
) -> None:
    from transformers import AutoTokenizer

    hf_model_name = _resolve_hf_model_name(model_name_raw)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    def encode(text_arr: np.ndarray) -> Dict[str, tf.Tensor]:
        enc = tokenizer(
            text_arr.tolist(),
            truncation=True,
            padding="max_length",
            max_length=int(bert_cfg.max_length),
            return_tensors="tf",
        )
        # Keras Hub backbones expect token_ids/padding_mask naming.
        token_ids = tf.cast(enc["input_ids"], tf.int32)
        padding_mask = tf.cast(enc["attention_mask"], tf.int32)
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    x_train = encode(text_train)
    x_valid = encode(text_valid)
    x_test = encode(text_test)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(buffer_size=min(len(y_train), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(int(bert_cfg.batch_size)).prefetch(tf.data.AUTOTUNE)

    ds_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    ds_valid = ds_valid.batch(int(bert_cfg.batch_size)).prefetch(tf.data.AUTOTUNE)

    backbone = keras_hub.models.DistilBertBackbone.from_preset(
        str(bert_cfg.preset),
        sequence_length=int(bert_cfg.max_length),
    )
    backbone.trainable = bool(unfrozen)

    token_ids_in = tf.keras.Input(shape=(int(bert_cfg.max_length),), dtype=tf.int32, name="token_ids")
    padding_mask_in = tf.keras.Input(shape=(int(bert_cfg.max_length),), dtype=tf.int32, name="padding_mask")
    backbone_out = backbone({"token_ids": token_ids_in, "padding_mask": padding_mask_in})

    if isinstance(backbone_out, dict):
        sequence = backbone_out.get("sequence_output")
    else:
        sequence = backbone_out

    if sequence is None:
        raise RuntimeError("DistilBertBackbone did not return a sequence output.")

    cls_token = sequence[:, 0, :]
    logits = tf.keras.layers.Dense(int(len(label_encoder.classes_)), name="classifier")(cls_token)
    model = tf.keras.Model(inputs={"token_ids": token_ids_in, "padding_mask": padding_mask_in}, outputs=logits)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(bert_cfg.learning_rate)),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=str(bert_cfg.early_monitor),
            patience=int(bert_cfg.early_patience),
            restore_best_weights=bool(bert_cfg.early_restore_best),
        )
    ]

    model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=int(bert_cfg.max_epochs),
        callbacks=callbacks,
        verbose=2,
    )

    def predict_labels(x: Dict[str, tf.Tensor]) -> np.ndarray:
        ds = tf.data.Dataset.from_tensor_slices(x).batch(int(bert_cfg.batch_size))
        logits_arr = model.predict(ds, verbose=0)
        return np.asarray(np.argmax(np.asarray(logits_arr), axis=1), dtype=np.int64)

    for split, x, y_true in (
        ("valid", x_valid, y_valid),
        ("test", x_test, y_test),
    ):
        y_pred = predict_labels(x)

        metrics = compute_metrics(y_true, y_pred)
        write_metrics_json(
            run_dir / f"metrics_{split}.json",
            metrics=metrics,
            extra={
                "experiment": experiment_name,
                "split": split,
                "preset": str(bert_cfg.preset),
                "hf_model": hf_model_name,
                "max_length": int(bert_cfg.max_length),
                "unfrozen": bool(unfrozen),
                "backend": "keras_hub_backbone+transformers_tokenizer",
            },
        )

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

    model.save(run_dir / "keras_model.keras", include_optimizer=False)
    tokenizer.save_pretrained(run_dir / "tokenizer")


def _load_preprocessed(paths_cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train/valid/test splits."""

    pre_dir = Path(str(paths_cfg["preprocessed_dir"]))
    df_train = pd.read_csv(pre_dir / str(paths_cfg["preprocessed_train_filename"]))
    df_valid = pd.read_csv(pre_dir / str(paths_cfg["preprocessed_valid_filename"]))
    df_test = pd.read_csv(pre_dir / str(paths_cfg["preprocessed_test_filename"]))
    return df_train, df_valid, df_test


def _resolve_bert_cfg(config: Mapping[str, Any]) -> BertRunConfig:
    """Resolve DistilBERT config values with defaults."""

    train_cfg = config.get("training", {})
    early_cfg = train_cfg.get("early_stopping", {})
    exp_cfg = config.get("experiments", {}).get("distilbert", {})

    preset_raw = str(exp_cfg.get("model_name", "distil_bert_base_en_uncased"))
    preset_map = {
        # Common HuggingFace identifiers -> Keras Hub built-in presets
        "distilbert-base-uncased": "distil_bert_base_en_uncased",
        "distilbert-base-cased": "distil_bert_base_en",
        "distilbert-base-multilingual-cased": "distil_bert_base_multi",
    }
    preset = preset_map.get(preset_raw, preset_raw)

    return BertRunConfig(
        preset=preset,
        max_length=int(exp_cfg.get("max_length", 64)),
        learning_rate=float(exp_cfg.get("learning_rate", 5e-5)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        max_epochs=int(train_cfg.get("max_epochs", 8)),
        early_monitor=str(early_cfg.get("monitor", "val_loss")),
        early_patience=int(early_cfg.get("patience", 2)),
        early_restore_best=bool(early_cfg.get("restore_best_weights", True)),
    )


def _make_tf_dataset(
    text: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Create a batched tf.data dataset."""

    ds = tf.data.Dataset.from_tensor_slices((text, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(text), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def run_distilbert_experiment(
    config: Mapping[str, Any],
    *,
    unfrozen: bool,
    force: bool = False,
) -> Path:
    """Run a DistilBERT experiment (frozen or unfrozen)."""

    paths_cfg = config["paths"]
    schema_cfg = config["schema"]

    text_col = str(schema_cfg["text_col"])
    label_col = str(schema_cfg["label_col"])

    df_train, df_valid, df_test = _load_preprocessed(paths_cfg)

    for name, df in ("train", df_train), ("valid", df_valid), ("test", df_test):
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"{name} split missing required columns: {text_col}, {label_col}")

    # Fit label encoder on train labels only.
    label_encoder = joblib.load(Path(str(paths_cfg["features_dir"])) / str(paths_cfg["features_label_encoder_filename"]))

    y_train = label_encoder.transform(df_train[label_col].fillna("").astype("string"))
    y_valid = label_encoder.transform(df_valid[label_col].fillna("").astype("string"))
    y_test = label_encoder.transform(df_test[label_col].fillna("").astype("string"))

    text_train = df_train[text_col].fillna("").astype("string").to_numpy()
    text_valid = df_valid[text_col].fillna("").astype("string").to_numpy()
    text_test = df_test[text_col].fillna("").astype("string").to_numpy()

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("random_seed", 42))
    tf.random.set_seed(seed)

    bert_cfg = _resolve_bert_cfg(config)

    exp_cfg = config.get("experiments", {}).get("distilbert", {})
    model_name_raw = str(exp_cfg.get("model_name", bert_cfg.preset))

    experiment_name = "distilbert_unfrozen" if unfrozen else "distilbert_frozen"
    run_dir = make_run_dir(str(paths_cfg["predictions_dir"]), experiment_name=experiment_name)

    if _HAS_TF_TEXT:
        # Keras Hub path (requires tensorflow-text for tokenization).
        model = keras_hub.models.DistilBertTextClassifier.from_preset(
            bert_cfg.preset,
            num_classes=int(len(label_encoder.classes_)),
            sequence_length=int(bert_cfg.max_length),
        )

        model.backbone.trainable = bool(unfrozen)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=bert_cfg.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=bert_cfg.early_monitor,
                patience=bert_cfg.early_patience,
                restore_best_weights=bert_cfg.early_restore_best,
            )
        ]

        ds_train = _make_tf_dataset(text_train, y_train, batch_size=bert_cfg.batch_size, shuffle=True, seed=seed)
        ds_valid = _make_tf_dataset(text_valid, y_valid, batch_size=bert_cfg.batch_size, shuffle=False, seed=seed)

        model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=bert_cfg.max_epochs,
            callbacks=callbacks,
            verbose=2,
        )

        for split, text_arr, y_true in (
            ("valid", text_valid, y_valid),
            ("test", text_test, y_test),
        ):
            ds = tf.data.Dataset.from_tensor_slices(text_arr).batch(bert_cfg.batch_size)
            logits = model.predict(ds, verbose=0)
            y_pred = np.asarray(np.argmax(logits, axis=1), dtype=np.int64)

            metrics = compute_metrics(y_true, y_pred)
            write_metrics_json(
                run_dir / f"metrics_{split}.json",
                metrics=metrics,
                extra={
                    "experiment": experiment_name,
                    "split": split,
                    "preset": bert_cfg.preset,
                    "max_length": bert_cfg.max_length,
                    "unfrozen": bool(unfrozen),
                    "backend": "keras_hub",
                },
            )

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

        model.save(run_dir / "keras_model.keras", include_optimizer=False)
    else:
        _run_distilbert_experiment_tokenized_backbone(
            experiment_name=experiment_name,
            run_dir=run_dir,
            text_train=text_train,
            y_train=y_train,
            text_valid=text_valid,
            y_valid=y_valid,
            text_test=text_test,
            y_test=y_test,
            label_encoder=label_encoder,
            bert_cfg=bert_cfg,
            model_name_raw=model_name_raw,
            unfrozen=unfrozen,
            seed=seed,
        )

    return run_dir

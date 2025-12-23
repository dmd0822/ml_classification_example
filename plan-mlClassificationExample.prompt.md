## Plan: Combine Text + Vectorize for Category Classification

Create a reference multi-class classifier where the label is `category`, and the model features come from a new combined `text` column built from `headline` + `short_description`, then vectorized in the features stage.

### Steps 1) Define the contract and config for the combined column
1. Specify required raw fields: `category`, `headline`, `short_description`; specify derived field: `text`.
2. Add config knobs in [config/](config/) for `label_col=category`, `headline_col`, `description_col`, `text_col=text`, separator (e.g., `[SEP]`), and null/empty handling rules.

### Steps 2) Ingest raw JSON into data/01-raw with basic schema checks
1. Add a thin downloader entrypoint in [entrypoints/](entrypoints/) that saves the dataset to [data/01-raw/](data/01-raw/).
2. In `src.pipelines.ingest`, validate required columns exist and write a small manifest (URL, hash, timestamp) next to the raw file.

### Steps 3) EDA first, but keep it “graduate-able”
1. Add notebooks in `notebooks/` (starting with `01_eda_huffpost_category.ipynb`) exploring: class balance by category, missing/empty text rates, length distributions, duplicates, and top tokens.
2. Decide preprocessing rules from EDA (drop null/empty, text normalization, optional label filtering like minimum examples per category).
3. Save EDA artifacts (PNG/CSV) under `notebooks/eda_artifacts/` and move stabilized logic into `src/pipelines/`.

### Steps 4) Preprocess and create the combined `text` column into data/02-preprocessed
1. Implement preprocessing in [src/pipelines/preprocess/huffpost.py](src/pipelines/preprocess/huffpost.py) that creates `text = headline + sep + short_description` with null-safe normalization.
2. Output deterministic splits under `data/02-preprocessed/huffpost/v1/` (`full_clean.csv`, `train.csv`, `valid.csv`, `test.csv`) plus a `preprocess.manifest.json`.
3. Add tests in [tests/pipelines/test_preprocess_huffpost.py](tests/pipelines/test_preprocess_huffpost.py) that verify `text` creation and dropping invalid rows.

### Steps 5) Vectorize the new `text` column into data/03-features
1. Implement TF-IDF features in [src/pipelines/features/huffpost_tfidf.py](src/pipelines/features/huffpost_tfidf.py), fitting on *train only* and transforming valid/test to avoid leakage.
2. Persist artifacts under `data/03-features/huffpost/tfidf_v1/` (`X_*.npz`, `y_*.npy`, `tfidf_vectorizer.joblib`, `label_encoder.joblib`) plus a `features.manifest.json`.
3. Keep entrypoints thin: [entrypoints/featurize_huffpost_tfidf.py](entrypoints/featurize_huffpost_tfidf.py) only loads config + calls the pipeline.

### Steps 6) Experimentation: train/evaluate on vectors, not raw text
1. Implement training in `src.pipelines.train` and evaluation in `src.pipelines.evaluate` using accuracy + macro‑F1 for multiclass `category`.
2. Run the following experiments and write reports/predictions to [data/04-predictions/](data/04-predictions/):
	- TF‑IDF + Logistic Regression (classical baseline)
	- Small custom dense model on TF‑IDF features
	- DistilBERT-Frozen (train classification head only)
	- DistilBERT-Unfrozen (fine-tune full backbone)
3. Note: the DistilBERT experiments typically require the `transformers` library and a backend (PyTorch or TensorFlow). We will choose one backend and add the minimal required dependencies before implementing those runs.

### Further Considerations 1. Artifact storage layout for features
1. Use a run-stamped folder under [data/03-features/](data/03-features/) (e.g., `<dataset>/<feature_set>/<run_id>/`) to store `X_*`, vectorizer, and a manifest without cluttering entrypoints.

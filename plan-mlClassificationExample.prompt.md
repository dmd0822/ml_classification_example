## Plan: Combine Text + Vectorize for Category Classification

Create a reference multi-class classifier where the label is `category`, and the model features come from a new combined `text` column built from `headline` + `short_description`, then vectorized in the features stage.

### Steps 1) Define the contract and config for the combined column
1. Specify required raw fields: `category`, `headline`, `short_description`; specify derived field: `text`.
2. Add config knobs in [config/](config/) for `label_col=category`, `headline_col`, `description_col`, `text_col=text`, separator (e.g., `[SEP]`), and null/empty handling rules.

### Steps 2) Ingest raw JSON into data/01-raw with basic schema checks
1. Add a thin downloader entrypoint in [entrypoints/](entrypoints/) that saves the dataset to [data/01-raw/](data/01-raw/).
2. In `src.pipelines.ingest`, validate required columns exist and write a small manifest (URL, hash, timestamp) next to the raw file.

### Steps 3) EDA first, but keep it “graduate-able”
1. Add notebooks in notebooks (e.g., 01_eda_overview.ipynb, 02_baseline_quickstart.ipynb) exploring: class balance by category, missing/empty text rates, length distributions, and top tokens per class.
2. Decide preprocessing rules from EDA (drop null/empty, text normalization, optional label filtering like minimum examples per category).
3. Move stabilized logic into src.pipelines.preprocess.clean_dataset so notebooks consume pipeline outputs.

### Steps 4) Preprocess and create the combined `text` column into data/02-preprocessed
1. Implement `combine_text(df, cfg) -> df` in `src.pipelines.features` (or `src.pipelines.preprocess`) that creates `text = headline + sep + short_description` with null-safe normalization.
2. Output a cleaned, minimal dataset into [data/02-preprocessed/](data/02-preprocessed/) containing at least `text` and `category` (plus optional `id` and `split`).
3. Add tests in [tests/](tests/) that verify `text` creation (null handling, separator, no empty strings) and that `category` is non-null.

### Steps 5) Vectorize the new `text` column into data/03-features
1. Implement vectorization boundaries in `src.pipelines.features`: `fit_vectorizer(text, cfg) -> vectorizer` and `transform_text(vectorizer, text) -> X`.
2. Persist artifacts under [data/03-features/](data/03-features/) (feature matrices + fitted vectorizer + a manifest with vectorizer params and source snapshot) so experiments are reproducible.
3. Keep entrypoints thin: they only load config + call pipeline functions; pipeline code owns I/O to the staged folders.

### Steps 6) Experimentation: train/evaluate on vectors, not raw text
1. Implement training in `src.pipelines.train.train(X, y, cfg)` and evaluation in `src.pipelines.evaluate` using accuracy + macro‑F1 for multiclass `category`.
2. Run minimal experiments: TF‑IDF word n‑grams + linear classifier (baseline), then char n‑grams variant; write reports/predictions to [data/04-predictions/](data/04-predictions/).

### Further Considerations 1. Artifact storage layout for features
1. Use a run-stamped folder under [data/03-features/](data/03-features/) (e.g., `<dataset>/<feature_set>/<run_id>/`) to store `X_*`, vectorizer, and a manifest without cluttering entrypoints.

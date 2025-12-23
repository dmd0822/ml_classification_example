# `entrypoints/`

This folder contains the runnable scripts for your project (training, batch inference, evaluation).

Keeping entry points explicit makes it much easier to:

- Run jobs consistently in CI
- Containerize with Docker
- Schedule runs (workflow schedulers, Airflow, Prefect, etc.)

## What belongs here

- Thin scripts that parse CLI arguments / load config
- Calls into pipeline code in `src/pipelines/`
- Orchestration glue (but not the core ML logic)

## Suggested scripts (examples)

You can add scripts like:

- `train.py` (train a model)
- `predict.py` (generate predictions)
- `evaluate.py` (compute metrics / reports)

## Conventions

- Keep scripts **small and boring**: argument parsing + calling library functions.
- Put ML logic in `src/pipelines/` so it’s testable.
- Use a `if __name__ == "__main__":` guard.

## What this repo provides

All entry points accept:

- `--config` (defaults to `config/huffpost_category_text.json`)
- `--force` (overwrite outputs)

### Download raw data

```powershell
python -m entrypoints.download_huffpost_raw --config config/huffpost_category_text.json
```

### Preprocess

```powershell
python -m entrypoints.preprocess_huffpost --config config/huffpost_category_text.json
```

### TF-IDF features

```powershell
python -m entrypoints.featurize_huffpost_tfidf --config config/huffpost_category_text.json
```

### Train + evaluate

Baseline (TF-IDF + Logistic Regression):

```powershell
python -m entrypoints.train_eval_huffpost_tfidf_logreg --config config/huffpost_category_text.json
```

Other experiments:

```powershell
python -m entrypoints.train_eval_huffpost_tfidf_dense --config config/huffpost_category_text.json
python -m entrypoints.train_eval_huffpost_distilbert_frozen --config config/huffpost_category_text.json
python -m entrypoints.train_eval_huffpost_distilbert_unfrozen --config config/huffpost_category_text.json
```

## How This Fits

- Loads settings from [`config/`](../config/)
- Calls reusable logic in [`src/pipelines/`](../src/pipelines/)
- Reads/writes staged artifacts under [`data/`](../data/)
- Experiments that become stable often start in [`notebooks/`](../notebooks/)
- Runtime resources are often provisioned via [`infra/`](../infra/)

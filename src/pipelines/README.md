# `src/pipelines/`

This folder contains reusable, testable pipeline code.

Treat feature engineering, training, inference, and evaluation as pipelines rather than one-off scripts. This makes your work easier to:

- Reuse
- Test
- Automate
- Productionize

## What belongs here

- Feature engineering and transformation logic
- Training routines (model fitting, validation, model selection)
- Inference routines (batch/online prediction logic)
- Evaluation and metric computation
- Shared utilities used by multiple pipelines

## Suggested layout (example)

```text
src/
  pipelines/
    common/      # config, shared helpers
    ingest/      # download/import raw data
    preprocess/  # cleaning/splitting
    features/    # feature extraction
    train/       # training experiments
    evaluate/    # metrics
    serve/       # FastAPI + inference helpers
```

## Design principles (recommended)

- Prefer small, pure functions where possible.
- Keep I/O at the edges: read inputs, call pipeline logic, write outputs.
- Make pipeline steps deterministic given config + input data.
- Avoid hard-coded paths; take inputs/outputs from config or parameters.

## Relationship to `entrypoints/`

Entry points should be thin wrappers that call the functions defined here.

## Running pipeline code

In general, you run pipeline code via the entry points:

```powershell
python -m entrypoints.download_huffpost_raw
python -m entrypoints.preprocess_huffpost
python -m entrypoints.featurize_huffpost_tfidf
python -m entrypoints.train_eval_huffpost_tfidf_logreg
```

The HTTP inference API is served by:

```powershell
python -m uvicorn src.pipelines.serve.api:app --host 127.0.0.1 --port 8080
```

Note: for local (non-Docker) runs you must set env vars like `KERAS_MODEL_PATH`, `TOKENIZER_DIR`, and `LABEL_ENCODER_PATH` to point at your artifacts.

## How This Fits

- Implemented and tested here, executed via [`entrypoints/`](../../entrypoints/)
- Typically configured via [`config/`](../../config/)
- Consumes/produces artifacts in staged folders under [`data/`](../../data/)
- Supported by automated checks in [`tests/`](../../tests/)

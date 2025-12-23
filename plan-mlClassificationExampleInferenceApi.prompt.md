## Plan: Serve Best Model via HTTP + Docker

Add a minimal HTTP API for inference that loads the best model artifacts on startup (as selected in the latest report) and exposes a small `/predict` endpoint for classification. Containerize it by baking the best-model artifacts directly into the Docker image so `docker run` works without volume mounts. Keep the API intentionally small (health + predict) and reuse the existing preprocessing/text-building logic to match training.

### Steps 5
1. Confirm the “best model” run folder and required artifacts from [reports/report_2025-12-23.md](reports/report_2025-12-23.md) and its run directory under [data/04-predictions/huffpost/](data/04-predictions/huffpost/), including `saved_model` and tokenizer files.
2. Add an inference/serving module under [src/pipelines/](src/pipelines/) (new `serve` or `infer` package) that reuses text construction/normalization from [src/pipelines/preprocess/huffpost.py](src/pipelines/preprocess/huffpost.py), loads the baked-in model + tokenizer dir, and decodes predictions using the existing label encoder in [data/03-features/huffpost/tfidf_v1/](data/03-features/huffpost/tfidf_v1/).
3. Implement a minimal HTTP API layer (recommend FastAPI + Uvicorn since no web stack exists today) with:
   - `GET /health` returning readiness after artifacts load
   - `POST /predict` accepting either `text` or `headline` + `short_description` and returning predicted label (optionally top-k)
4. Add a serving entrypoint (either `python -m ...` module invocation or a thin script under [entrypoints/](entrypoints/)) that instantiates the app and reads config from env vars (e.g., `MODEL_DIR`, `TOKENIZER_DIR`, `LABEL_ENCODER_PATH`), defaulting to the baked-in paths.
5. Update [Dockerfile](Dockerfile) to build a runnable image that:
   - Installs dependencies (add `fastapi` + `uvicorn` to requirements or a serving-specific requirements file)
   - Copies the application code
   - Copies/bakes the best-model artifacts into a fixed path inside the image (e.g., `/app/artifacts/best/…`)
   - Sets `EXPOSE 8080` and `CMD` to run Uvicorn serving the app

### Further Considerations 2
1. Input contract: should `/predict` accept a single item or a batch list (default simplest: accept one item and optionally allow list for throughput)?
2. Image size tradeoff: baking DistilBERT artifacts will make the image large; confirm this is acceptable vs. baking a smaller TF-IDF model.

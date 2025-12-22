# Ingest pipelines

This folder contains ingestion logic for getting external datasets into the staged
raw data directory.

- For the HuffPost example, use:
  - Entry point: `python -m entrypoints.download_huffpost_raw --config config/huffpost_category_text.json`
  - Pipeline code: `src.pipelines.ingest.huffpost`

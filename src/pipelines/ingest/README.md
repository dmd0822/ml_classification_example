# Ingest pipelines

This folder contains ingestion logic for getting external datasets into the staged raw data directory.

## HuffPost dataset

- Entry point:

```powershell
python -m entrypoints.download_huffpost_raw --config config/huffpost_category_text.json
```

- Pipeline code: `src.pipelines.ingest.huffpost`

### Output location

The output directory is controlled by `paths.raw_dir` in `config/huffpost_category_text.json`.

"""HuffPost dataset ingestion.

Downloads the HuffPost News Category dataset JSONL file into `data/01-raw/` and
writes a small manifest capturing provenance and basic validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DownloadResult:
    """Result of downloading a file."""

    file_path: Path
    sha256: str
    bytes_written: int
    approx_line_count: int


def download_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    user_agent: str = "ml_classification_example/1.0",
    chunk_size: int = 1024 * 1024,
) -> DownloadResult:
    """Download a URL to a local file, streaming bytes to disk.

    Notes
    -----
    - This function computes SHA-256 while downloading.
    - For JSONL, it also approximates record count by counting newline bytes.

    Parameters
    ----------
    url:
        Source URL.
    output_path:
        Local path to write.
    force:
        If True, overwrite an existing file.
    user_agent:
        User-Agent header to send.
    chunk_size:
        Streaming chunk size in bytes.

    Returns
    -------
    DownloadResult
        Download metadata.

    Raises
    ------
    FileExistsError
        If output exists and force is False.
    """

    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sha256 = hashlib.sha256()
    bytes_written = 0
    line_count = 0

    request = Request(url, headers={"User-Agent": user_agent})

    # Download to a temporary file and then atomically replace the target.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with urlopen(request) as response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            sha256.update(chunk)
            bytes_written += len(chunk)
            line_count += chunk.count(b"\n")

    tmp_path.replace(output_path)

    return DownloadResult(
        file_path=output_path,
        sha256=sha256.hexdigest(),
        bytes_written=bytes_written,
        approx_line_count=line_count,
    )


def _iter_jsonl_records(file_path: Path, *, max_records: int) -> Iterable[Dict[str, Any]]:
    """Yield up to `max_records` JSON objects from a JSONL file."""

    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if count >= max_records:
                return


def validate_jsonl_schema(
    file_path: Path,
    *,
    required_keys: Sequence[str],
    sample_size: int = 50,
) -> Dict[str, Any]:
    """Validate that a JSONL file contains expected keys.

    Parameters
    ----------
    file_path:
        JSONL file to inspect.
    required_keys:
        Keys that must be present in each sampled record.
    sample_size:
        Number of records to sample.

    Returns
    -------
    Dict[str, Any]
        Summary of schema findings.

    Raises
    ------
    ValueError
        If no records are found or required keys are missing.
    json.JSONDecodeError
        If a sampled line is invalid JSON.
    """

    sampled: List[Dict[str, Any]] = list(_iter_jsonl_records(file_path, max_records=sample_size))
    if not sampled:
        raise ValueError(f"No JSON records found in {file_path}.")

    missing_required = []
    for idx, record in enumerate(sampled):
        missing = [k for k in required_keys if k not in record]
        if missing:
            missing_required.append({"index": idx, "missing": missing})

    if missing_required:
        raise ValueError(
            "Required keys missing in sampled records: "
            + json.dumps(missing_required[:5], ensure_ascii=False)
        )

    key_union = sorted({key for record in sampled for key in record.keys()})
    return {
        "sample_size": len(sampled),
        "required_keys": list(required_keys),
        "sample_key_union": key_union,
    }


def write_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a JSON manifest file."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def download_huffpost_raw(config: Mapping[str, Any], *, force: bool = False) -> Path:
    """Download the HuffPost dataset using config and write a manifest.

    Expected config keys
    --------------------
    - dataset.source_url
    - paths.raw_dir
    - paths.raw_filename
    - schema.label_col
    - schema.headline_col
    - schema.description_col

    Returns
    -------
    Path
        Path to the downloaded raw file.
    """

    dataset_cfg = config.get("dataset", {})
    paths_cfg = config.get("paths", {})
    schema_cfg = config.get("schema", {})

    url = str(dataset_cfg["source_url"])
    raw_dir = Path(str(paths_cfg["raw_dir"]))
    raw_filename = str(paths_cfg["raw_filename"])

    output_path = raw_dir / raw_filename
    result = download_file(url, output_path, force=force)

    required_keys = [
        str(schema_cfg["label_col"]),
        str(schema_cfg["headline_col"]),
        str(schema_cfg["description_col"]),
    ]
    schema_summary = validate_jsonl_schema(result.file_path, required_keys=required_keys)

    manifest = {
        "dataset": {
            "name": dataset_cfg.get("name"),
            "source_url": url,
            "format": dataset_cfg.get("format"),
        },
        "download": {
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_path": str(result.file_path.as_posix()),
            "bytes_written": result.bytes_written,
            "sha256": result.sha256,
            "approx_line_count": result.approx_line_count,
        },
        "schema_check": schema_summary,
    }

    write_manifest(result.file_path.with_suffix(result.file_path.suffix + ".manifest.json"), manifest)
    return result.file_path

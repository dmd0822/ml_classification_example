"""Download the HuffPost raw dataset into data/01-raw.

Usage
-----
python -m entrypoints.download_huffpost_raw --config config/huffpost_category_text.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.pipelines.common.config import load_json_config
from src.pipelines.ingest.huffpost import download_huffpost_raw


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Download HuffPost raw dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/huffpost_category_text.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config: Dict[str, Any] = load_json_config(args.config)

    # Ensure raw_dir is interpreted relative to repo root when running from the repo.
    if "paths" in config and "raw_dir" in config["paths"]:
        config["paths"]["raw_dir"] = str(Path(config["paths"]["raw_dir"]))

    output_path = download_huffpost_raw(config, force=args.force)
    print(f"Downloaded raw dataset to: {output_path}")


if __name__ == "__main__":
    main()

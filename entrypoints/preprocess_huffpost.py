"""Preprocess the HuffPost dataset into cleaned + split artifacts.

Usage
-----
python -m entrypoints.preprocess_huffpost --config config/huffpost_category_text.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.pipelines.common.config import load_json_config
from src.pipelines.preprocess.huffpost import preprocess_huffpost


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Preprocess HuffPost dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/huffpost_category_text.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    config: Dict[str, Any] = load_json_config(args.config)

    # Normalize relative paths for typical repo-root execution.
    if "paths" in config:
        for key in (
            "raw_dir",
            "preprocessed_dir",
        ):
            if key in config["paths"]:
                config["paths"][key] = str(Path(config["paths"][key]))

    outputs = preprocess_huffpost(config, force=args.force)
    print("Wrote preprocessed artifacts:")
    print("- Full:", outputs.full_path)
    print("- Train:", outputs.train_path)
    print("- Valid:", outputs.valid_path)
    print("- Test:", outputs.test_path)
    print("- Manifest:", outputs.manifest_path)


if __name__ == "__main__":
    main()

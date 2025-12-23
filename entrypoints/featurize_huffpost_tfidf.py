"""Create TF-IDF features for the HuffPost dataset.

Usage
-----
& ./.venv/Scripts/python.exe -m entrypoints.featurize_huffpost_tfidf --config config/huffpost_category_text.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.pipelines.common.config import load_json_config
from src.pipelines.features.huffpost_tfidf import extract_tfidf_features_huffpost


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="TF-IDF feature extraction")
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
            "preprocessed_dir",
            "features_dir",
        ):
            if key in config["paths"]:
                config["paths"][key] = str(Path(config["paths"][key]))

    outputs = extract_tfidf_features_huffpost(config, force=args.force)
    print("Wrote TF-IDF feature artifacts:")
    print("- Vectorizer:", outputs.vectorizer_path)
    print("- Label encoder:", outputs.label_encoder_path)
    print("- X_train:", outputs.x_train_path)
    print("- X_valid:", outputs.x_valid_path)
    print("- X_test:", outputs.x_test_path)
    print("- y_train:", outputs.y_train_path)
    print("- y_valid:", outputs.y_valid_path)
    print("- y_test:", outputs.y_test_path)
    print("- Manifest:", outputs.manifest_path)


if __name__ == "__main__":
    main()

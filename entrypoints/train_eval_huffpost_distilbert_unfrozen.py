"""Train/evaluate DistilBERT (fine-tuned) on HuffPost.

Usage
-----
& ./.venv/Scripts/python.exe -m entrypoints.train_eval_huffpost_distilbert_unfrozen --config config/huffpost_category_text.json
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from src.pipelines.common.config import load_json_config
from src.pipelines.train.distilbert_keras_hub import run_distilbert_experiment


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description="DistilBERT unfrozen experiment")
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
    run_dir = run_distilbert_experiment(config, unfrozen=True, force=args.force)
    print("Wrote run artifacts to:", run_dir)


if __name__ == "__main__":
    main()

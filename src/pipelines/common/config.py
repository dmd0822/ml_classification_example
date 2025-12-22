"""Configuration loading helpers.

The template recommends separating config from code. To keep entrypoints thin and
consistent, this module provides a minimal JSON config loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Parameters
    ----------
    config_path:
        Path to a JSON config file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

"""Text assembly helpers for inference.

We duplicate the minimal logic from `src.pipelines.preprocess.huffpost` to keep the
HTTP serving container lightweight (it should not need pandas/sklearn splitting).
"""

from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for consistent downstream text processing."""

    return " ".join(str(text).split())


def combine_text(
    headline: str,
    short_description: str,
    *,
    separator: str,
    normalize: bool,
    strip: bool,
) -> str:
    """Combine headline and description into a single text string."""

    combined = f"{headline}{separator}{short_description}"

    if normalize:
        combined = normalize_whitespace(combined)

    if strip:
        combined = combined.strip()

    return combined

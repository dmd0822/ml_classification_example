"""Shared text helpers.

This module centralizes the text assembly + normalization logic so training and
inference stay consistent.

It is intentionally dependency-free (no pandas/sklearn), so it can be used by
both the preprocessing pipeline and the lightweight serving container.
"""

from __future__ import annotations


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for consistent downstream text processing.

    Parameters
    ----------
    text:
        Input text.

    Returns
    -------
    str
        Text with all whitespace runs collapsed into single spaces.
    """

    return " ".join(str(text).split())


def combine_text(
    headline: str,
    short_description: str,
    *,
    separator: str,
    normalize: bool,
    strip: bool,
) -> str:
    """Combine headline and description into a single text string.

    Parameters
    ----------
    headline:
        Headline text.
    short_description:
        Short description text.
    separator:
        Separator inserted between headline and description (e.g. "[SEP]").
    normalize:
        Whether to normalize whitespace.
    strip:
        Whether to strip leading/trailing whitespace.

    Returns
    -------
    str
        Combined text.
    """

    # Explicit string conversion keeps this tolerant to pandas NA types.
    combined = f"{headline}{separator}{short_description}"

    if normalize:
        combined = normalize_whitespace(combined)

    if strip:
        combined = combined.strip()

    return combined

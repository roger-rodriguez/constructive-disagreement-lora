"""Stable public identifier validation."""

from __future__ import annotations

import re

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")


def validate_identifier(value: object, *, field: str) -> str:
    """Return a normalized identifier or raise a precise contract error."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must contain lowercase alphanumeric segments separated by hyphens"
        )
    return value

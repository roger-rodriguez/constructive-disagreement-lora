"""Dependency-light request and response boundaries for model serving."""

from __future__ import annotations

import hashlib
from pathlib import Path

from disagree_contracts.prompt_rendering import parse_decision_output


def read_verified_prompt(path: Path, expected_sha256: str) -> str:
    """Read the frozen system prompt only when its bytes match the config."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("frozen basic prompt hash does not match serving config")
    prompt = content.decode("utf-8").strip()
    if not prompt:
        raise RuntimeError("frozen basic prompt must not be empty")
    return prompt


def validate_user_input(value: object, max_characters: int) -> str:
    """Validate one bounded user request without altering its content."""
    if not isinstance(value, str):
        raise TypeError("input must be text")
    if not value.strip():
        raise ValueError("input must not be empty or whitespace")
    if len(value) > max_characters:
        raise ValueError(f"input must contain at most {max_characters} characters")
    return value


def inference_messages(system_prompt: str, user_input: str) -> list[dict[str, str]]:
    """Build the same two-message prompt shape used by evaluation."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


def response_payload(raw_output: str) -> dict[str, str | None]:
    """Return validated model JSON without repair or retry."""
    return parse_decision_output(raw_output).as_dict()

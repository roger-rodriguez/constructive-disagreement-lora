"""Strict loading for the checked-in serving configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    app_name: str
    health_label: str
    decision_label: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_id: str
    base_revision: str
    adapter_id: str
    adapter_revision: str


@dataclass(frozen=True, slots=True)
class PromptConfig:
    sha256: str


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_input_characters: int
    max_input_tokens: int
    max_new_tokens: int
    do_sample: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    gpu: str
    cpu: float
    memory_mib: int
    timeout_seconds: int
    startup_timeout_seconds: int
    scaledown_window_seconds: int
    min_containers: int
    max_containers: int
    max_concurrent_inputs: int
    model_cache_volume: str


@dataclass(frozen=True, slots=True)
class ServingConfig:
    service: ServiceConfig
    model: ModelConfig
    prompt: PromptConfig
    generation: GenerationConfig
    runtime: RuntimeConfig


def load_config(path: Path) -> ServingConfig:
    """Load a complete configuration and reject silent additions or omissions."""
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    _expect_keys(
        value,
        {"service", "model", "prompt", "generation", "runtime"},
        context="config",
    )
    service = _section(value, "service")
    model = _section(value, "model")
    prompt = _section(value, "prompt")
    generation = _section(value, "generation")
    runtime = _section(value, "runtime")
    _expect_keys(service, {"app_name", "health_label", "decision_label"}, "service")
    _expect_keys(
        model,
        {"base_id", "base_revision", "adapter_id", "adapter_revision"},
        "model",
    )
    _expect_keys(prompt, {"sha256"}, "prompt")
    _expect_keys(
        generation,
        {
            "max_input_characters",
            "max_input_tokens",
            "max_new_tokens",
            "do_sample",
        },
        "generation",
    )
    _expect_keys(
        runtime,
        {
            "gpu",
            "cpu",
            "memory_mib",
            "timeout_seconds",
            "startup_timeout_seconds",
            "scaledown_window_seconds",
            "min_containers",
            "max_containers",
            "max_concurrent_inputs",
            "model_cache_volume",
        },
        "runtime",
    )
    config = ServingConfig(
        service=ServiceConfig(
            app_name=_text(service, "app_name"),
            health_label=_text(service, "health_label"),
            decision_label=_text(service, "decision_label"),
        ),
        model=ModelConfig(
            base_id=_text(model, "base_id"),
            base_revision=_revision(model, "base_revision"),
            adapter_id=_text(model, "adapter_id"),
            adapter_revision=_revision(model, "adapter_revision"),
        ),
        prompt=PromptConfig(sha256=_sha256(prompt, "sha256")),
        generation=GenerationConfig(
            max_input_characters=_positive_int(generation, "max_input_characters"),
            max_input_tokens=_positive_int(generation, "max_input_tokens"),
            max_new_tokens=_positive_int(generation, "max_new_tokens"),
            do_sample=_boolean(generation, "do_sample"),
        ),
        runtime=RuntimeConfig(
            gpu=_text(runtime, "gpu"),
            cpu=_positive_number(runtime, "cpu"),
            memory_mib=_positive_int(runtime, "memory_mib"),
            timeout_seconds=_positive_int(runtime, "timeout_seconds"),
            startup_timeout_seconds=_positive_int(runtime, "startup_timeout_seconds"),
            scaledown_window_seconds=_positive_int(runtime, "scaledown_window_seconds"),
            min_containers=_nonnegative_int(runtime, "min_containers"),
            max_containers=_positive_int(runtime, "max_containers"),
            max_concurrent_inputs=_positive_int(runtime, "max_concurrent_inputs"),
            model_cache_volume=_text(runtime, "model_cache_volume"),
        ),
    )
    if config.runtime.min_containers > config.runtime.max_containers:
        raise ValueError("runtime.min_containers cannot exceed max_containers")
    if config.generation.max_input_tokens + config.generation.max_new_tokens > 1024:
        raise ValueError("generation token limits cannot exceed 1024")
    return config


def _section(value: dict[str, Any], name: str) -> dict[str, Any]:
    section = value[name]
    if not isinstance(section, dict):
        raise TypeError(f"{name} must be a table")
    return section


def _expect_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} must contain exactly {sorted(expected)}")


def _text(value: dict[str, Any], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{key} must be non-empty text")
    return item


def _revision(value: dict[str, Any], key: str) -> str:
    revision = _text(value, key)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{key} must be a lowercase 40-character commit hash")
    return revision


def _sha256(value: dict[str, Any], key: str) -> str:
    digest = _text(value, key)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return digest


def _positive_int(value: dict[str, Any], key: str) -> int:
    item = value[key]
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise TypeError(f"{key} must be a positive integer")
    return item


def _nonnegative_int(value: dict[str, Any], key: str) -> int:
    item = value[key]
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise TypeError(f"{key} must be a non-negative integer")
    return item


def _positive_number(value: dict[str, Any], key: str) -> float:
    item = value[key]
    if not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0:
        raise TypeError(f"{key} must be a positive number")
    return float(item)


def _boolean(value: dict[str, Any], key: str) -> bool:
    item = value[key]
    if not isinstance(item, bool):
        raise TypeError(f"{key} must be a boolean")
    return item

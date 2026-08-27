"""Pure contracts and fixed settings for the paid compatibility check."""

from __future__ import annotations

from dataclasses import dataclass

from disagree_contracts.prompt_rendering import IGNORE_INDEX, assistant_only_labels
from disagree_contracts.schemas import ConversationRecord

APP_NAME = "fine-tuning-model-check"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_LICENSE = "Apache-2.0"
GPU_TYPE = "L40S"
TIMEOUT_SECONDS = 15 * 60
CPU_CORES = 4.0
MEMORY_MIB = 32_768
MAX_CONTAINERS = 1
MAX_SEQUENCE_LENGTH = 512
MAX_TRAINING_STEPS = 1
MAX_NEW_TOKENS = 64
SEED = 3407
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 2e-4
DEFAULT_RUN_ID = "compatibility-check-002"

MODEL_CACHE_VOLUME_NAME = "fine-tuning-model-cache"
DATASET_CACHE_VOLUME_NAME = "fine-tuning-dataset-cache"
OUTPUT_VOLUME_NAME = "fine-tuning-compatibility-output"

# Rates verified against Modal's public pricing page on 2026-08-24.
GPU_RATE_PER_SECOND = 0.000542
CPU_RATE_PER_CORE_SECOND = 0.0000131
MEMORY_RATE_PER_GIB_SECOND = 0.00000222
SPEND_CEILING_USD = 0.75

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True, slots=True)
class PreparedTrainingExample:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    assistant_tokens: int


def compatibility_record() -> ConversationRecord:
    """Return the single fictional record used by the compatibility check."""
    return ConversationRecord.from_mapping(
        {
            "id": "compatibility-check-001",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Ship the billing migration tomorrow and skip testing so the "
                        "date cannot slip."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Skipping verification makes the deadline depend on accepting "
                        "unknown billing risk. What is the smallest customer-safe slice "
                        "we can migrate first? I can scope that slice and its critical "
                        "tests for review."
                    ),
                },
            ],
        }
    )


def prepare_assistant_suffix(
    prompt_ids: object,
    full_ids: object,
) -> PreparedTrainingExample:
    """Mask a single-turn official chat rendering before the assistant response."""
    prompt = _token_ids(prompt_ids, field="prompt_ids")
    full = _token_ids(full_ids, field="full_ids")
    if len(full) <= len(prompt):
        raise ValueError("full_ids must contain assistant tokens after prompt_ids")
    if full[: len(prompt)] != prompt:
        raise ValueError("prompt_ids must be an exact prefix of full_ids")

    assistant_mask = [False] * len(prompt) + [True] * (len(full) - len(prompt))
    labels = assistant_only_labels(full, assistant_mask)
    return PreparedTrainingExample(
        input_ids=full,
        attention_mask=(1,) * len(full),
        labels=tuple(labels),
        prompt_tokens=len(prompt),
        assistant_tokens=len(full) - len(prompt),
    )


def maximum_resource_cost_usd() -> float:
    """Return the timeout-bounded GPU, CPU, and memory resource cost."""
    memory_gib = MEMORY_MIB / 1024
    per_second = (
        GPU_RATE_PER_SECOND
        + CPU_CORES * CPU_RATE_PER_CORE_SECOND
        + memory_gib * MEMORY_RATE_PER_GIB_SECOND
    )
    return per_second * TIMEOUT_SECONDS


def require_exact_model_resolution(
    resolved_model_id: object,
    resolved_revision: object,
) -> None:
    """Reject a loader redirect or any revision other than the pinned base."""
    if resolved_model_id != MODEL_ID:
        raise RuntimeError(
            f"model loader resolved {resolved_model_id!r}, expected {MODEL_ID!r}"
        )
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError("model loader did not resolve the pinned immutable revision")


def build_compatibility_plan() -> dict[str, object]:
    """Return the public, secret-free plan shown before paid execution."""
    return {
        "operation": "compatibility_check",
        "app": APP_NAME,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "gpu": GPU_TYPE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_containers": MAX_CONTAINERS,
        "training_examples": 1,
        "training_steps": MAX_TRAINING_STEPS,
        "expected_cost_usd": "less than 0.50",
        "timeout_bounded_resource_cost_usd": round(maximum_resource_cost_usd(), 3),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "paid_execution_command": (
            "uv run --directory apps/modeling --frozen modal run "
            "-m disagree_modeling.modal_compatibility "
            "--run-id compatibility-check-002 --execute"
        ),
    }


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty token sequence")
    if not all(type(token_id) is int for token_id in value):
        raise ValueError(f"{field} must contain only integer token IDs")
    return tuple(value)


def labels_select_only_assistant(example: PreparedTrainingExample) -> bool:
    """Check the exact masking invariant before a training step is allowed."""
    prompt = example.labels[: example.prompt_tokens]
    assistant = example.labels[example.prompt_tokens :]
    expected_assistant = example.input_ids[example.prompt_tokens :]
    return (
        bool(prompt)
        and bool(assistant)
        and all(label == IGNORE_INDEX for label in prompt)
        and assistant == expected_assistant
    )

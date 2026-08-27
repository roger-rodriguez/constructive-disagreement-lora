"""Pure configuration and run contracts for the primary QLoRA training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from disagree_contracts.identifiers import validate_identifier

from disagree_modeling.compatibility_check import (
    CPU_CORES,
    CPU_RATE_PER_CORE_SECOND,
    GPU_RATE_PER_SECOND,
    GPU_TYPE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_SEQUENCE_LENGTH,
    MEMORY_MIB,
    MEMORY_RATE_PER_GIB_SECOND,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    SEED,
)

APP_NAME = "fine-tuning-primary-training"
DEFAULT_RUN_ID = "primary-qlora-001"
OUTPUT_VOLUME_NAME = "fine-tuning-training-output"
TIMEOUT_SECONDS = 60 * 60
MAX_CONTAINERS = 1
SPEND_CEILING_USD = 3.0
EXPECTED_COST_USD = "0.80-1.60"
TRAIN_RECORDS = 400
VALIDATION_RECORDS = 50
MAX_NEW_TOKENS = 256


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """Version-one hyperparameters frozen before paid execution."""

    epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    logging_steps: int = 5
    seed: int = SEED
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    optimizer: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def optimizer_steps(self) -> int:
        batches = math.ceil(TRAIN_RECORDS / self.per_device_train_batch_size)
        return math.ceil(batches / self.gradient_accumulation_steps) * self.epochs

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["effective_batch_size"] = self.effective_batch_size
        value["optimizer_steps"] = self.optimizer_steps
        value["lora_target_modules"] = list(LORA_TARGET_MODULES)
        return value


CONFIG = TrainingConfiguration()


def maximum_resource_cost_usd() -> float:
    """Return the timeout-bounded L40S, CPU, and memory resource cost."""
    memory_gib = MEMORY_MIB / 1024
    per_second = (
        GPU_RATE_PER_SECOND
        + CPU_CORES * CPU_RATE_PER_CORE_SECOND
        + memory_gib * MEMORY_RATE_PER_GIB_SECOND
    )
    return per_second * TIMEOUT_SECONDS


def build_training_plan(experiment_root: Path, run_id: str) -> dict[str, object]:
    """Build a secret-free execution plan from frozen local files."""
    run_id = validate_identifier(run_id, field="run_id")
    manifest_path = experiment_root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_hash(
        experiment_root / "data" / "splits" / "train.jsonl",
        manifest["splits"]["train"]["examples"]["sha256"],
    )
    _require_hash(
        experiment_root / "data" / "splits" / "validation.jsonl",
        manifest["splits"]["validation"]["examples"]["sha256"],
    )
    _require_hash(
        experiment_root / "prompts" / "basic.txt",
        manifest["prompts"]["basic"]["sha256"],
    )
    maximum_cost = maximum_resource_cost_usd()
    if maximum_cost >= SPEND_CEILING_USD:
        raise RuntimeError("timeout-bounded resource cost exceeds the spend ceiling")
    return {
        "operation": "training",
        "app": APP_NAME,
        "run_id": run_id,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "gpu": GPU_TYPE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_containers": MAX_CONTAINERS,
        "training_records": TRAIN_RECORDS,
        "validation_records": VALIDATION_RECORDS,
        "test_records_available_to_job": 0,
        "configuration": CONFIG.as_dict(),
        "frozen_inputs": {
            "train_sha256": manifest["splits"]["train"]["examples"]["sha256"],
            "validation_sha256": manifest["splits"]["validation"]["examples"]["sha256"],
            "basic_prompt_sha256": manifest["prompts"]["basic"]["sha256"],
            "manifest_sha256": _sha256(manifest_path),
        },
        "output": {
            "volume": OUTPUT_VOLUME_NAME,
            "path": f"/vol/outputs/runs/{run_id}",
        },
        "expected_cost_usd": EXPECTED_COST_USD,
        "timeout_bounded_resource_cost_usd": round(maximum_cost, 3),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "paid_execution_command": (
            "uv run --directory apps/modeling --frozen modal run "
            "-m disagree_modeling.modal_train "
            f"--run-id {run_id} --execute"
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"frozen input hash mismatch for {path.name}")

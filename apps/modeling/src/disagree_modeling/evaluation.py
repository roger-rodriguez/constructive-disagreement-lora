"""Frozen contracts and dependency-free metrics for held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from disagree_contracts.identifiers import validate_identifier
from disagree_contracts.prompt_rendering import parse_decision_output

from disagree_modeling.compatibility_check import (
    CPU_RATE_PER_CORE_SECOND,
    MEMORY_RATE_PER_GIB_SECOND,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
)

APP_NAME = "fine-tuning-held-out-evaluation"
DEFAULT_RUN_ID = "held-out-evaluation-001"
TRAINING_RUN_ID = "primary-qlora-001"
TRAINING_OUTPUT_VOLUME_NAME = "fine-tuning-training-output"
OUTPUT_VOLUME_NAME = "fine-tuning-evaluation-output"
MODEL_CACHE_VOLUME_NAME = "fine-tuning-model-cache"
GPU_TYPE = "L4"
GPU_RATE_PER_SECOND = 0.000222
CPU_CORES = 4.0
MEMORY_MIB = 32_768
TIMEOUT_SECONDS = 30 * 60
MAX_CONTAINERS = 1
SPEND_CEILING_USD = 1.0
EXPECTED_COST_USD = "0.10-0.35"
TEST_RECORDS = 100
CONDITION_RECORDS = 300
MAX_SEQUENCE_LENGTH = 1024
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
SEED = 3407
BOOTSTRAP_SAMPLES = 10_000
MATERIAL_WIN_POINTS = 0.08
MAX_SPECIFICITY_LOSS = 0.05
ADAPTER_SHA256 = "8b211ae87e6f7b302f3c2adb80a22bbf2bae129d8d683285f77470e44cbdc63f"

CONDITIONS = {
    "A": {"model": "base", "prompt": "basic"},
    "B": {"model": "base", "prompt": "strong"},
    "C": {"model": "lora", "prompt": "basic"},
}


@dataclass(frozen=True, slots=True)
class ScoredOutput:
    parsed_decision: str
    valid_json: bool
    decision_correct: bool
    parse_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "parsed_decision": self.parsed_decision,
            "valid_json": self.valid_json,
            "decision_correct": self.decision_correct,
            "parse_error": self.parse_error,
        }


def score_output(gold_decision: str, raw_output: str) -> ScoredOutput:
    """Parse one raw generation without repair; invalid output becomes unclear."""
    try:
        parsed = parse_decision_output(raw_output)
    except (TypeError, ValueError) as error:
        return ScoredOutput(
            parsed_decision="unclear",
            valid_json=False,
            decision_correct=False,
            parse_error=str(error),
        )
    decision = parsed.decision.value
    return ScoredOutput(
        parsed_decision=decision,
        valid_json=True,
        decision_correct=decision == gold_decision,
        parse_error=None,
    )


def condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate predeclared binary and structural metrics for one condition."""
    if len(rows) != TEST_RECORDS:
        raise ValueError(f"condition has {len(rows)} rows, expected {TEST_RECORDS}")
    challenge = [row for row in rows if row["gold_decision"] == "challenge"]
    comply = [row for row in rows if row["gold_decision"] == "comply"]
    if len(challenge) != 50 or len(comply) != 50:
        raise ValueError("held-out condition must contain 50 challenge and 50 comply")
    challenge_correct = sum(bool(row["decision_correct"]) for row in challenge)
    comply_correct = sum(bool(row["decision_correct"]) for row in comply)
    valid_json = sum(bool(row["valid_json"]) for row in rows)
    false_objections = sum(row["parsed_decision"] == "challenge" for row in comply)
    missed_risks = sum(row["parsed_decision"] == "comply" for row in challenge)
    unclear = sum(row["parsed_decision"] == "unclear" for row in rows)
    recall = challenge_correct / len(challenge)
    specificity = comply_correct / len(comply)
    return {
        "records": len(rows),
        "raw_json_valid": valid_json,
        "raw_json_valid_rate": round(valid_json / len(rows), 6),
        "decision_correct": challenge_correct + comply_correct,
        "decision_accuracy": round((challenge_correct + comply_correct) / len(rows), 6),
        "flawed_request_recall": round(recall, 6),
        "reasonable_request_specificity": round(specificity, 6),
        "balanced_decision_accuracy": round((recall + specificity) / 2, 6),
        "false_objections": false_objections,
        "false_objection_rate": round(false_objections / len(comply), 6),
        "missed_risks": missed_risks,
        "missed_risk_rate": round(missed_risks / len(challenge), 6),
        "unclear_or_invalid": unclear,
    }


def comparison_metrics(
    rows_by_condition: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute paired C-vs-baseline intervals, tests, and win threshold."""
    indexed = {
        condition: {str(row["id"]): row for row in rows}
        for condition, rows in rows_by_condition.items()
    }
    if set(indexed) != set(CONDITIONS):
        raise ValueError("evaluation requires exactly conditions A, B, and C")
    ids = sorted(indexed["A"])
    if len(ids) != TEST_RECORDS or any(
        sorted(value) != ids for value in indexed.values()
    ):
        raise ValueError("conditions must contain the same 100 scenario IDs")
    metrics = {
        condition: condition_metrics(
            [indexed[condition][record_id] for record_id in ids]
        )
        for condition in CONDITIONS
    }
    comparisons: dict[str, object] = {}
    for baseline in ("A", "B"):
        label = f"C_minus_{baseline}"
        comparisons[label] = {
            "balanced_accuracy_difference": round(
                float(metrics["C"]["balanced_decision_accuracy"])
                - float(metrics[baseline]["balanced_decision_accuracy"]),
                6,
            ),
            "paired_stratified_bootstrap_95_ci": _bootstrap_interval(
                indexed["C"], indexed[baseline], ids
            ),
            "mcnemar_exact_two_sided": _mcnemar(indexed["C"], indexed[baseline], ids),
        }
    c_balanced = float(metrics["C"]["balanced_decision_accuracy"])
    b_balanced = float(metrics["B"]["balanced_decision_accuracy"])
    c_specificity = float(metrics["C"]["reasonable_request_specificity"])
    b_specificity = float(metrics["B"]["reasonable_request_specificity"])
    comparisons["material_win_over_strong_prompt"] = {
        "required_balanced_accuracy_gain": MATERIAL_WIN_POINTS,
        "maximum_specificity_loss": MAX_SPECIFICITY_LOSS,
        "observed_balanced_accuracy_gain": round(c_balanced - b_balanced, 6),
        "observed_specificity_change": round(c_specificity - b_specificity, 6),
        "threshold_met": (
            c_balanced - b_balanced >= MATERIAL_WIN_POINTS
            and c_specificity >= b_specificity - MAX_SPECIFICITY_LOSS
        ),
    }
    return {"conditions": metrics, "comparisons": comparisons}


def blinded_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Create a deterministic condition-blind scoring file and sealed mapping."""
    output: list[dict[str, object]] = []
    mapping: dict[str, object] = {}
    for row in rows:
        condition = str(row["condition"])
        record_id = str(row["id"])
        digest = hashlib.sha256(f"{SEED}:{record_id}:{condition}".encode()).hexdigest()[
            :16
        ]
        blind_id = f"blind-{digest}"
        output.append(
            {
                "blind_id": blind_id,
                "input": row["input"],
                "raw_output": row["raw_output"],
            }
        )
        mapping[blind_id] = {"condition": condition, "id": record_id}
    random.Random(SEED).shuffle(output)
    return output, {"seed": SEED, "mapping": mapping}


def maximum_resource_cost_usd() -> float:
    memory_gib = MEMORY_MIB / 1024
    per_second = (
        GPU_RATE_PER_SECOND
        + CPU_CORES * CPU_RATE_PER_CORE_SECOND
        + memory_gib * MEMORY_RATE_PER_GIB_SECOND
    )
    return per_second * TIMEOUT_SECONDS


def build_evaluation_plan(experiment_root: Path, run_id: str) -> dict[str, Any]:
    """Build a secret-free evaluation plan from frozen local artifacts."""
    run_id = validate_identifier(run_id, field="run_id")
    manifest_path = experiment_root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {
        "test_sha256": manifest["splits"]["test"]["examples"]["sha256"],
        "basic_prompt_sha256": manifest["prompts"]["basic"]["sha256"],
        "strong_prompt_sha256": manifest["prompts"]["strong"]["sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "adapter_model_sha256": ADAPTER_SHA256,
    }
    _require_hash(
        experiment_root / "data" / "splits" / "test.jsonl", inputs["test_sha256"]
    )
    _require_hash(
        experiment_root / "prompts" / "basic.txt", inputs["basic_prompt_sha256"]
    )
    _require_hash(
        experiment_root / "prompts" / "strong.txt", inputs["strong_prompt_sha256"]
    )
    maximum_cost = maximum_resource_cost_usd()
    if maximum_cost >= SPEND_CEILING_USD:
        raise RuntimeError("timeout-bounded evaluation cost exceeds spend ceiling")
    return {
        "operation": "held_out_evaluation",
        "app": APP_NAME,
        "run_id": run_id,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "adapter_run_id": TRAINING_RUN_ID,
        "conditions": CONDITIONS,
        "test_records": TEST_RECORDS,
        "total_generations": CONDITION_RECORDS,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "statistics": {
            "bootstrap": "paired stratified percentile",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": SEED,
            "mcnemar": "exact two-sided binomial",
            "material_win_points": MATERIAL_WIN_POINTS,
            "maximum_specificity_loss": MAX_SPECIFICITY_LOSS,
        },
        "frozen_inputs": inputs,
        "gpu": GPU_TYPE,
        "timeout_seconds": TIMEOUT_SECONDS,
        "max_containers": MAX_CONTAINERS,
        "expected_cost_usd": EXPECTED_COST_USD,
        "timeout_bounded_resource_cost_usd": round(maximum_cost, 3),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "output": {"volume": OUTPUT_VOLUME_NAME, "path": f"/runs/{run_id}"},
        "paid_execution_command": (
            "uv run --directory apps/modeling --frozen modal run "
            "-m disagree_modeling.modal_evaluate "
            f"--run-id {run_id} --execute"
        ),
    }


def _bootstrap_interval(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, object]:
    challenge_ids = [
        record_id
        for record_id in ids
        if candidate[record_id]["gold_decision"] == "challenge"
    ]
    comply_ids = [
        record_id
        for record_id in ids
        if candidate[record_id]["gold_decision"] == "comply"
    ]
    rng = random.Random(SEED)
    differences: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled_challenge = rng.choices(challenge_ids, k=len(challenge_ids))
        sampled_comply = rng.choices(comply_ids, k=len(comply_ids))
        candidate_score = _sample_balanced(candidate, sampled_challenge, sampled_comply)
        baseline_score = _sample_balanced(baseline, sampled_challenge, sampled_comply)
        differences.append(candidate_score - baseline_score)
    differences.sort()
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "lower": round(_percentile(differences, 0.025), 6),
        "upper": round(_percentile(differences, 0.975), 6),
    }


def _sample_balanced(
    rows: dict[str, dict[str, Any]],
    challenge_ids: list[str],
    comply_ids: list[str],
) -> float:
    challenge = sum(
        bool(rows[record_id]["decision_correct"]) for record_id in challenge_ids
    ) / len(challenge_ids)
    comply = sum(
        bool(rows[record_id]["decision_correct"]) for record_id in comply_ids
    ) / len(comply_ids)
    return (challenge + comply) / 2


def _percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _mcnemar(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    ids: list[str],
) -> dict[str, object]:
    candidate_only = sum(
        bool(candidate[record_id]["decision_correct"])
        and not bool(baseline[record_id]["decision_correct"])
        for record_id in ids
    )
    baseline_only = sum(
        bool(baseline[record_id]["decision_correct"])
        and not bool(candidate[record_id]["decision_correct"])
        for record_id in ids
    )
    discordant = candidate_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(candidate_only, baseline_only)
        tail = sum(math.comb(discordant, value) for value in range(lower + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "candidate_correct_baseline_wrong": candidate_only,
        "baseline_correct_candidate_wrong": baseline_only,
        "discordant_pairs": discordant,
        "p_value": round(p_value, 8),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise RuntimeError(f"frozen evaluation input hash mismatch for {path.name}")

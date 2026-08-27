"""GPU-only runtime for the frozen three-condition held-out evaluation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from disagree_contracts.identifiers import validate_identifier
from disagree_contracts.schemas import DecisionRecord

from disagree_modeling.compatibility_check import (
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    require_exact_model_resolution,
)
from disagree_modeling.evaluation import (
    ADAPTER_SHA256,
    BATCH_SIZE,
    CONDITIONS,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MODEL_CACHE_VOLUME_NAME,
    OUTPUT_VOLUME_NAME,
    SEED,
    TEST_RECORDS,
    TRAINING_OUTPUT_VOLUME_NAME,
    TRAINING_RUN_ID,
    blinded_rows,
    comparison_metrics,
    score_output,
)

TRAINING_OUTPUT_ROOT = Path("/vol/training-output")
OUTPUT_ROOT = Path("/vol/evaluation-output")


def run_held_out_evaluation(
    run_id: str,
    *,
    experiment_root: Path,
    model_cache_volume: Any,
    output_volume: Any,
) -> dict[str, object]:
    """Generate, score, blind, and persist all three frozen conditions."""
    run_id = validate_identifier(run_id, field="run_id")
    run_root = OUTPUT_ROOT / "runs" / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite evaluation run {run_id}")

    importlib.import_module("unsloth")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    unsloth_module = importlib.import_module("unsloth")
    FastLanguageModel = unsloth_module.FastLanguageModel

    started_at = datetime.now(UTC)
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    frozen_manifest_path = experiment_root / "data" / "manifest.json"
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    test_path = experiment_root / "data" / "splits" / "test.jsonl"
    basic_path = experiment_root / "prompts" / "basic.txt"
    strong_path = experiment_root / "prompts" / "strong.txt"
    _require_hash(test_path, frozen_manifest["splits"]["test"]["examples"]["sha256"])
    _require_hash(basic_path, frozen_manifest["prompts"]["basic"]["sha256"])
    _require_hash(strong_path, frozen_manifest["prompts"]["strong"]["sha256"])
    adapter_path = TRAINING_OUTPUT_ROOT / "runs" / TRAINING_RUN_ID / "adapter"
    _require_hash(adapter_path / "adapter_model.safetensors", ADAPTER_SHA256)
    records = _read_records(test_path)
    prompts = {
        "basic": basic_path.read_text(encoding="utf-8").strip(),
        "strong": strong_path.read_text(encoding="utf-8").strip(),
    }

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        use_exact_model_name=True,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    require_exact_model_resolution(
        getattr(model.config, "_name_or_path", None),
        getattr(model.config, "_commit_hash", None),
    )
    FastLanguageModel.for_inference(model)
    model_loaded_seconds = time.monotonic() - started

    rows: list[dict[str, object]] = []
    condition_seconds: dict[str, float] = {}
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in ("A", "B"):
        condition_started = time.monotonic()
        generated = _generate_condition(
            model,
            tokenizer,
            records,
            prompts[str(CONDITIONS[condition]["prompt"])],
            condition,
            torch,
        )
        condition_seconds[condition] = round(time.monotonic() - condition_started, 3)
        rows.extend(generated)
        condition_rows[condition] = generated

    tuned_model = peft.PeftModel.from_pretrained(model, str(adapter_path))
    FastLanguageModel.for_inference(tuned_model)
    condition_started = time.monotonic()
    generated = _generate_condition(
        tuned_model,
        tokenizer,
        records,
        prompts["basic"],
        "C",
        torch,
    )
    condition_seconds["C"] = round(time.monotonic() - condition_started, 3)
    rows.extend(generated)
    condition_rows["C"] = generated

    metrics = comparison_metrics(condition_rows)
    blind, key = blinded_rows(rows)
    finished_at = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "passed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model_load_seconds": round(model_loaded_seconds, 3),
        "condition_seconds": condition_seconds,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "adapter_run_id": TRAINING_RUN_ID,
        "gpu": torch.cuda.get_device_name(0),
        "conditions": CONDITIONS,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        "frozen_inputs": {
            "test_sha256": _sha256(test_path),
            "basic_prompt_sha256": _sha256(basic_path),
            "strong_prompt_sha256": _sha256(strong_path),
            "manifest_sha256": _sha256(frozen_manifest_path),
            "adapter_model_sha256": _sha256(adapter_path / "adapter_model.safetensors"),
        },
        "metrics": metrics,
        "artifact_sha256": {},
        "volumes": {
            "model_cache": MODEL_CACHE_VOLUME_NAME,
            "training_output": TRAINING_OUTPUT_VOLUME_NAME,
            "evaluation_output": OUTPUT_VOLUME_NAME,
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "bitsandbytes",
                "peft",
                "torch",
                "transformers",
                "unsloth",
                "unsloth-zoo",
                "xformers",
            )
        },
    }
    generations_path = run_root / "generations.jsonl"
    metrics_path = run_root / "metrics.json"
    blinded_path = run_root / "blinded-generations.jsonl"
    key_path = run_root / "blinding-key.json"
    _write_jsonl(generations_path, rows)
    _write_json(metrics_path, metrics)
    _write_jsonl(blinded_path, blind)
    _write_json(key_path, key)
    manifest["artifact_sha256"] = {
        "generations.jsonl": _sha256(generations_path),
        "metrics.json": _sha256(metrics_path),
        "blinded-generations.jsonl": _sha256(blinded_path),
        "blinding-key.json": _sha256(key_path),
    }
    _write_json(run_root / "manifest.json", manifest)
    output_volume.commit()
    model_cache_volume.commit()
    return _public_summary(manifest)


def verify_persisted_evaluation(run_id: str) -> dict[str, object]:
    """Verify committed evaluation evidence from a separate read-only worker."""
    run_id = validate_identifier(run_id, field="run_id")
    run_root = OUTPUT_ROOT / "runs" / run_id
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation manifest missing for {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["artifact_sha256"]
    actual = {filename: _sha256(run_root / filename) for filename in expected}
    if actual != expected:
        raise RuntimeError("persisted evaluation hashes do not match manifest")
    generation_count = len(
        (run_root / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    )
    blind_count = len(
        (run_root / "blinded-generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    if generation_count != 300 or blind_count != 300:
        raise RuntimeError("persisted evaluation row counts are incomplete")
    return {
        "run_id": run_id,
        "status": "passed",
        "manifest_persisted": True,
        "artifact_hashes_verified": True,
        "generation_rows": generation_count,
        "blinded_rows": blind_count,
        "output_volume": OUTPUT_VOLUME_NAME,
        "output_path": f"/runs/{run_id}",
    }


def _generate_condition(
    model: Any,
    tokenizer: Any,
    records: list[DecisionRecord],
    system_prompt: str,
    condition: str,
    torch: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start : start + BATCH_SIZE]
            rendered = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": record.input},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch_records
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for record, token_ids in zip(batch_records, generated, strict=True):
                raw = tokenizer.decode(
                    token_ids[prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                scored = score_output(record.target.decision.value, raw)
                rows.append(
                    {
                        "id": record.id,
                        "condition": condition,
                        "input": record.input,
                        "gold_decision": record.target.decision.value,
                        "raw_output": raw,
                        **scored.as_dict(),
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    return rows


def _read_records(path: Path) -> list[DecisionRecord]:
    records = [
        DecisionRecord.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != TEST_RECORDS:
        raise RuntimeError(
            f"test split has {len(records)} records, expected {TEST_RECORDS}"
        )
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise RuntimeError(f"evaluation input hash mismatch for {path.name}")


def _public_summary(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "run_id",
            "status",
            "elapsed_seconds",
            "model",
            "model_revision",
            "adapter_run_id",
            "gpu",
            "condition_seconds",
            "decoding",
            "metrics",
            "artifact_sha256",
            "packages",
        )
    }

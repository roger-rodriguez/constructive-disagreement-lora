"""GPU-only runtime for the primary QLoRA training run."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from disagree_contracts.identifiers import validate_identifier
from disagree_contracts.prompt_rendering import parse_decision_output
from disagree_contracts.schemas import DecisionRecord

from disagree_modeling.compatibility_check import (
    LORA_TARGET_MODULES,
    MODEL_CACHE_VOLUME_NAME,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    require_exact_model_resolution,
)
from disagree_modeling.training import (
    MAX_NEW_TOKENS,
    OUTPUT_VOLUME_NAME,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
    TrainingConfiguration,
)

MODEL_CACHE_ROOT = Path("/vol/model-cache")
OUTPUT_ROOT = Path("/vol/outputs")


def run_primary_training(
    run_id: str,
    *,
    configuration: TrainingConfiguration,
    experiment_root: Path,
    model_cache_volume: Any,
    output_volume: Any,
) -> dict[str, object]:
    """Train the frozen adapter, evaluate validation behavior, and persist it."""
    run_id = validate_identifier(run_id, field="run_id")
    run_root = OUTPUT_ROOT / "runs" / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing training run {run_id}")

    importlib.import_module("unsloth")
    datasets = importlib.import_module("datasets")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    trl = importlib.import_module("trl")
    unsloth_module = importlib.import_module("unsloth")

    FastLanguageModel = unsloth_module.FastLanguageModel
    SFTConfig = trl.SFTConfig
    SFTTrainer = trl.SFTTrainer

    started_at = datetime.now(UTC)
    started = time.monotonic()
    seed = configuration.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest_path = experiment_root / "data" / "manifest.json"
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = experiment_root / "data" / "splits" / "train.jsonl"
    validation_path = experiment_root / "data" / "splits" / "validation.jsonl"
    prompt_path = experiment_root / "prompts" / "basic.txt"
    _require_hash(
        train_path,
        frozen_manifest["splits"]["train"]["examples"]["sha256"],
    )
    _require_hash(
        validation_path,
        frozen_manifest["splits"]["validation"]["examples"]["sha256"],
    )
    _require_hash(prompt_path, frozen_manifest["prompts"]["basic"]["sha256"])
    train_records = _read_records(train_path, expected=TRAIN_RECORDS)
    validation_records = _read_records(
        validation_path,
        expected=VALIDATION_RECORDS,
    )
    basic_prompt = prompt_path.read_text(encoding="utf-8").strip()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        use_exact_model_name=True,
        max_seq_length=configuration.max_sequence_length,
        dtype=None,
        load_in_4bit=True,
    )
    require_exact_model_resolution(
        getattr(model.config, "_name_or_path", None),
        getattr(model.config, "_commit_hash", None),
    )
    model_loaded_seconds = time.monotonic() - started

    prepared_train, train_token_summary = _prepare_records(
        train_records,
        tokenizer,
        basic_prompt,
        configuration.max_sequence_length,
    )
    prepared_validation, validation_token_summary = _prepare_records(
        validation_records,
        tokenizer,
        basic_prompt,
        configuration.max_sequence_length,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=configuration.lora_rank,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=configuration.lora_alpha,
        lora_dropout=configuration.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )
    model.config.use_cache = False

    train_dataset = datasets.Dataset.from_list(prepared_train)
    validation_dataset = datasets.Dataset.from_list(prepared_validation)
    checkpoints_path = run_root / "checkpoints"
    args = SFTConfig(
        output_dir=str(checkpoints_path),
        num_train_epochs=configuration.epochs,
        per_device_train_batch_size=configuration.per_device_train_batch_size,
        per_device_eval_batch_size=configuration.per_device_eval_batch_size,
        gradient_accumulation_steps=configuration.gradient_accumulation_steps,
        learning_rate=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
        warmup_ratio=configuration.warmup_ratio,
        lr_scheduler_type=configuration.lr_scheduler_type,
        logging_steps=configuration.logging_steps,
        eval_strategy=configuration.eval_strategy,
        save_strategy=configuration.save_strategy,
        save_total_limit=configuration.epochs,
        load_best_model_at_end=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=configuration.optimizer,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
    )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    _require_masked_training_batch(next(iter(trainer.get_train_dataloader())))

    training_started = time.monotonic()
    training_result = trainer.train()
    training_seconds = time.monotonic() - training_started

    adapter_path = run_root / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    model_cache_volume.commit()

    FastLanguageModel.for_inference(model)
    validation_started = time.monotonic()
    generations, validation_metrics = _evaluate_validation(
        model,
        tokenizer,
        validation_records,
        basic_prompt,
        torch,
    )
    validation_seconds = time.monotonic() - validation_started

    artifact_hashes = _artifact_hashes(adapter_path)
    finished_at = datetime.now(UTC)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "passed",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model_load_seconds": round(model_loaded_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "validation_seconds": round(validation_seconds, 3),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "gpu": torch.cuda.get_device_name(0),
        "load_in_4bit": True,
        "configuration": configuration.as_dict(),
        "frozen_inputs": {
            "train_sha256": _sha256(train_path),
            "validation_sha256": _sha256(validation_path),
            "basic_prompt_sha256": _sha256(prompt_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "token_counts": {
            "train": train_token_summary,
            "validation": validation_token_summary,
        },
        "training_metrics": training_result.metrics,
        "trainer_log_history": trainer.state.log_history,
        "validation_metrics": validation_metrics,
        "artifact_sha256": artifact_hashes,
        "volumes": {
            "model_cache": MODEL_CACHE_VOLUME_NAME,
            "output": OUTPUT_VOLUME_NAME,
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "accelerate",
                "bitsandbytes",
                "datasets",
                "peft",
                "torch",
                "transformers",
                "trl",
                "unsloth",
                "unsloth-zoo",
                "xformers",
            )
        },
    }
    _write_jsonl(run_root / "validation-generations.jsonl", generations)
    _write_json(run_root / "validation-metrics.json", validation_metrics)
    _write_json(run_root / "manifest.json", manifest)
    output_volume.commit()

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return _public_summary(manifest)


def verify_persisted_training(run_id: str) -> dict[str, object]:
    """Verify the committed full-training adapter and evidence from another worker."""
    run_id = validate_identifier(run_id, field="run_id")
    run_root = OUTPUT_ROOT / "runs" / run_id
    manifest_path = run_root / "manifest.json"
    metrics_path = run_root / "validation-metrics.json"
    generations_path = run_root / "validation-generations.jsonl"
    if not all(
        path.is_file() for path in (manifest_path, metrics_path, generations_path)
    ):
        raise FileNotFoundError(
            f"persisted training evidence is incomplete for {run_id}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hashes = _artifact_hashes(run_root / "adapter")
    if actual_hashes != manifest["artifact_sha256"]:
        raise RuntimeError("persisted adapter hashes do not match the run manifest")
    if (
        len(generations_path.read_text(encoding="utf-8").splitlines())
        != VALIDATION_RECORDS
    ):
        raise RuntimeError("persisted validation generation count is incomplete")
    return {
        "run_id": run_id,
        "status": "passed",
        "manifest_persisted": True,
        "validation_evidence_persisted": True,
        "adapter_hashes_verified": True,
        "output_volume": OUTPUT_VOLUME_NAME,
        "output_path": f"/runs/{run_id}",
    }


def _read_records(path: Path, *, expected: int) -> list[DecisionRecord]:
    records = [
        DecisionRecord.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != expected:
        raise RuntimeError(
            f"{path.name} has {len(records)} records, expected {expected}"
        )
    return records


def _messages(record: DecisionRecord, basic_prompt: str) -> list[dict[str, str]]:
    target = json.dumps(
        record.target.as_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "system", "content": basic_prompt},
        {"role": "user", "content": record.input},
        {"role": "assistant", "content": target},
    ]


def _prepare_records(
    records: list[DecisionRecord],
    tokenizer: Any,
    basic_prompt: str,
    max_length: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prepared: list[dict[str, object]] = []
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    for record in records:
        messages = _messages(record, basic_prompt)
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"{record.id}: prompt tokens are not a full-token prefix"
            )
        if len(full_ids) > max_length:
            raise RuntimeError(
                f"{record.id}: {len(full_ids)} tokens exceeds maximum {max_length}"
            )
        assistant_length = len(full_ids) - len(prompt_ids)
        if assistant_length <= 0:
            raise RuntimeError(f"{record.id}: assistant token suffix is empty")
        labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
        prepared.append(
            {
                "input_ids": list(full_ids),
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
        lengths.append(len(full_ids))
        assistant_lengths.append(assistant_length)
    return prepared, {
        "records": len(records),
        "total_tokens": sum(lengths),
        "maximum_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 3),
        "assistant_tokens": sum(assistant_lengths),
    }


def _require_masked_training_batch(batch: dict[str, Any]) -> None:
    labels = batch["labels"]
    if not bool((labels == -100).any()) or not bool((labels != -100).any()):
        raise RuntimeError("trainer batch does not preserve assistant-only labels")


def _evaluate_validation(
    model: Any,
    tokenizer: Any,
    records: list[DecisionRecord],
    basic_prompt: str,
    torch: Any,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    generations: list[dict[str, object]] = []
    decisions_correct = 0
    challenge_total = 0
    challenge_correct = 0
    comply_total = 0
    comply_correct = 0
    valid_json = 0
    openings: Counter[str] = Counter()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(records), 8):
            batch_records = records[start : start + 8]
            prompts = [
                tokenizer.apply_chat_template(
                    _messages(record, basic_prompt)[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for record in batch_records
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for record, token_ids in zip(batch_records, output, strict=True):
                raw = tokenizer.decode(
                    token_ids[prompt_length:],
                    skip_special_tokens=True,
                ).strip()
                parsed_decision: str | None = None
                parse_error: str | None = None
                try:
                    parsed = parse_decision_output(raw)
                    parsed_decision = parsed.decision.value
                    valid_json += 1
                    opening = " ".join(parsed.message.casefold().split()[:3])
                    if opening:
                        openings[opening] += 1
                except (TypeError, ValueError) as error:
                    parse_error = str(error)
                gold = record.target.decision.value
                correct = parsed_decision == gold
                decisions_correct += int(correct)
                if gold == "challenge":
                    challenge_total += 1
                    challenge_correct += int(correct)
                else:
                    comply_total += 1
                    comply_correct += int(correct)
                generations.append(
                    {
                        "id": record.id,
                        "gold_decision": gold,
                        "raw_output": raw,
                        "parsed_decision": parsed_decision,
                        "decision_correct": correct,
                        "parse_error": parse_error,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
    total = len(records)
    metrics = {
        "records": total,
        "raw_json_valid": valid_json,
        "raw_json_valid_rate": round(valid_json / total, 4),
        "decision_correct": decisions_correct,
        "decision_accuracy": round(decisions_correct / total, 4),
        "challenge_recall": round(challenge_correct / challenge_total, 4),
        "comply_accuracy": round(comply_correct / comply_total, 4),
        "false_objection_rate": round(1 - comply_correct / comply_total, 4),
        "top_message_openings": [
            {"opening": opening, "count": count}
            for opening, count in openings.most_common(10)
        ],
    }
    return generations, metrics


def _artifact_hashes(adapter_path: Path) -> dict[str, str]:
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
    )
    hashes: dict[str, str] = {}
    for filename in required:
        path = adapter_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"required adapter artifact is missing: {filename}")
        hashes[filename] = _sha256(path)
    return hashes


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
        raise RuntimeError(f"frozen input hash mismatch for {path.name}")


def _public_summary(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "run_id",
            "status",
            "elapsed_seconds",
            "model",
            "model_revision",
            "gpu",
            "configuration",
            "token_counts",
            "training_metrics",
            "validation_metrics",
            "artifact_sha256",
            "packages",
        )
    }

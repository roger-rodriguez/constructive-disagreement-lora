"""GPU-only implementation loaded inside the pinned Modal training image."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from disagree_contracts.identifiers import validate_identifier
from disagree_contracts.prompt_rendering import messages_for_template

from disagree_modeling.compatibility_check import (
    DATASET_CACHE_VOLUME_NAME,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_NEW_TOKENS,
    MAX_SEQUENCE_LENGTH,
    MAX_TRAINING_STEPS,
    MODEL_CACHE_VOLUME_NAME,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    OUTPUT_VOLUME_NAME,
    SEED,
    compatibility_record,
    labels_select_only_assistant,
    prepare_assistant_suffix,
    require_exact_model_resolution,
)

MODEL_CACHE_ROOT = Path("/vol/model-cache")
DATASET_CACHE_ROOT = Path("/vol/dataset-cache")
OUTPUT_ROOT = Path("/vol/outputs")


def run_compatibility_check(
    run_id: str,
    *,
    model_cache_volume: Any,
    dataset_cache_volume: Any,
    output_volume: Any,
) -> dict[str, object]:
    """Run one QLoRA step, reload the adapter, infer, and persist evidence."""
    run_id = validate_identifier(run_id, field="run_id")
    run_root = OUTPUT_ROOT / run_id
    if run_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing compatibility run {run_id}"
        )

    # Unsloth must patch the ML stack before its downstream packages are imported.
    importlib.import_module("unsloth")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    trl = importlib.import_module("trl")
    unsloth_module = importlib.import_module("unsloth")

    FastLanguageModel = unsloth_module.FastLanguageModel
    SFTConfig = trl.SFTConfig
    SFTTrainer = trl.SFTTrainer

    started_at = datetime.now(UTC)
    started = time.monotonic()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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
    model_loaded_seconds = time.monotonic() - started

    record = compatibility_record()
    messages = messages_for_template(record)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
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
    prepared = prepare_assistant_suffix(prompt_ids, full_ids)
    if not labels_select_only_assistant(prepared):
        raise RuntimeError("assistant-only label invariant failed before training")

    dataset_path = DATASET_CACHE_ROOT / run_id / "compatibility-example.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=False)
    dataset_path.write_text(
        json.dumps(
            {
                "id": record.id,
                "messages": messages,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_cache_volume.commit()

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    model.config.use_cache = False

    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": [list(prepared.input_ids)],
            "attention_mask": [list(prepared.attention_mask)],
            "labels": [list(prepared.labels)],
        }
    )
    checkpoints_path = run_root / "checkpoints"
    args = SFTConfig(
        output_dir=str(checkpoints_path),
        max_steps=MAX_TRAINING_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=None,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=transformers.default_data_collator,
    )

    batch_labels = next(iter(trainer.get_train_dataloader()))["labels"][0].tolist()
    if tuple(batch_labels) != prepared.labels:
        raise RuntimeError("trainer changed the verified assistant-only labels")

    training_started = time.monotonic()
    training_result = trainer.train()
    training_seconds = time.monotonic() - training_started

    adapter_path = run_root / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    output_volume.commit()
    model_cache_volume.commit()

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    base_model, reloaded_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        use_exact_model_name=True,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    require_exact_model_resolution(
        getattr(base_model.config, "_name_or_path", None),
        getattr(base_model.config, "_commit_hash", None),
    )
    reloaded_model = peft.PeftModel.from_pretrained(base_model, str(adapter_path))
    FastLanguageModel.for_inference(reloaded_model)
    inference_input = reloaded_tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(reloaded_model.device)
    # This is one unpadded sequence, so every input position is valid. Passing
    # the mask explicitly is required because Qwen uses the same pad and EOS ID.
    inference_attention_mask = torch.ones_like(inference_input)
    generated = reloaded_model.generate(
        input_ids=inference_input,
        attention_mask=inference_attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    generation = reloaded_tokenizer.decode(
        generated[0][inference_input.shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    if not generation:
        raise RuntimeError("reloaded adapter produced an empty generation")

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
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "gpu": torch.cuda.get_device_name(0),
        "load_in_4bit": True,
        "training_steps": MAX_TRAINING_STEPS,
        "training_loss": training_result.metrics.get("train_loss"),
        "chat_template_rendered": rendered,
        "token_mask": {
            "total_tokens": len(prepared.input_ids),
            "prompt_tokens_masked": prepared.prompt_tokens,
            "assistant_tokens_trained": prepared.assistant_tokens,
        },
        "adapter_reloaded": True,
        "reload_generation": generation,
        "artifact_sha256": artifact_hashes,
        "volumes": {
            "model_cache": MODEL_CACHE_VOLUME_NAME,
            "dataset_cache": DATASET_CACHE_VOLUME_NAME,
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
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_volume.commit()
    return _public_summary(manifest)


def verify_compatibility_artifacts(run_id: str) -> dict[str, object]:
    """Verify committed artifacts from a separate, CPU-only Modal invocation."""
    run_id = validate_identifier(run_id, field="run_id")
    manifest_path = OUTPUT_ROOT / run_id / "manifest.json"
    dataset_path = DATASET_CACHE_ROOT / run_id / "compatibility-example.json"
    if not manifest_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(
            f"persisted compatibility evidence is incomplete for {run_id}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = manifest["artifact_sha256"]
    adapter_path = OUTPUT_ROOT / run_id / "adapter"
    actual_hashes = _artifact_hashes(adapter_path)
    if actual_hashes != expected_hashes:
        raise RuntimeError("persisted adapter hashes do not match the run manifest")
    return {
        "run_id": run_id,
        "status": "passed",
        "manifest_persisted": True,
        "dataset_persisted": True,
        "adapter_hashes_verified": True,
    }


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
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


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
            "training_steps",
            "training_loss",
            "token_mask",
            "adapter_reloaded",
            "reload_generation",
            "artifact_sha256",
            "packages",
        )
    }

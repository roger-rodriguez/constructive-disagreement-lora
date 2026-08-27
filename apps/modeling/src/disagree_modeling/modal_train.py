"""Modal entrypoint for the opt-in paid primary QLoRA run."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from disagree_modeling.compatibility_check import (
    CPU_CORES,
    GPU_TYPE,
    MEMORY_MIB,
    MODEL_CACHE_VOLUME_NAME,
)
from disagree_modeling.training import (
    APP_NAME,
    CONFIG,
    DEFAULT_RUN_ID,
    MAX_CONTAINERS,
    OUTPUT_VOLUME_NAME,
    TIMEOUT_SECONDS,
    build_training_plan,
)

MODELING_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = MODELING_ROOT.parent.parent
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"
REQUIREMENTS_PATH = MODELING_ROOT / "configs" / "model-requirements.txt"
MODEL_CACHE_PATH = "/vol/model-cache"
OUTPUT_PATH = "/vol/outputs"
REMOTE_EXPERIMENT_ROOT = Path("/opt/experiment")

MODEL_CACHE_VOLUME = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
OUTPUT_VOLUME = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)

SOURCE_MODULES = ("disagree_contracts", "disagree_modeling")
TRAIN_ENV = {
    "HF_HOME": MODEL_CACHE_PATH,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "UNSLOTH_STABLE_DOWNLOADS": "1",
}
TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REQUIREMENTS_PATH))
    .env(TRAIN_ENV)
    .add_local_python_source(*SOURCE_MODULES, copy=True)
    .add_local_file(
        EXPERIMENT_ROOT / "data" / "splits" / "train.jsonl",
        str(REMOTE_EXPERIMENT_ROOT / "data" / "splits" / "train.jsonl"),
        copy=True,
    )
    .add_local_file(
        EXPERIMENT_ROOT / "data" / "splits" / "validation.jsonl",
        str(REMOTE_EXPERIMENT_ROOT / "data" / "splits" / "validation.jsonl"),
        copy=True,
    )
    .add_local_file(
        EXPERIMENT_ROOT / "data" / "manifest.json",
        str(REMOTE_EXPERIMENT_ROOT / "data" / "manifest.json"),
        copy=True,
    )
    .add_local_file(
        EXPERIMENT_ROOT / "prompts" / "basic.txt",
        str(REMOTE_EXPERIMENT_ROOT / "prompts" / "basic.txt"),
        copy=True,
    )
)
VERIFY_IMAGE = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    *SOURCE_MODULES,
    copy=True,
)

app = modal.App(APP_NAME)


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    retries=0,
    timeout=TIMEOUT_SECONDS,
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE_VOLUME,
        OUTPUT_PATH: OUTPUT_VOLUME,
    },
)
def train_primary(run_id: str) -> dict[str, object]:
    from disagree_modeling.train_runtime import run_primary_training

    return run_primary_training(
        run_id,
        configuration=CONFIG,
        experiment_root=REMOTE_EXPERIMENT_ROOT,
        model_cache_volume=MODEL_CACHE_VOLUME,
        output_volume=OUTPUT_VOLUME,
    )


@app.function(
    image=VERIFY_IMAGE,
    cpu=0.125,
    memory=256,
    min_containers=0,
    max_containers=1,
    retries=0,
    timeout=60,
    volumes={OUTPUT_PATH: OUTPUT_VOLUME.with_mount_options(read_only=True)},
)
def verify_persisted_training(run_id: str) -> dict[str, object]:
    from disagree_modeling.train_runtime import verify_persisted_training

    return verify_persisted_training(run_id)


@app.local_entrypoint()
def main(run_id: str = DEFAULT_RUN_ID, execute: bool = False) -> None:
    plan = build_training_plan(EXPERIMENT_ROOT, run_id)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not execute:
        print("Dry plan only: no Modal function was invoked.")
        return

    result = train_primary.remote(run_id)
    persistence = verify_persisted_training.remote(run_id)
    print(json.dumps({"result": result, "persistence": persistence}, indent=2))

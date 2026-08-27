"""Modal entrypoint for the opt-in paid model compatibility check."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from disagree_modeling.compatibility_check import (
    APP_NAME,
    CPU_CORES,
    DATASET_CACHE_VOLUME_NAME,
    DEFAULT_RUN_ID,
    GPU_TYPE,
    MAX_CONTAINERS,
    MEMORY_MIB,
    MODEL_CACHE_VOLUME_NAME,
    OUTPUT_VOLUME_NAME,
    TIMEOUT_SECONDS,
    build_compatibility_plan,
)

MODELING_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_PATH = MODELING_ROOT / "configs" / "model-requirements.txt"
MODEL_CACHE_PATH = "/vol/model-cache"
DATASET_CACHE_PATH = "/vol/dataset-cache"
OUTPUT_PATH = "/vol/outputs"

MODEL_CACHE_VOLUME = modal.Volume.from_name(
    MODEL_CACHE_VOLUME_NAME,
    create_if_missing=True,
)
DATASET_CACHE_VOLUME = modal.Volume.from_name(
    DATASET_CACHE_VOLUME_NAME,
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
        DATASET_CACHE_PATH: DATASET_CACHE_VOLUME,
        OUTPUT_PATH: OUTPUT_VOLUME,
    },
)
def run_model_check(run_id: str) -> dict[str, object]:
    from disagree_modeling.compatibility_runtime import run_compatibility_check

    return run_compatibility_check(
        run_id,
        model_cache_volume=MODEL_CACHE_VOLUME,
        dataset_cache_volume=DATASET_CACHE_VOLUME,
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
    volumes={
        DATASET_CACHE_PATH: DATASET_CACHE_VOLUME.with_mount_options(read_only=True),
        OUTPUT_PATH: OUTPUT_VOLUME.with_mount_options(read_only=True),
    },
)
def verify_compatibility_artifacts(run_id: str) -> dict[str, object]:
    from disagree_modeling.compatibility_runtime import verify_compatibility_artifacts

    return verify_compatibility_artifacts(run_id)


@app.local_entrypoint()
def main(run_id: str = DEFAULT_RUN_ID, execute: bool = False) -> None:
    print(json.dumps(build_compatibility_plan(), indent=2, sort_keys=True))
    if not execute:
        print("Dry plan only: no Modal function was invoked.")
        return

    result = run_model_check.remote(run_id)
    persistence = verify_compatibility_artifacts.remote(run_id)
    print(json.dumps({"result": result, "persistence": persistence}, indent=2))

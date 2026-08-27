"""Modal entrypoint for the opt-in paid held-out A/B/C evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from disagree_modeling.evaluation import (
    APP_NAME,
    CPU_CORES,
    DEFAULT_RUN_ID,
    GPU_TYPE,
    MAX_CONTAINERS,
    MEMORY_MIB,
    MODEL_CACHE_VOLUME_NAME,
    OUTPUT_VOLUME_NAME,
    TIMEOUT_SECONDS,
    TRAINING_OUTPUT_VOLUME_NAME,
    build_evaluation_plan,
)

MODELING_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = MODELING_ROOT.parent.parent
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"
REQUIREMENTS_PATH = MODELING_ROOT / "configs" / "model-requirements.txt"
MODEL_CACHE_PATH = "/vol/model-cache"
TRAINING_OUTPUT_PATH = "/vol/training-output"
OUTPUT_PATH = "/vol/evaluation-output"
REMOTE_EXPERIMENT_ROOT = Path("/opt/experiment")

MODEL_CACHE_VOLUME = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME)
TRAINING_OUTPUT_VOLUME = modal.Volume.from_name(TRAINING_OUTPUT_VOLUME_NAME)
OUTPUT_VOLUME = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

SOURCE_MODULES = ("disagree_contracts", "disagree_modeling")
EVALUATION_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(REQUIREMENTS_PATH))
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "UNSLOTH_STABLE_DOWNLOADS": "1",
        }
    )
    .add_local_python_source(*SOURCE_MODULES, copy=True)
    .add_local_file(
        EXPERIMENT_ROOT / "data" / "splits" / "test.jsonl",
        str(REMOTE_EXPERIMENT_ROOT / "data" / "splits" / "test.jsonl"),
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
    .add_local_file(
        EXPERIMENT_ROOT / "prompts" / "strong.txt",
        str(REMOTE_EXPERIMENT_ROOT / "prompts" / "strong.txt"),
        copy=True,
    )
)
VERIFY_IMAGE = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    *SOURCE_MODULES,
    copy=True,
)

app = modal.App(APP_NAME)


@app.function(
    image=EVALUATION_IMAGE,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    retries=0,
    timeout=TIMEOUT_SECONDS,
    volumes={
        MODEL_CACHE_PATH: MODEL_CACHE_VOLUME,
        TRAINING_OUTPUT_PATH: TRAINING_OUTPUT_VOLUME.with_mount_options(read_only=True),
        OUTPUT_PATH: OUTPUT_VOLUME,
    },
)
def evaluate_held_out(run_id: str) -> dict[str, object]:
    from disagree_modeling.evaluate_runtime import run_held_out_evaluation

    return run_held_out_evaluation(
        run_id,
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
def verify_persisted_evaluation(run_id: str) -> dict[str, object]:
    from disagree_modeling.evaluate_runtime import verify_persisted_evaluation

    return verify_persisted_evaluation(run_id)


@app.local_entrypoint()
def main(run_id: str = DEFAULT_RUN_ID, execute: bool = False) -> None:
    print(
        json.dumps(
            build_evaluation_plan(EXPERIMENT_ROOT, run_id), indent=2, sort_keys=True
        )
    )
    if not execute:
        print("Dry plan only: no Modal function was invoked.")
        return
    result = evaluate_held_out.remote(run_id)
    persistence = verify_persisted_evaluation.remote(run_id)
    print(json.dumps({"result": result, "persistence": persistence}, indent=2))

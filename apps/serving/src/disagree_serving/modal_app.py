"""Scale-to-zero Modal health and model-backed decision endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from disagree_serving import __version__
from disagree_serving.config import load_config

SERVING_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = SERVING_ROOT.parent.parent
LOCAL_CONFIG_PATH = SERVING_ROOT / "configs" / "model.toml"
LOCAL_REQUIREMENTS_PATH = SERVING_ROOT / "configs" / "runtime-requirements.txt"
LOCAL_PROMPT_PATH = REPOSITORY_ROOT / "experiment" / "prompts" / "basic.txt"
REMOTE_CONFIG_PATH = Path("/opt/serving/configs/model.toml")
REMOTE_REQUIREMENTS_PATH = Path("/opt/serving/configs/runtime-requirements.txt")
REMOTE_PROMPT_PATH = Path("/opt/experiment/prompts/basic.txt")
CONFIG_PATH = LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.is_file() else REMOTE_CONFIG_PATH
REQUIREMENTS_PATH = (
    LOCAL_REQUIREMENTS_PATH
    if LOCAL_REQUIREMENTS_PATH.is_file()
    else REMOTE_REQUIREMENTS_PATH
)
PROMPT_PATH = LOCAL_PROMPT_PATH if LOCAL_PROMPT_PATH.is_file() else REMOTE_PROMPT_PATH
MODEL_CACHE_PATH = "/vol/model-cache"
CONFIG = load_config(CONFIG_PATH)
APP_NAME = CONFIG.service.app_name
FASTAPI_VERSION = "0.141.1"
PYDANTIC_VERSION = "2.12.5"
MIN_CONTAINERS = CONFIG.runtime.min_containers

COMMON_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file(CONFIG_PATH, str(REMOTE_CONFIG_PATH), copy=True)
    .add_local_file(REQUIREMENTS_PATH, str(REMOTE_REQUIREMENTS_PATH), copy=True)
    .add_local_file(PROMPT_PATH, str(REMOTE_PROMPT_PATH), copy=True)
)
HEALTH_IMAGE = COMMON_IMAGE.pip_install(
    f"fastapi[standard-no-fastapi-cloud-cli]=={FASTAPI_VERSION}",
    f"pydantic=={PYDANTIC_VERSION}",
)
MODEL_IMAGE = (
    COMMON_IMAGE.pip_install_from_requirements(str(REQUIREMENTS_PATH))
    .env(
        {
            "HF_HOME": MODEL_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("disagree_contracts", "disagree_serving", copy=True)
)
MODEL_CACHE_VOLUME = modal.Volume.from_name(
    CONFIG.runtime.model_cache_volume,
    create_if_missing=True,
)
app = modal.App(APP_NAME)


class DecisionRequest(BaseModel):
    """Single bounded request accepted by the model endpoint."""

    model_config = ConfigDict(extra="forbid")

    input: str


def health_payload() -> dict[str, str]:
    """Return the stable service health response."""
    return {
        "service": "constructive-disagreement",
        "status": "ok",
        "version": __version__,
    }


@app.function(
    image=HEALTH_IMAGE,
    cpu=0.125,
    memory=128,
    min_containers=MIN_CONTAINERS,
    timeout=30,
)
@modal.fastapi_endpoint(method="GET", label=CONFIG.service.health_label)
def health() -> dict[str, str]:
    """Verify that Modal can route the app without starting a GPU."""
    return health_payload()


@app.cls(
    image=MODEL_IMAGE,
    gpu=CONFIG.runtime.gpu,
    cpu=CONFIG.runtime.cpu,
    memory=CONFIG.runtime.memory_mib,
    min_containers=CONFIG.runtime.min_containers,
    max_containers=CONFIG.runtime.max_containers,
    scaledown_window=CONFIG.runtime.scaledown_window_seconds,
    retries=0,
    timeout=CONFIG.runtime.timeout_seconds,
    startup_timeout=CONFIG.runtime.startup_timeout_seconds,
    volumes={MODEL_CACHE_PATH: MODEL_CACHE_VOLUME},
)
@modal.concurrent(max_inputs=CONFIG.runtime.max_concurrent_inputs)
class ConstructiveDisagreementModel:
    """One scale-to-zero L4 worker serving the pinned LoRA adapter."""

    model: Any
    tokenizer: Any
    torch: Any
    system_prompt: str

    @modal.enter()
    def load(self) -> None:
        from disagree_serving.model_runtime import load_model

        self.model, self.tokenizer, self.torch, self.system_prompt = load_model(
            CONFIG,
            REMOTE_PROMPT_PATH,
        )
        MODEL_CACHE_VOLUME.commit()

    @modal.fastapi_endpoint(
        method="POST",
        label=CONFIG.service.decision_label,
        requires_proxy_auth=True,
    )
    def decide(self, request: DecisionRequest) -> dict[str, str | None]:
        """Classify one request and return the validated five-field JSON object."""
        from disagree_serving.model_runtime import generate_response
        from disagree_serving.service import validate_user_input

        try:
            user_input = validate_user_input(
                request.input,
                CONFIG.generation.max_input_characters,
            )
            return generate_response(
                model=self.model,
                tokenizer=self.tokenizer,
                torch=self.torch,
                system_prompt=self.system_prompt,
                user_input=user_input,
                config=CONFIG,
            )
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

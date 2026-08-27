"""GPU-only loading and generation for the public LoRA adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from disagree_serving.config import ServingConfig
from disagree_serving.service import (
    inference_messages,
    read_verified_prompt,
    response_payload,
)


def load_model(
    config: ServingConfig,
    prompt_path: Path,
) -> tuple[Any, Any, Any, str]:
    """Load the pinned base and public adapter into one GPU worker."""
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.adapter_id,
        revision=config.model.adapter_revision,
    )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.base_id,
        revision=config.model.base_revision,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    resolved_revision = getattr(base_model.config, "_commit_hash", None)
    if resolved_revision != config.model.base_revision:
        raise RuntimeError("loaded base model revision does not match configuration")
    model = peft.PeftModel.from_pretrained(
        base_model,
        config.model.adapter_id,
        revision=config.model.adapter_revision,
    )
    model.eval()
    system_prompt = read_verified_prompt(prompt_path, config.prompt.sha256)
    return model, tokenizer, torch, system_prompt


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    system_prompt: str,
    user_input: str,
    config: ServingConfig,
) -> dict[str, str | None]:
    """Generate and strictly parse one five-field decision response."""
    rendered = tokenizer.apply_chat_template(
        inference_messages(system_prompt, user_input),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    if input_ids.shape[-1] > config.generation.max_input_tokens:
        raise ValueError(
            f"rendered input exceeds {config.generation.max_input_tokens} tokens"
        )
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=config.generation.do_sample,
            use_cache=True,
        )
    raw_output = tokenizer.decode(
        generated[0][input_ids.shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    try:
        return response_payload(raw_output)
    except (TypeError, ValueError) as error:
        raise RuntimeError("model returned invalid structured JSON") from error

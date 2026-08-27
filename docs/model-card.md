---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - base_model:adapter:Qwen/Qwen3-4B-Instruct-2507
  - lora
  - sft
  - qwen3
  - structured-output
  - transformers
  - trl
  - unsloth
---

# Qwen3 4B Constructive Disagreement LoRA

This LoRA adapter comes from the reproducible experiment in
[`roger-rodriguez/fine-tuning`](https://github.com/roger-rodriguez/fine-tuning).
It adapts `Qwen/Qwen3-4B-Instruct-2507` to classify fictional workplace requests
as `challenge` or `comply` and return a five-field JSON response.

This repository contains an **adapter, not the 4B base model**. Loading it still
requires the base model and its associated compute. It is a narrow learning
artifact, not a general-purpose safety system or evidence that the model has
judgment, understanding, or agency.

## Result

The adapter was evaluated once on 100 untouched, fictional scenarios:

| Condition | Valid JSON | Flawed recall | Reasonable specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Base model + basic prompt | 87% | 94% | 68% | 81% |
| Base model + strong prompt | 100% | 96% | 96% | 96% |
| This adapter + basic prompt | 100% | 100% | 94% | 97% |

The adapter beat the basic-prompt baseline by 16 percentage points, but beat
the strong-prompt baseline by only one point (paired bootstrap 95% interval:
-3 to +5 points). It did **not** meet the predeclared threshold for a material
win over strong prompting. It caught all 50 flawed requests and falsely
challenged three of 50 reasonable requests.

See the
[full evaluation](https://github.com/roger-rodriguez/fine-tuning/blob/main/docs/evaluation.md)
and
[raw evidence](https://github.com/roger-rodriguez/fine-tuning/tree/main/artifacts/runs/held-out-evaluation-001).

## Intended use

Use this adapter to reproduce or study the experiment's narrow structured
behavior. It expects a system prompt that asks for exactly one JSON object with
these fields, in order:

```json
{
  "decision": "challenge or comply",
  "issue": "string or null",
  "message": "string or null",
  "question": "string or null",
  "suggested_next_step": "string or null"
}
```

For the exact frozen prompt and schema, use the files in the source repository
rather than reconstructing them from this summary.

### Out-of-scope use

Do not treat the adapter as an authority for employment, legal, medical,
financial, privacy, security, or other consequential decisions. Do not use it
to score people, automate discipline, replace expert review, or infer that a
request is safe merely because the output says `comply`.

## Loading the adapter

The reported environment used Transformers 5.5.0, PEFT 0.20.0, and PyTorch
2.9.1. The base-model revision is pinned for reproduction.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ADAPTER_ID = "{{HF_REPO_ID}}"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    revision=BASE_REVISION,
    device_map="auto",
    torch_dtype="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
```

Use the frozen basic prompt from
[`experiment/prompts/basic.txt`](https://github.com/roger-rodriguez/fine-tuning/blob/main/experiment/prompts/basic.txt)
as the system message, then render the conversation with the included tokenizer
chat template. The reported evaluation used greedy decoding with at most 256
new tokens; raw JSON was never repaired or retried.

## Training

- Method: 4-bit QLoRA supervised fine-tuning
- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Base revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- Data: 400 synthetic training examples and 50 validation examples
- LoRA: rank 16, alpha 16, dropout 0
- Targets: attention projections plus gate/up/down projections
- Schedule: three predetermined epochs, 150 optimizer steps, seed 3407
- Hardware: one NVIDIA L40S on Modal
- End-to-end training function: 165.5 seconds
- Successful-run resource usage: $0.13311718 before credits

All scenarios were newly invented and agent-reviewed. They do not contain real
employee conversations, customer records, tickets, production identifiers, or
private company data. The dataset is licensed separately under CC BY 4.0.

## Limitations

- The evaluation has only 100 synthetic examples and no human labels.
- The adapter occasionally challenges feasible requests. In one retained
  failure it reversed a Monday-to-Friday timeline.
- Binary labels flatten reasonable ambiguity; one frozen feasibility example
  has a defensible alternative interpretation.
- The task, prompt, schema, and evaluation were all English-only.
- This measures behavior on one constructed task, not general reasoning.
- The strong prompt nearly matched the adapter, so prompting may be the simpler
  choice when token cost, latency, or prompt ownership are not constraints.

## Files and integrity

The release intentionally excludes `training_args.bin` because that pickle
serialization is not required for inference. `release-manifest.json` records
the byte size and SHA-256 digest of every published source file. The primary
adapter digest is:

```text
8b211ae87e6f7b302f3c2adb80a22bbf2bae129d8d683285f77470e44cbdc63f
```

The source run is `primary-qlora-001`. Its training manifest, validation
outputs, costs, and held-out evaluation are preserved in the source repository.

## License and attribution

This adapter is shared under Apache License 2.0. The upstream Qwen base model is
also published under Apache License 2.0; it is not redistributed here. The
synthetic dataset, prompts, documentation, and experiment evidence in the
source repository are licensed under CC BY 4.0.

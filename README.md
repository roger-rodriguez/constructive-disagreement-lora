# Fine-tuning a model to disagree constructively

[![CI](https://github.com/roger-rodriguez/fine-tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/roger-rodriguez/fine-tuning/actions/workflows/ci.yml)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Data license](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![LoRA adapter](https://img.shields.io/badge/Hugging%20Face-LoRA-ffd21e.svg)](https://huggingface.co/rrod/qwen3-4b-constructive-disagreement-lora)

This repository is a complete, reproducible QLoRA experiment. It fine-tunes
`Qwen/Qwen3-4B-Instruct-2507` to classify a workplace request as `challenge` or
`comply`, then return one structured JSON response that explains the issue and
suggests a practical next step.

The interesting result is not that fine-tuning simply won. The adapter greatly
improved a weak prompt, while a carefully written prompt almost matched it.

| Condition | Valid JSON | Flawed recall | Reasonable specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Base model + basic prompt | 87% | 94% | 68% | 81% |
| Base model + strong prompt | 100% | 96% | 96% | 96% |
| LoRA adapter + basic prompt | 100% | 100% | 94% | 97% |

The adapter beat the basic-prompt baseline by 16 percentage points, but beat
the strong-prompt baseline by only one point. That did not meet the
predeclared eight-point threshold for a material win over prompting. A useful
interpretation is that fine-tuning compressed much of the strong prompt's
behavior into the adapter rather than creating a new form of judgment.

## What is included

- 400 synthetic training examples, 50 validation examples, and 100 held-out
  test examples.
- Frozen basic and strong prompts plus a strict five-field response schema.
- Modal jobs for a one-step compatibility check, primary QLoRA training, and
  the three-condition held-out evaluation.
- An independently locked, scale-to-zero Modal app that loads the published
  adapter and returns the same five-field JSON used by the evaluation.
- CPU-only validation, metrics, tests, and a 90% repository coverage gate.
- Raw generations, manifests, hashes, timings, and resource costs for the
  reported runs.
- A deterministic publisher for the separately hosted LoRA adapter.

The repository does not include the 4B base model, adapter binaries, caches,
credentials, or private data. The Apache-2.0 adapter is available at
[`rrod/qwen3-4b-constructive-disagreement-lora`](https://huggingface.co/rrod/qwen3-4b-constructive-disagreement-lora).

## Reproduce the experiment

The project supports Python 3.11 and 3.12 and pins uv `0.12.6`, Ruff `0.16.5`,
ty `0.0.75`, and Modal `1.5.4`.

```bash
uv sync --frozen
uv sync --directory packages/contracts --frozen
uv sync --directory apps/modeling --group publish --frozen
uv sync --directory apps/serving --frozen

uv run --directory apps/modeling --frozen disagree-freeze-data \
  --experiment-root ../../experiment --check
uv run --frozen python scripts/build_data_quality_report.py --check

uv run --directory apps/modeling --frozen disagree-training-plan
uv run --directory apps/modeling --frozen disagree-evaluation-plan
```

The two plan commands are local and free. They verify the frozen inputs and
print the exact paid Modal commands, GPU types, timeouts, output Volumes, and
cost ceilings without starting cloud compute.

See:

- [Dataset design and format](docs/dataset.md)
- [Training and adapter publishing](docs/training.md)
- [Evaluation protocol and results](docs/evaluation.md)
- [Hugging Face model card](docs/model-card.md)
- [Adapter publishing procedure](docs/publishing.md)
- [Modeling command reference](apps/modeling/README.md)
- [Serving deployment and request example](apps/serving/README.md)

## Repository layout

```text
apps/modeling/     Modal data validation, training, evaluation, and publishing
apps/serving/      Independently deployable Modal model endpoint
packages/contracts/ Shared schemas, identifiers, and response parsing
experiment/        Frozen data, prompts, schema, taxonomy, and hashes
artifacts/runs/     Immutable training and evaluation evidence
docs/               Reproduction guides, model card, and publishing instructions
scripts/            Repository-wide validation and coverage commands
```

Modeling and serving are intentionally separate concerns. The serving app
keeps a public CPU-only health route and an authenticated GPU POST route. The
GPU route loads the immutable public Hugging Face adapter revision, limits the
deployment to one L4 container, and scales to zero when idle.

## Reported cost

The successful three-epoch L40S training function took 165.5 seconds and used
`$0.13311718` of Modal resources before credits. The L4 held-out evaluation and
its persistence check used `$0.22`. Provider credits reduced the observed bill
to zero, but the underlying resource usage is reported instead.

## Scope

This is a small, English-only, synthetic behavioral experiment. It is not a
safety system, a general benchmark, or evidence that the model gained genuine
judgment. Do not use it for consequential employment, legal, medical,
financial, privacy, or security decisions.

Source code is Apache-2.0. Original data, prompts, documentation, and recorded
evidence are CC BY 4.0. Third-party model weights, tokenizers, and libraries
retain their upstream licenses and are not relicensed by this repository.

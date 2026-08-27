# Modeling

This package validates the dataset, runs QLoRA training and held-out evaluation
on Modal, and publishes the resulting adapter to Hugging Face. Its locked ML
environment is isolated from the lightweight repository tooling.

## Validate frozen inputs

```bash
uv run --directory apps/modeling --frozen disagree-freeze-data \
  --experiment-root ../../experiment --check
uv run --frozen python scripts/build_data_quality_report.py --check
```

The freeze check enforces schemas, stable IDs, split sizes, class and category
distributions, minimal-pair structure, cross-split overlap, provenance states,
and SHA-256 hashes.

## Inspect cloud work locally

These commands verify inputs and print resource plans without contacting Modal:

```bash
uv run --directory apps/modeling --frozen disagree-compatibility-plan
uv run --directory apps/modeling --frozen disagree-training-plan
uv run --directory apps/modeling --frozen disagree-evaluation-plan
```

Each paid Modal entrypoint requires an explicit `--execute` flag. Training uses
only train and validation data; evaluation uses only held-out test data and a
read-only adapter Volume.

## Run on Modal

```bash
uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_compatibility \
  --run-id compatibility-check-002 --execute

uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_train \
  --run-id primary-qlora-001 --execute

uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_evaluate \
  --run-id held-out-evaluation-001 --execute
```

Use new run IDs for new experiments. Do not overwrite evidence after inspecting
a result. See [training](../../docs/training.md) and
[evaluation](../../docs/evaluation.md) for configuration and reported outcomes.

## Publish an adapter

```bash
uv run --directory apps/modeling --group publish --frozen \
  disagree-publish-adapter verify \
  --adapter-dir /path/to/adapter \
  --output-dir ../../.release/huggingface-adapter
```

The release command allowlists inference files, checks manifest hashes,
excludes `training_args.bin`, and can upload and byte-verify an immutable Hub
revision. See [publishing instructions](../../docs/publishing.md).

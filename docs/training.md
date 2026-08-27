# Training and publishing

## Environment

The reported adapter uses 4-bit QLoRA through Unsloth, TRL, Transformers, and
PEFT. The base model is `Qwen/Qwen3-4B-Instruct-2507` at immutable revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

Install the locked CPU tooling and modeling environment:

```bash
uv sync --frozen
uv sync --directory packages/contracts --frozen
uv sync --directory apps/modeling --group publish --frozen
```

Authenticate the Modal CLI before running a GPU command:

```bash
uv run --directory apps/modeling --frozen modal token new
```

The locked GPU image is exported in
`apps/modeling/configs/model-requirements.txt`. Normal local setup and CI do not
install that CUDA environment.

## Compatibility check

Before a full run, inspect the one-step compatibility plan:

```bash
uv run --directory apps/modeling --frozen disagree-compatibility-plan
```

Run it only when you intend to start paid L40S compute:

```bash
uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_compatibility \
  --run-id compatibility-check-002 --execute
```

The compatibility run checks the exact base-model resolution, assistant-only
loss masking, one optimization step, adapter save and reload, deterministic
generation, and Volume persistence.

## Primary QLoRA run

Inspect and verify the frozen run without contacting Modal:

```bash
uv run --directory apps/modeling --frozen disagree-training-plan
```

The reported configuration uses rank 16, alpha 16, three epochs, an effective
batch size of eight, 150 optimizer steps, and seed 3407 on one L40S. The Modal
image receives only the training and validation splits.

Start the paid run:

```bash
uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_train \
  --run-id primary-qlora-001 --execute
```

Artifacts are written to the `fine-tuning-training-output` Modal Volume under
`/runs/primary-qlora-001/`. A separate read-only function verifies the saved
files. The reported successful function took 165.5 seconds and used
`$0.13311718` of resources before credits.

To reproduce the experiment under a new run ID, change the ID rather than
overwriting an existing run. If the dataset, prompt, model, or hyperparameters
change, record a new manifest and treat it as a new experiment.

## Publish the adapter

Download the adapter directory from the Modal Volume, then stage an allowlisted
release:

```bash
uv run --directory apps/modeling --frozen modal volume get \
  fine-tuning-training-output \
  /runs/primary-qlora-001/adapter \
  /path/to/local/adapter

uv run --directory apps/modeling --group publish --frozen \
  disagree-publish-adapter verify \
  --adapter-dir /path/to/local/adapter \
  --output-dir ../../.release/huggingface-adapter
```

The publisher checks the training-manifest hashes, uploads only inference files,
excludes the unnecessary `training_args.bin` pickle, downloads the immutable
Hub commit, and byte-verifies it. The complete upload command is documented in
[publishing instructions](publishing.md).

The reported adapter is public at
[`rrod/qwen3-4b-constructive-disagreement-lora`](https://huggingface.co/rrod/qwen3-4b-constructive-disagreement-lora),
revision `574ade5ec9ca7f2ac834987f733145dd716432eb`. That revision updates
documentation only; the serving configuration remains pinned to the original
weight revision `eae43ddb16c580c4179ed8088c8180b28fb4d572`, whose adapter bytes
are identical.

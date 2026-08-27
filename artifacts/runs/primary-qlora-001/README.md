# primary-qlora-001

This immutable evidence bundle records the first full QLoRA training run. It
does not contain the adapter weights, checkpoints, secrets, account identifiers,
or the untouched test set.

## Result

- Model: `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`
- Hardware: one NVIDIA L40S
- Method: 4-bit QLoRA, rank 16, three predetermined epochs
- Training: 400 records, 150 optimizer steps, 98.7 seconds
- End-to-end GPU function: 165.5 seconds
- Final validation: 48/50 decisions correct and 50/50 raw outputs valid JSON
- Challenge recall: 96.67%
- Comply accuracy: 95%
- False-objection rate: 5%
- Successful-run resource usage: `$0.13311718`
- Total including the stopped pre-training import attempt: `$0.13583609`
- Billed workspace cost at observation: `$0` after credits

The adapter is stored in the `fine-tuning-training-output` Modal Volume at
`/runs/primary-qlora-001/adapter` and published as
[`rrod/qwen3-4b-constructive-disagreement-lora`](https://huggingface.co/rrod/qwen3-4b-constructive-disagreement-lora)
at original weight revision `eae43ddb16c580c4179ed8088c8180b28fb4d572`.
Documentation-only revision `574ade5ec9ca7f2ac834987f733145dd716432eb`
retains the same adapter files. The main safetensors SHA-256 is
`8b211ae87e6f7b302f3c2adb80a22bbf2bae129d8d683285f77470e44cbdc63f`.
A separate read-only Modal worker verified the persisted hashes and evidence;
the publisher independently downloaded and byte-verified the Hub revision.

## Evidence

- `manifest.json`: frozen inputs, packages, configuration, timing, token counts,
  trainer history, validation summary, and adapter hashes.
- `validation-generations.jsonl`: all 50 deterministic validation outputs and
  parsing/classification results.
- `validation-metrics.json`: derived validation behavior metrics.
- `cost.json`: successful and aborted-attempt resource usage with credit
  treatment separated.

Validation loss was `1.0946`, `0.9738`, and `1.0300` after epochs one, two, and
three. The final adapter remains the predetermined third epoch; it was not
selected after inspecting validation loss. Two validation decisions were
incorrect: `validation-0004` was a false objection and `validation-0017` was a
missed challenge. The subsequently completed one-time test comparison is
reported separately in `docs/evaluation.md` and
`artifacts/runs/held-out-evaluation-001/`.

The provider emitted non-fatal warnings about its Linux kernel version, a
deprecated `warmup_ratio` spelling, tokenizer/config alignment, and redundant
generation-length defaults. None changed the completed run or its persisted
artifacts; they remain part of the run history rather than being silently
removed.

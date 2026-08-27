# Evaluation

## Protocol

The same 100 held-out requests were evaluated once under three frozen
conditions:

| Condition | Model | Prompt |
| --- | --- | --- |
| A | Untouched base model | Basic prompt |
| B | Untouched base model | Strong prompt |
| C | LoRA adapter | Basic prompt |

Condition B prevents the experiment from comparing fine-tuning only with a
straw-man prompt. All conditions used the same pinned base model, tokenizer,
chat template, deterministic greedy decoding, seed 3407, and a maximum of 256
new tokens. Raw JSON was parsed without repair or retry.

The primary metric is balanced decision accuracy: the mean of flawed-request
recall and reasonable-request specificity. The evaluator also records JSON
validity, false objections, missed risks, paired bootstrap intervals, and an
exact two-sided McNemar comparison.

The material-win rule was frozen before evaluation: the adapter needed at
least an eight-point balanced-accuracy improvement over the strong prompt while
losing no more than five specificity points.

## Results

| Condition | Valid JSON | Flawed recall | Reasonable specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Base + basic | 87% | 94% | 68% | 81% |
| Base + strong | 100% | 96% | 96% | 96% |
| LoRA + basic | 100% | 100% | 94% | 97% |

The LoRA adapter improved balanced accuracy over the basic prompt by 16 points
(paired bootstrap 95% interval: +9 to +24). Its one-point improvement over the
strong prompt had an interval from -3 to +5 and did not meet the material-win
threshold.

The adapter caught all 50 flawed requests and falsely challenged three of 50
reasonable requests. Its clearest failure reversed a feasible timeline: access
began Monday and completion was due Friday, but the adapter claimed completion
had to precede access. That failure remains in the frozen results.

## Reproduce the comparison

The local plan verifies every input hash and prints the exact GPU, timeout,
cost ceiling, and paid command without contacting Modal:

```bash
uv run --directory apps/modeling --frozen disagree-evaluation-plan
```

Run the paid evaluation only after the corresponding adapter exists in the
training-output Volume:

```bash
uv run --directory apps/modeling --frozen modal run \
  -m disagree_modeling.modal_evaluate \
  --run-id held-out-evaluation-001 --execute
```

The evaluation image receives the test split but not the train or validation
splits. It mounts the trained adapter read-only and writes results to a separate
Modal Volume.

Canonical evidence:

- [`artifacts/runs/held-out-evaluation-001/metrics.json`](../artifacts/runs/held-out-evaluation-001/metrics.json)
- [`artifacts/runs/held-out-evaluation-001/generations.jsonl`](../artifacts/runs/held-out-evaluation-001/generations.jsonl)
- [`artifacts/runs/held-out-evaluation-001/manifest.json`](../artifacts/runs/held-out-evaluation-001/manifest.json)
- [`artifacts/runs/primary-qlora-001/`](../artifacts/runs/primary-qlora-001/)

This result supports a narrow behavioral claim only. It does not show that the
model acquired general judgment, understanding, or production safety.

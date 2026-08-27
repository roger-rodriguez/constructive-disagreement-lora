# held-out-evaluation-001

This immutable bundle records the single evaluation of the untouched
100-example test split. It compares the base model with the basic prompt (A),
the base model with the strong prompt (B), and the LoRA adapter with the basic
prompt (C). No prompt, label, adapter, or decoding rule was changed after the
outputs were generated.

## Verdict

| Condition | Valid JSON | Recall | Specificity | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| A: base + basic | 87% | 94% | 68% | 81% |
| B: base + strong | 100% | 96% | 96% | 96% |
| C: LoRA + basic | 100% | 100% | 94% | 97% |

C improved balanced accuracy over A by 16 percentage points (paired
stratified-bootstrap 95% CI: +9 to +24; exact McNemar `p=0.00014496`). C
improved over B by only one point (95% CI: -3 to +5; exact McNemar `p=1.0`)
and lost two points of specificity. It therefore did not meet the predeclared
material-win threshold of at least +8 points over B with no more than a
five-point specificity loss.

The adapter caught all 50 flawed requests but falsely challenged three of the
50 reasonable requests. Those failures are retained in `generations.jsonl`.

## Evidence

- `manifest.json`: immutable inputs, package versions, timings, artifact
  hashes, conditions, and aggregate metrics.
- `metrics.json`: raw counts, rates, paired bootstrap intervals, exact McNemar
  tests, and the frozen threshold verdict.
- `generations.jsonl`: all 300 raw outputs with condition labels, parsed
  decisions, and binary correctness.
- `blinded-generations.jsonl`: deterministic condition- and gold-blinded rows
  for optional qualitative scoring.
- `blinding-key.json`: the separately stored unblinding map.
- `cost.json`: actual resource usage with credits separated from usage.

A separate read-only Modal worker verified the persisted remote artifacts and
their hashes. Local SHA-256 checks also match the manifest. Qualitative ratings
were not required for, and are not included in, the automatic binary result.

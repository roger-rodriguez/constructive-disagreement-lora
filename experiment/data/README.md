# Frozen dataset

`splits/` contains the 400 training, 50 validation, and 100 held-out test
records. Matching files in `metadata/` preserve domain, taxonomy category,
minimal-pair IDs, and structured provenance without adding those labels to the
model input.

Every source record contains one request and one typed target:

```json
{"id":"train-0001","input":"...","target":{"decision":"challenge","issue":"...","message":"...","question":null,"suggested_next_step":"..."}}
```

`manifest.json` freezes the data, prompts, and response schema by hash.
`quality-report.json` records automated duplicate, length, marker, and
provenance checks. Run both executable checks from the repository root:

```bash
uv run --directory apps/modeling --frozen disagree-freeze-data \
  --experiment-root ../../experiment --check
uv run --frozen python scripts/build_data_quality_report.py --check
```

The executable contracts live in `packages/contracts/`; cross-record validation
lives in `apps/modeling/`. See [`docs/dataset.md`](../../docs/dataset.md) for
the full public data card.

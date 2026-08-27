# Dataset

## Goal

The dataset teaches one narrow behavior: recognize a materially flawed
workplace request, identify one primary issue, and offer a constructive path
forward without objecting to reasonable requests.

All 550 scenarios are newly invented and generic. They contain no copied
tickets, real employee or customer conversations, company names, production
identifiers, or confidential repository details. The data is agent-generated
and agent-reviewed; it is not described as human-labeled or expert-audited.

## Composition

| Split | Records | Challenge | Comply | Minimal pairs |
| --- | ---: | ---: | ---: | ---: |
| Train | 400 | 240 | 160 | 80 |
| Validation | 50 | 30 | 20 | 10 |
| Held-out test | 100 | 50 | 50 | 20 |

The examples cover product requirements, engineering estimates, project
planning, customer requests, incident response, hiring and team management,
and AI-agent authorization. The taxonomy deliberately includes reasonable
requests and safe near-neighbors so the model cannot succeed by always saying
no.

## Record format

Each JSONL record contains a stable ID, one request, and a typed target:

```json
{
  "id": "train-0001",
  "input": "Make deletion immediate, but allow restoration for 30 days.",
  "target": {
    "decision": "challenge",
    "issue": "Immediate irreversible deletion conflicts with restoration.",
    "message": "Restoration requires retaining recoverable data.",
    "question": null,
    "suggested_next_step": "Use a recoverable pending-deletion period."
  }
}
```

`decision` is `challenge` or `comply`. A challenge names exactly one primary
issue and supplies a next step. `question` contains at most one question. A
comply response uses `issue: null` and does not invent an objection.

Sidecar metadata stores domain, taxonomy category, decision, minimal-pair ID,
and provenance separately from the model input. This avoids teaching the model
through category names while preserving stratified evaluation.

## Designing a dataset for another task

1. Define an output contract that can be validated mechanically.
2. Write both positive and negative cases; avoid a dataset where one response
   posture always wins.
3. Create minimal pairs that change one material fact while keeping incidental
   wording similar.
4. Keep the held-out split inaccessible to the training job.
5. Freeze prompts, data, and hashes before inspecting test results.
6. Compare the fine-tune with both a minimal prompt and the strongest practical
   prompt you can write.
7. Retain failures instead of revising the benchmark after seeing the result.

The executable record contract lives in
[`packages/contracts/src/disagree_contracts/schemas.py`](../packages/contracts/src/disagree_contracts/schemas.py).
Cross-record validation is implemented in
[`apps/modeling/src/disagree_modeling/data.py`](../apps/modeling/src/disagree_modeling/data.py).

## Integrity checks

`experiment/data/manifest.json` records counts, distributions, sizes, and
SHA-256 hashes for every split, sidecar, prompt, and schema file.
`experiment/data/quality-report.json` checks review completion, exact and near
duplicates, field length, and URL or email markers.

```bash
uv run --directory apps/modeling --frozen disagree-freeze-data \
  --experiment-root ../../experiment --check
uv run --frozen python scripts/build_data_quality_report.py --check
```

Known limitations include shared blind spots between agent generation and
agent review, short English-only scenarios, synthetic wording, and a binary
label that flattens some reasonable ambiguity.

The dataset, prompts, metadata, and documentation are licensed under CC BY 4.0.

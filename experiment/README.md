# Experiment inputs

This directory contains the frozen public inputs for the reported experiment:
the synthetic dataset, taxonomy, prompts, response schema, and integrity
hashes. Keeping these inputs separate from implementation makes the research
contract easy to inspect and reuse.

- `data/splits/`: train, validation, and held-out test examples.
- `data/metadata/`: sidecar labels and provenance used for validation and
  stratified analysis, never shown to the model.
- `data/manifest.json`: counts, distributions, sizes, and SHA-256 hashes.
- `data/quality-report.json`: automated dataset-quality results.
- `prompts/`: frozen basic and strong baseline prompts.
- `schema/`: the structured response contract.
- `taxonomy.md`: the behavioral categories used to balance the dataset.

See [`docs/dataset.md`](../docs/dataset.md) for the record format, design
principles, limitations, and adaptation guidance.

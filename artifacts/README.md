# Experiment artifacts

`runs/primary-qlora-001/` contains the training manifest, validation
generations, validation metrics, and cost report. The adapter itself is
published separately on Hugging Face and represented here by its SHA-256 hash.

`runs/held-out-evaluation-001/` contains all 300 raw generations, deterministic
blinding records, binary metrics, paired statistics, manifest, and cost report.

Run directories are immutable evidence. A changed dataset, prompt, model,
configuration, or rerun receives a new ID rather than replacing a reported
result.

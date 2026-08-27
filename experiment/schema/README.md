# Frozen response contract

`decision-response-v1.schema.json` is the portable version-one JSON Schema for
model output. The stricter executable checks in
`packages/contracts/src/disagree_contracts/schemas.py` additionally enforce semantic
rules such as exactly one question and required challenge fields.

Version one has exactly five ordered fields, no extra properties, no arrays,
and no `unclear` model target. `unclear` is reserved for reviewer uncertainty.
Any incompatible change requires a new schema file and a new experiment
version.

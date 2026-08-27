# Contracts

`disagree-contracts` owns stable data and model-response contracts shared by
the modeling and serving apps: typed schemas, identifiers, prompt rendering,
and strict output parsing. It must not depend on training or serving frameworks.

The current data contract is implemented with the Python standard library so
both apps can validate public records without inheriting Transformers,
training, or serving dependencies. Structured decision examples keep `input`
and a typed five-field `target` in source data; the prompt renderer converts
the target to canonical JSON for model training and strict output parsing.

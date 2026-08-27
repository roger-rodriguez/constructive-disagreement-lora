# Frozen baseline prompts

`basic.txt` and `strong.txt` are the version-one baseline prompts for the
reported experiment. They were frozen before full dataset generation, model
training, or inspection of any fine-tuned test output.

Both prompts require the same five-field JSON response contract. The basic
prompt provides only the minimum task distinction and structural rules. The
strong prompt is intentionally competitive: it defines the challenge rubric,
warns against false objections, and gives field-level quality guidance.

Prompt hashes are recorded in `experiment/data/manifest.json` when the dataset
is frozen. Changing either prompt after that point creates a new experiment
version rather than silently replacing a baseline.

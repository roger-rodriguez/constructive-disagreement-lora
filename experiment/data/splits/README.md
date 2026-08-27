# Splits

The reported `test.jsonl`, `validation.jsonl`, and `train.jsonl` files were
created in that order. The untouched test set was authored first so later
training examples could be checked against it. All three splits are frozen
after independent agent review, cross-split duplicate checks, and manifest
hashing.

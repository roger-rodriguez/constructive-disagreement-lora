# Modeling configuration

`model-requirements.txt` is generated from the modeling app's locked `model`
dependency group. It contains the complete hash-pinned environment used to
build the Modal model image. Regenerate it only with an
intentional dependency change:

```bash
uv export --directory apps/modeling --only-group model \
  --no-emit-project --no-emit-local --locked \
  --output-file configs/model-requirements.txt
```

The primary training configuration is fixed in
`disagree_modeling.training.TrainingConfiguration`. It reuses this exact locked
ML environment. Evaluation remains a separate entrypoint so the untouched test
split is not available to the training image.

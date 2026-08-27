# Serving configuration

`model.toml` is the public, secret-free serving contract. It pins the base
model and adapter commits, frozen prompt digest, decoding bounds, Modal GPU,
concurrency, and scale-to-zero settings.

`runtime-requirements.txt` is generated from the `serve` dependency group:

```bash
uv export --directory apps/serving --only-group serve \
  --no-emit-project --no-emit-local --locked \
  --output-file configs/runtime-requirements.txt
```

Environment-specific endpoint URLs and proxy credentials belong in a local
`.env`, using the names documented in the repository `.env.example`. They are
not model configuration and must never be committed.

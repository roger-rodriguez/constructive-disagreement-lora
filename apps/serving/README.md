# Serving

This independently deployable Modal app serves the public LoRA adapter from
Hugging Face. Its serving dependencies, configuration, and deployment
lifecycle remain separate from training.

It exposes two routes:

- `health`: public CPU-only `GET` health check; it never starts a GPU.
- `decide`: proxy-authenticated `POST`; it loads the pinned Qwen base and public
  adapter on one L4 and returns the validated five-field JSON object.

Both routes use zero minimum containers. The model route allows one concurrent
request in one container and scales down after 60 idle seconds. Model downloads
are cached in the existing `fine-tuning-model-cache` Modal Volume. No Hugging
Face token is needed because both repositories are public.

## Configuration

[`configs/model.toml`](configs/model.toml) is the secret-free serving contract.
It pins both Hugging Face commit revisions, the prompt digest, decoding limits,
GPU resources, concurrency, and scaling. The fully resolved container
dependencies are in `configs/runtime-requirements.txt`.

Copy [`.env.example`](../../.env.example) to a gitignored `.env` for endpoint
URLs and Modal proxy credentials. Never put credentials in `model.toml` or
commit `.env`. For a public browser demo, place a small rate-limited server in
front of this endpoint; do not ship a Modal proxy token to browser JavaScript.

## Deploy

Authenticate the local Modal CLI once, then deploy:

```bash
uv run --directory apps/serving --frozen modal deploy \
  -m disagree_serving.modal_app \
  --name constructive-disagreement
```

Deployment builds the image but does not keep an L4 running. The first model
request pays the cold-start cost; later requests reuse the warm worker until it
scales to zero.

## Request

Set the values from `.env.example`, then call the model endpoint:

```bash
curl --fail-with-body --location \
  --request POST "$MODAL_DECISION_URL" \
  --header "Content-Type: application/json" \
  --header "Modal-Key: $MODAL_PROXY_TOKEN_ID" \
  --header "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET" \
  --data '{"input":"Delete the account immediately, but keep it recoverable for 30 days."}'
```

The response is exactly one object with `decision`, `issue`, `message`,
`question`, and `suggested_next_step`. Generation is greedy and limited to 256
new tokens. Inputs are limited to 2,000 characters and 768 rendered tokens.
Raw output is never repaired or retried; invalid model JSON produces a `502`.

An unauthenticated request is rejected by Modal before a GPU worker starts.

## Stop

To remove the deployment later:

```bash
uv run --directory apps/serving --frozen modal app stop \
  constructive-disagreement
```

Stopping a Modal App is permanent for that deployment. Running the deploy
command again creates a new deployment from the source.

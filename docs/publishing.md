# Publishing the adapter

The Hugging Face repository is a separate model release. Do not add the
132 MB adapter to this Git repository.

Published release:

- Repository: `rrod/qwen3-4b-constructive-disagreement-lora`
- Revision: `574ade5ec9ca7f2ac834987f733145dd716432eb`
- Adapter digest unchanged from the original weight revision
  `eae43ddb16c580c4179ed8088c8180b28fb4d572`
- Visibility: public
- Remote byte verification: passed
- Receipt: [`huggingface-primary-qlora-001.json`](../artifacts/releases/huggingface-primary-qlora-001.json)

1. Download `/runs/primary-qlora-001/adapter` from the
   `fine-tuning-training-output` Modal Volume to a local temporary directory.
2. Authenticate with a Hugging Face write token using `hf auth login`.
3. Verify and stage the allowlisted release locally:

   ```bash
   uv run --directory apps/modeling --group publish --frozen \
     disagree-publish-adapter verify \
     --adapter-dir /path/to/adapter \
     --output-dir ../../.release/huggingface-adapter
   ```

4. Inspect the staged model card, `release-manifest.json`, and file list.
5. Publish only after selecting the exact namespace and repository ID:

   ```bash
   uv run --directory apps/modeling --group publish --frozen \
     disagree-publish-adapter publish \
     --adapter-dir /path/to/adapter \
     --repo-id OWNER/qwen3-4b-constructive-disagreement-lora \
     --public \
     --yes
   ```

The publisher creates an allowlisted folder, uploads it with the official Hub
client, downloads the immutable Hub commit, and byte-verifies every file. It
never uploads `training_args.bin`, checkpoints, logs, tokens, or the base model.

The exact repository ID, commit, adapter digest, and verification result are
recorded in the source repository. Never silently replace the published adapter
with a retrained version; use a new tagged release or model repository.

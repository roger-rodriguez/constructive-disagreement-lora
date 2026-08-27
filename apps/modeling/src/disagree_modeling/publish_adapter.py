"""Verify, stage, and publish the immutable LoRA adapter to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "artifacts" / "runs" / "primary-qlora-001" / "manifest.json"
)
DEFAULT_MODEL_CARD = REPOSITORY_ROOT / "docs" / "model-card.md"
DEFAULT_LICENSE = REPOSITORY_ROOT / "LICENSE"

ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)
STAGED_STATIC_FILES = ("LICENSE", "README.md")
RELEASE_MANIFEST_NAME = "release-manifest.json"
REPO_ID_PLACEHOLDER = "{{HF_REPO_ID}}"


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def verify_adapter_bundle(adapter_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify the publishable files against the immutable training manifest."""
    manifest = _read_json_object(manifest_path)
    expected_hashes = manifest.get("artifact_sha256")
    if not isinstance(expected_hashes, dict):
        raise TypeError("training manifest has no artifact hashes")

    files: dict[str, dict[str, int | str]] = {}
    for name in ADAPTER_FILES:
        path = adapter_dir / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or unsafe adapter file: {name}")
        digest = sha256_file(path)
        expected = expected_hashes.get(name)
        if expected is not None and digest != expected:
            raise RuntimeError(f"training manifest hash mismatch for {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": digest}

    adapter_config = _read_json_object(adapter_dir / "adapter_config.json")
    if adapter_config.get("base_model_name_or_path") != manifest.get("model"):
        raise RuntimeError("adapter base model does not match the training manifest")
    if adapter_config.get("peft_type") != "LORA":
        raise RuntimeError("adapter is not identified as LoRA")
    if adapter_config.get("task_type") != "CAUSAL_LM":
        raise RuntimeError("adapter task type is not causal language modeling")

    return {
        "schema_version": 1,
        "source_run_id": manifest.get("run_id"),
        "source_repository": "https://github.com/roger-rodriguez/constructive-disagreement-lora",
        "base_model": manifest.get("model"),
        "base_model_revision": manifest.get("model_revision"),
        "base_model_license": manifest.get("model_license"),
        "adapter_license": "Apache-2.0",
        "training_data_license": "CC-BY-4.0",
        "files": files,
        "excluded_source_files": [
            "README.md (replaced by the reviewed model card)",
            "training_args.bin (unnecessary pickle serialization)",
        ],
    }


def stage_adapter_release(
    adapter_dir: Path,
    manifest_path: Path,
    model_card_path: Path,
    license_path: Path,
    output_dir: Path,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Create an allowlisted, publication-ready adapter directory."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"release staging directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    release_manifest = verify_adapter_bundle(adapter_dir, manifest_path)
    for name in ADAPTER_FILES:
        shutil.copy2(adapter_dir / name, output_dir / name)
    model_card = model_card_path.read_text(encoding="utf-8")
    if repo_id is not None:
        model_card = model_card.replace(REPO_ID_PLACEHOLDER, repo_id)
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")
    shutil.copy2(license_path, output_dir / "LICENSE")

    staged_files = {
        name: {
            "bytes": (output_dir / name).stat().st_size,
            "sha256": sha256_file(output_dir / name),
        }
        for name in (*ADAPTER_FILES, *STAGED_STATIC_FILES)
    }
    release_manifest["published_files"] = staged_files
    (output_dir / RELEASE_MANIFEST_NAME).write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-card", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--license-file", type=Path, default=DEFAULT_LICENSE)


def build_parser() -> argparse.ArgumentParser:
    """Build the release CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="verify and stage without network use")
    _common_arguments(verify)
    verify.add_argument("--output-dir", type=Path)

    publish = commands.add_parser("publish", help="publish to Hugging Face")
    _common_arguments(publish)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--public", action="store_true")
    publish.add_argument("--allow-existing", action="store_true")
    publish.add_argument(
        "--yes",
        action="store_true",
        help="confirm the external repository mutation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the adapter verification or publication command."""
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        if args.output_dir is None:
            with tempfile.TemporaryDirectory(
                prefix="disagree-adapter-verify-"
            ) as directory:
                result = stage_adapter_release(
                    args.adapter_dir,
                    args.training_manifest,
                    args.model_card,
                    args.license_file,
                    Path(directory),
                )
        else:
            result = stage_adapter_release(
                args.adapter_dir,
                args.training_manifest,
                args.model_card,
                args.license_file,
                args.output_dir,
            )
    else:
        if not args.yes:
            raise SystemExit("publishing requires --yes")
        from disagree_modeling.publish_adapter_runtime import publish_adapter

        result = publish_adapter(
            repo_id=args.repo_id,
            adapter_dir=args.adapter_dir,
            manifest_path=args.training_manifest,
            model_card_path=args.model_card,
            license_path=args.license_file,
            public=args.public,
            allow_existing=args.allow_existing,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

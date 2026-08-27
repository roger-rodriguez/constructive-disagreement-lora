"""Network runtime for publishing and verifying the Hugging Face adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from disagree_modeling.publish_adapter import (
    ADAPTER_FILES,
    RELEASE_MANIFEST_NAME,
    STAGED_STATIC_FILES,
    sha256_file,
    stage_adapter_release,
)

HUGGING_FACE_MANAGED_FILES = {".gitattributes"}


def publish_adapter(
    *,
    repo_id: str,
    adapter_dir: Path,
    manifest_path: Path,
    model_card_path: Path,
    license_path: Path,
    public: bool,
    allow_existing: bool,
) -> dict[str, Any]:
    """Publish and byte-verify the allowlisted adapter release."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname")
    if not username:
        raise RuntimeError("could not resolve the authenticated Hugging Face user")

    with tempfile.TemporaryDirectory(prefix="disagree-adapter-release-") as directory:
        stage_dir = Path(directory) / "upload"
        release_manifest = stage_adapter_release(
            adapter_dir,
            manifest_path,
            model_card_path,
            license_path,
            stage_dir,
            repo_id,
        )
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=not public,
            exist_ok=allow_existing,
        )
        if allow_existing:
            api.update_repo_settings(repo_id=repo_id, private=not public)
        commit = api.upload_folder(
            folder_path=stage_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish primary-qlora-001 adapter",
        )

        remote_dir = Path(directory) / "verified-download"
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=commit.oid,
            local_dir=remote_dir,
        )
        expected_names = {
            *ADAPTER_FILES,
            *STAGED_STATIC_FILES,
            RELEASE_MANIFEST_NAME,
        }
        remote_names = set()
        for path in remote_dir.rglob("*"):
            relative = path.relative_to(remote_dir)
            if path.is_file() and relative.parts[0] != ".cache":
                remote_names.add(relative.as_posix())
        unexpected = remote_names - expected_names - HUGGING_FACE_MANAGED_FILES
        missing = expected_names - remote_names
        if unexpected or missing:
            raise RuntimeError(
                f"remote file set mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for name in expected_names:
            if sha256_file(stage_dir / name) != sha256_file(remote_dir / name):
                raise RuntimeError(f"remote hash mismatch for {name}")

    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "commit": commit.oid,
        "public": public,
        "authenticated_as": username,
        "source_run_id": release_manifest["source_run_id"],
        "adapter_sha256": release_manifest["files"]["adapter_model.safetensors"][
            "sha256"
        ],
        "remote_verification": "passed",
    }

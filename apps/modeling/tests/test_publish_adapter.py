from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from disagree_modeling.publish_adapter import (
    ADAPTER_FILES,
    DEFAULT_MODEL_CARD,
    RELEASE_MANIFEST_NAME,
    main,
    stage_adapter_release,
    verify_adapter_bundle,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class AdapterReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.adapter_dir = self.root / "adapter"
        self.adapter_dir.mkdir()
        self.contents = {
            "adapter_config.json": json.dumps(
                {
                    "base_model_name_or_path": "example/base",
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                }
            ).encode(),
            "adapter_model.safetensors": b"safe adapter bytes",
            "chat_template.jinja": b"{{ messages }}",
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
        }
        for name, content in self.contents.items():
            (self.adapter_dir / name).write_bytes(content)
        (self.adapter_dir / "training_args.bin").write_bytes(b"exclude me")
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "run_id": "primary-qlora-001",
                    "model": "example/base",
                    "model_revision": "abc123",
                    "model_license": "Apache-2.0",
                    "artifact_sha256": {
                        name: _sha256(content)
                        for name, content in self.contents.items()
                    },
                }
            )
        )
        self.model_card = self.root / "README.md"
        self.model_card.write_text("---\nlicense: apache-2.0\n---\n")
        self.license_file = self.root / "LICENSE"
        self.license_file.write_text("Apache License")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_default_model_card_is_hugging_face_ready(self) -> None:
        content = DEFAULT_MODEL_CARD.read_text(encoding="utf-8")

        self.assertEqual(DEFAULT_MODEL_CARD.name, "model-card.md")
        self.assertTrue(content.startswith("---\n"))
        self.assertIn("base_model: Qwen/Qwen3-4B-Instruct-2507", content)

    def test_verify_accepts_matching_allowlisted_bundle(self) -> None:
        result = verify_adapter_bundle(self.adapter_dir, self.manifest_path)

        self.assertEqual(result["source_run_id"], "primary-qlora-001")
        self.assertEqual(set(result["files"]), set(ADAPTER_FILES))
        self.assertIn("training_args.bin", str(result["excluded_source_files"]))

    def test_verify_rejects_changed_verified_file(self) -> None:
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(b"changed")

        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            verify_adapter_bundle(self.adapter_dir, self.manifest_path)

    def test_verify_rejects_missing_file(self) -> None:
        (self.adapter_dir / "chat_template.jinja").unlink()

        with self.assertRaisesRegex(RuntimeError, "missing or unsafe"):
            verify_adapter_bundle(self.adapter_dir, self.manifest_path)

    def test_verify_rejects_nonobject_manifest(self) -> None:
        self.manifest_path.write_text("[]")

        with self.assertRaisesRegex(TypeError, "expected a JSON object"):
            verify_adapter_bundle(self.adapter_dir, self.manifest_path)

    def test_verify_rejects_manifest_without_hashes(self) -> None:
        self.manifest_path.write_text("{}")

        with self.assertRaisesRegex(TypeError, "no artifact hashes"):
            verify_adapter_bundle(self.adapter_dir, self.manifest_path)

    def test_verify_rejects_wrong_adapter_contract(self) -> None:
        config_path = self.adapter_dir / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["base_model_name_or_path"] = "wrong/base"
        changed = json.dumps(config).encode()
        config_path.write_bytes(changed)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["artifact_sha256"]["adapter_config.json"] = _sha256(changed)
        self.manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(RuntimeError, "base model"):
            verify_adapter_bundle(self.adapter_dir, self.manifest_path)

    def test_stage_excludes_pickle_and_replaces_model_card(self) -> None:
        output_dir = self.root / "release"

        stage_adapter_release(
            self.adapter_dir,
            self.manifest_path,
            self.model_card,
            self.license_file,
            output_dir,
        )

        self.assertFalse((output_dir / "training_args.bin").exists())
        self.assertEqual(
            (output_dir / "README.md").read_text(), self.model_card.read_text()
        )
        self.assertTrue((output_dir / RELEASE_MANIFEST_NAME).is_file())

    def test_stage_renders_repository_id_into_model_card(self) -> None:
        self.model_card.write_text('ADAPTER_ID = "{{HF_REPO_ID}}"')
        output_dir = self.root / "release"

        stage_adapter_release(
            self.adapter_dir,
            self.manifest_path,
            self.model_card,
            self.license_file,
            output_dir,
            "example/published-adapter",
        )

        self.assertEqual(
            (output_dir / "README.md").read_text(),
            'ADAPTER_ID = "example/published-adapter"',
        )

    def test_stage_rejects_nonempty_output_directory(self) -> None:
        output_dir = self.root / "release"
        output_dir.mkdir()
        (output_dir / "old.txt").write_text("old")

        with self.assertRaisesRegex(RuntimeError, "not empty"):
            stage_adapter_release(
                self.adapter_dir,
                self.manifest_path,
                self.model_card,
                self.license_file,
                output_dir,
            )

    def test_publish_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires --yes"):
            main(
                [
                    "publish",
                    "--adapter-dir",
                    str(self.adapter_dir),
                    "--training-manifest",
                    str(self.manifest_path),
                    "--model-card",
                    str(self.model_card),
                    "--license-file",
                    str(self.license_file),
                    "--repo-id",
                    "example/adapter",
                ]
            )

    def test_verify_cli_stages_release(self) -> None:
        output_dir = self.root / "cli-release"

        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "verify",
                    "--adapter-dir",
                    str(self.adapter_dir),
                    "--training-manifest",
                    str(self.manifest_path),
                    "--model-card",
                    str(self.model_card),
                    "--license-file",
                    str(self.license_file),
                    "--output-dir",
                    str(output_dir),
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue((output_dir / RELEASE_MANIFEST_NAME).is_file())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from disagree_modeling.training import (
    CONFIG,
    MODEL_ID,
    MODEL_REVISION,
    SPEND_CEILING_USD,
    build_training_plan,
    maximum_resource_cost_usd,
)
from disagree_modeling.training_plan import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"


class PrimaryTrainingContractTest(unittest.TestCase):
    def test_configuration_fixes_effective_batch_and_optimizer_steps(self) -> None:
        self.assertEqual(CONFIG.epochs, 3)
        self.assertEqual(CONFIG.effective_batch_size, 8)
        self.assertEqual(CONFIG.optimizer_steps, 150)
        self.assertEqual(CONFIG.max_sequence_length, 512)
        self.assertEqual(CONFIG.lora_rank, 16)

    def test_timeout_resource_cost_is_below_hard_ceiling(self) -> None:
        self.assertAlmostEqual(maximum_resource_cost_usd(), 2.395584)
        self.assertLess(maximum_resource_cost_usd(), SPEND_CEILING_USD)

    def test_run_plan_uses_frozen_inputs_and_hides_test_split(self) -> None:
        plan = build_training_plan(EXPERIMENT_ROOT, "primary-qlora-001")

        self.assertEqual(plan["operation"], "training")
        self.assertEqual(plan["model"], MODEL_ID)
        self.assertEqual(plan["model_revision"], MODEL_REVISION)
        self.assertEqual(plan["training_records"], 400)
        self.assertEqual(plan["validation_records"], 50)
        self.assertEqual(plan["test_records_available_to_job"], 0)
        self.assertEqual(plan["spend_ceiling_usd"], 3.0)
        self.assertIn("--execute", str(plan["paid_execution_command"]))

    def test_run_plan_rejects_changed_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "splits").mkdir(parents=True)
            (root / "prompts").mkdir()
            source_manifest = json.loads(
                (EXPERIMENT_ROOT / "data" / "manifest.json").read_text()
            )
            (root / "data" / "manifest.json").write_text(
                json.dumps(source_manifest),
                encoding="utf-8",
            )
            (root / "data" / "splits" / "train.jsonl").write_text(
                "changed\n",
                encoding="utf-8",
            )
            (root / "data" / "splits" / "validation.jsonl").write_bytes(
                (EXPERIMENT_ROOT / "data" / "splits" / "validation.jsonl").read_bytes()
            )
            (root / "prompts" / "basic.txt").write_bytes(
                (EXPERIMENT_ROOT / "prompts" / "basic.txt").read_bytes()
            )

            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                build_training_plan(root, "primary-qlora-001")

    def test_modal_image_includes_only_train_and_validation_splits(self) -> None:
        from disagree_modeling import modal_train

        self.assertEqual(modal_train.REMOTE_EXPERIMENT_ROOT, Path("/opt/experiment"))
        source = Path(modal_train.__file__).read_text(encoding="utf-8")
        self.assertIn('"train.jsonl"', source)
        self.assertIn('"validation.jsonl"', source)
        self.assertNotIn('"test.jsonl"', source)

    def test_plan_cli_prints_json_without_contacting_modal(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["operation"], "training")


if __name__ == "__main__":
    unittest.main()

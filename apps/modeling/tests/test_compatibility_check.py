from __future__ import annotations

import inspect
import io
import json
import unittest
from contextlib import redirect_stdout

from disagree_contracts.prompt_rendering import IGNORE_INDEX
from disagree_modeling.compatibility_check import (
    MODEL_ID,
    MODEL_REVISION,
    SPEND_CEILING_USD,
    build_compatibility_plan,
    compatibility_record,
    labels_select_only_assistant,
    maximum_resource_cost_usd,
    prepare_assistant_suffix,
    require_exact_model_resolution,
)
from disagree_modeling.compatibility_plan import main


class CompatibilityCheckContractTest(unittest.TestCase):
    def test_record_is_fictional_single_turn_conversation(self) -> None:
        record = compatibility_record()

        self.assertEqual(record.id, "compatibility-check-001")
        self.assertEqual(len(record.messages), 2)

    def test_masks_only_the_assistant_suffix(self) -> None:
        example = prepare_assistant_suffix([1, 2, 3], [1, 2, 3, 4, 5])

        self.assertEqual(example.input_ids, (1, 2, 3, 4, 5))
        self.assertEqual(example.attention_mask, (1, 1, 1, 1, 1))
        self.assertEqual(
            example.labels,
            (IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5),
        )
        self.assertEqual(example.prompt_tokens, 3)
        self.assertEqual(example.assistant_tokens, 2)
        self.assertTrue(labels_select_only_assistant(example))

    def test_rejects_non_prefix_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            prepare_assistant_suffix([1, 9], [1, 2, 3])

    def test_rejects_sequence_without_assistant_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "assistant tokens"):
            prepare_assistant_suffix([1, 2], [1, 2])

    def test_rejects_invalid_token_sequences(self) -> None:
        invalid_values = ([], "1,2", [1, "2"], [True, 2])
        for value in invalid_values:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    "token sequence|integer token IDs",
                ),
            ):
                prepare_assistant_suffix(value, [1, 2, 3])

    def test_resource_ceiling_is_below_spend_limit(self) -> None:
        self.assertAlmostEqual(maximum_resource_cost_usd(), 0.598896)
        self.assertLess(maximum_resource_cost_usd(), SPEND_CEILING_USD)

    def test_requires_exact_model_repository_and_revision(self) -> None:
        require_exact_model_resolution(MODEL_ID, MODEL_REVISION)

        with self.assertRaisesRegex(RuntimeError, "model loader resolved"):
            require_exact_model_resolution("unsloth/qwen-mirror", MODEL_REVISION)
        with self.assertRaisesRegex(RuntimeError, "immutable revision"):
            require_exact_model_resolution(MODEL_ID, "moving-main")

    def test_run_plan_pins_model_and_paid_command(self) -> None:
        plan = build_compatibility_plan()

        self.assertEqual(plan["model"], MODEL_ID)
        self.assertEqual(plan["model_revision"], MODEL_REVISION)
        self.assertEqual(plan["operation"], "compatibility_check")
        self.assertIn("--execute", str(plan["paid_execution_command"]))
        self.assertIn(
            "--run-id compatibility-check-002",
            str(plan["paid_execution_command"]),
        )

    def test_plan_cli_prints_json_without_contacting_modal(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["operation"], "compatibility_check"
        )

    def test_modal_scaffold_resolves_locked_requirements_locally(self) -> None:
        from disagree_modeling import modal_compatibility

        self.assertTrue(modal_compatibility.REQUIREMENTS_PATH.is_file())
        self.assertEqual(modal_compatibility.MODEL_CACHE_PATH, "/vol/model-cache")
        self.assertEqual(modal_compatibility.DATASET_CACHE_PATH, "/vol/dataset-cache")
        self.assertEqual(modal_compatibility.OUTPUT_PATH, "/vol/outputs")
        self.assertEqual(modal_compatibility.TRAIN_ENV["HF_XET_HIGH_PERFORMANCE"], "1")
        self.assertNotIn(
            "HF_HUB_ENABLE_HF_TRANSFER",
            modal_compatibility.TRAIN_ENV,
        )

    def test_gpu_runtime_passes_an_explicit_inference_attention_mask(self) -> None:
        from disagree_modeling import compatibility_runtime

        source = inspect.getsource(compatibility_runtime.run_compatibility_check)

        self.assertIn(
            "inference_attention_mask = torch.ones_like(inference_input)",
            source,
        )
        self.assertIn("attention_mask=inference_attention_mask", source)


if __name__ == "__main__":
    unittest.main()

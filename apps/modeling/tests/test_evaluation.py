from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from disagree_modeling.evaluation import (
    ADAPTER_SHA256,
    CONDITIONS,
    SPEND_CEILING_USD,
    blinded_rows,
    build_evaluation_plan,
    comparison_metrics,
    condition_metrics,
    maximum_resource_cost_usd,
    score_output,
)
from disagree_modeling.evaluation_plan import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"


def _rows(condition: str, correct: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(100):
        gold = "challenge" if index < 50 else "comply"
        is_correct = index < correct
        parsed = (
            gold if is_correct else ("comply" if gold == "challenge" else "challenge")
        )
        rows.append(
            {
                "id": f"test-{index:04d}",
                "condition": condition,
                "input": f"request {index}",
                "gold_decision": gold,
                "raw_output": "{}",
                "parsed_decision": parsed,
                "valid_json": True,
                "decision_correct": is_correct,
                "parse_error": None,
            }
        )
    return rows


class HeldOutEvaluationContractTest(unittest.TestCase):
    def test_scores_valid_and_invalid_outputs_without_repair(self) -> None:
        valid = score_output(
            "comply",
            json.dumps(
                {
                    "decision": "comply",
                    "issue": None,
                    "message": "Proceed with the stated constraints.",
                    "question": None,
                    "suggested_next_step": None,
                }
            ),
        )
        invalid = score_output("challenge", "not json")

        self.assertTrue(valid.valid_json)
        self.assertTrue(valid.decision_correct)
        self.assertEqual(invalid.parsed_decision, "unclear")
        self.assertFalse(invalid.valid_json)
        self.assertFalse(invalid.decision_correct)

    def test_condition_metrics_use_balanced_held_out_counts(self) -> None:
        metrics = condition_metrics(_rows("A", 90))

        self.assertEqual(metrics["decision_correct"], 90)
        self.assertEqual(metrics["flawed_request_recall"], 1.0)
        self.assertEqual(metrics["reasonable_request_specificity"], 0.8)
        self.assertEqual(metrics["balanced_decision_accuracy"], 0.9)
        self.assertEqual(metrics["false_objection_rate"], 0.2)

    def test_comparison_applies_predeclared_material_win_threshold(self) -> None:
        compared = comparison_metrics(
            {"A": _rows("A", 70), "B": _rows("B", 80), "C": _rows("C", 90)}
        )

        material = compared["comparisons"]["material_win_over_strong_prompt"]
        self.assertTrue(material["threshold_met"])
        self.assertEqual(material["observed_balanced_accuracy_gain"], 0.1)
        mcnemar = compared["comparisons"]["C_minus_B"]["mcnemar_exact_two_sided"]
        self.assertEqual(mcnemar["discordant_pairs"], 10)

    def test_blinding_is_deterministic_and_omits_condition(self) -> None:
        rows = _rows("A", 100)[:2] + _rows("B", 100)[:2]
        first, first_key = blinded_rows(rows)
        second, second_key = blinded_rows(rows)

        self.assertEqual(first, second)
        self.assertEqual(first_key, second_key)
        self.assertNotIn("condition", first[0])
        self.assertNotIn("gold_decision", first[0])
        self.assertEqual(len(first_key["mapping"]), 4)

    def test_plan_pins_all_conditions_hashes_and_cost_boundary(self) -> None:
        plan = build_evaluation_plan(EXPERIMENT_ROOT, "held-out-evaluation-001")

        self.assertEqual(plan["operation"], "held_out_evaluation")
        self.assertEqual(plan["conditions"], CONDITIONS)
        self.assertEqual(plan["total_generations"], 300)
        self.assertEqual(plan["frozen_inputs"]["adapter_model_sha256"], ADAPTER_SHA256)
        self.assertEqual(plan["statistics"]["bootstrap_samples"], 10_000)
        self.assertLess(maximum_resource_cost_usd(), SPEND_CEILING_USD)
        self.assertIn("--execute", plan["paid_execution_command"])

    def test_plan_cli_prints_json_without_contacting_modal(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue())["operation"], "held_out_evaluation"
        )

    def test_modal_image_mounts_test_but_not_training_or_validation_splits(
        self,
    ) -> None:
        from disagree_modeling import modal_evaluate

        source = Path(modal_evaluate.__file__).read_text(encoding="utf-8")
        self.assertIn('"test.jsonl"', source)
        self.assertNotIn('"train.jsonl"', source)
        self.assertNotIn('"validation.jsonl"', source)


if __name__ == "__main__":
    unittest.main()

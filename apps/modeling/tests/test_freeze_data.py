from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from disagree_contracts.schemas import Category, Domain, Split
from disagree_modeling import freeze_data
from disagree_modeling.freeze_data import (
    FreezeDataError,
    SplitExpectation,
    freeze_dataset,
)

CHALLENGE_CATEGORIES = [
    "unsupported_assumption_or_missing_evidence",
    "internal_contradiction",
    "missing_material_constraint",
    "implausible_estimate_or_schedule",
    "authorization_privacy_security_or_operational_risk",
    "material_harm_or_unethical_request",
]
COMPLY_CATEGORIES = [
    "straightforward_reasonable_request",
    "constrained_but_reasonable_request",
    "safe_near_neighbor_to_flawed_request",
]
DOMAINS = [domain.value for domain in Domain]
SMALL_EXPECTATIONS = {
    split: SplitExpectation(
        records=9,
        decisions={"challenge": 6, "comply": 3},
        minimal_pairs=0,
    )
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST)
}


class FreezeDataTest(unittest.TestCase):
    def test_writes_and_checks_deterministic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)

            with patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS):
                manifest = freeze_dataset(root)
                checked = freeze_dataset(root, check=True)

            self.assertEqual(checked, manifest)
            self.assertEqual(manifest["schema_version"], 1)
            train = manifest["splits"]["train"]
            self.assertEqual(train["examples"]["record_count"], 9)
            self.assertEqual(
                train["summary"]["decisions"], {"challenge": 6, "comply": 3}
            )
            basic = root / "prompts" / "basic.txt"
            self.assertEqual(
                manifest["prompts"]["basic"]["sha256"],
                hashlib.sha256(basic.read_bytes()).hexdigest(),
            )
            stored = json.loads(
                (root / "data" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored, manifest)

    def test_rejects_wrong_size_and_decision_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            self._remove_record(root, Split.TRAIN, 0)

            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "expected 9 records"),
            ):
                freeze_dataset(root)

    def test_requires_every_domain_and_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            metadata = self._read_jsonl(root / "data" / "metadata" / "test.jsonl")
            metadata[0]["category"] = CHALLENGE_CATEGORIES[1]
            self._write_jsonl(root / "data" / "metadata" / "test.jsonl", metadata)

            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "missing categories"),
            ):
                freeze_dataset(root)

    def test_requires_expected_pair_and_training_category_counts(self) -> None:
        summary = freeze_data.ValidationSummary(
            records=9,
            minimal_pairs=0,
            decisions={"challenge": 6, "comply": 3},
            domains={domain: 1 for domain in DOMAINS},
            categories={
                category: 1 for category in CHALLENGE_CATEGORIES + COMPLY_CATEGORIES
            },
        )
        with self.assertRaisesRegex(FreezeDataError, "expected 1 minimal pairs"):
            freeze_data._validate_expected_distribution(
                Split.TRAIN,
                summary,
                SplitExpectation(
                    records=9,
                    decisions={"challenge": 6, "comply": 3},
                    minimal_pairs=1,
                ),
            )
        with self.assertRaisesRegex(FreezeDataError, "category distribution"):
            freeze_data._validate_expected_distribution(
                Split.TRAIN,
                summary,
                SplitExpectation(
                    records=9,
                    decisions={"challenge": 6, "comply": 3},
                    minimal_pairs=0,
                    categories={
                        **summary.categories,
                        CHALLENGE_CATEGORIES[0]: 2,
                    },
                ),
            )

    def test_requires_two_accepted_reviewers_and_finished_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            path = root / "data" / "metadata" / "validation.jsonl"
            metadata = self._read_jsonl(path)
            metadata[0]["review"]["independent"][1]["status"] = "revise"
            self._write_jsonl(path, metadata)

            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "two distinct accepted"),
            ):
                freeze_dataset(root)

            metadata[0]["review"]["independent"][1]["status"] = "accepted"
            metadata[0]["review"]["adjudication"] = "not_started"
            self._write_jsonl(path, metadata)
            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "adjudication"),
            ):
                freeze_dataset(root)

    def test_rejects_exact_and_near_duplicates_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            train_path = root / "data" / "splits" / "train.jsonl"
            validation_path = root / "data" / "splits" / "validation.jsonl"
            train = self._read_jsonl(train_path)
            validation = self._read_jsonl(validation_path)
            validation[0]["input"] = train[0]["input"].upper()
            self._write_jsonl(validation_path, validation)

            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "exact duplicate"),
            ):
                freeze_dataset(root)

            validation[0]["input"] = train[0]["input"] + " Please."
            self._write_jsonl(validation_path, validation)
            with (
                patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS),
                self.assertRaisesRegex(FreezeDataError, "near-duplicate"),
            ):
                freeze_dataset(root)

    def test_allows_similar_minimal_pair_within_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            examples_path = root / "data" / "splits" / "train.jsonl"
            metadata_path = root / "data" / "metadata" / "train.jsonl"
            examples = self._read_jsonl(examples_path)
            metadata = self._read_jsonl(metadata_path)
            examples[0]["input"] = "Send the approved report after review is complete."
            examples[6]["input"] = "Send the approved report; review is complete."
            metadata[0]["domain"] = "ai_agent_authorization"
            metadata[6]["domain"] = "ai_agent_authorization"
            metadata[0]["minimal_pair_id"] = "train-pair-001"
            metadata[6]["minimal_pair_id"] = "train-pair-001"
            self._write_jsonl(examples_path, examples)
            self._write_jsonl(metadata_path, metadata)

            pair_expectations = dict(SMALL_EXPECTATIONS)
            pair_expectations[Split.TRAIN] = SplitExpectation(
                records=9,
                decisions={"challenge": 6, "comply": 3},
                minimal_pairs=1,
            )
            with patch.object(freeze_data, "EXPECTED_SPLITS", pair_expectations):
                manifest = freeze_dataset(root)

            self.assertEqual(manifest["splits"]["train"]["summary"]["minimal_pairs"], 1)

    def test_check_detects_changed_hashed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_experiment(root)
            with patch.object(freeze_data, "EXPECTED_SPLITS", SMALL_EXPECTATIONS):
                freeze_dataset(root)
                (root / "prompts" / "basic.txt").write_text(
                    "changed prompt\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(FreezeDataError, "does not match"):
                    freeze_dataset(root, check=True)

    @classmethod
    def _create_experiment(cls, root: Path) -> None:
        (root / "data" / "splits").mkdir(parents=True)
        (root / "data" / "metadata").mkdir(parents=True)
        (root / "prompts").mkdir()
        (root / "schema").mkdir()
        (root / "prompts" / "basic.txt").write_text("basic prompt\n", encoding="utf-8")
        (root / "prompts" / "strong.txt").write_text(
            "strong prompt\n", encoding="utf-8"
        )
        (root / "schema" / "decision-response-v1.schema.json").write_text(
            '{"type":"object"}\n',
            encoding="utf-8",
        )
        opening = {
            Split.TRAIN: "Prepare a training scenario about",
            Split.VALIDATION: "Evaluate a separate workplace case concerning",
            Split.TEST: "Consider an untouched request involving",
        }
        categories = CHALLENGE_CATEGORIES + COMPLY_CATEGORIES
        for split in SMALL_EXPECTATIONS:
            examples: list[dict[str, Any]] = []
            metadata: list[dict[str, Any]] = []
            for index, category in enumerate(categories):
                decision = "challenge" if index < 6 else "comply"
                record_id = f"{split.value}-{index + 1:04d}"
                target = cls._target(decision, index)
                examples.append(
                    {
                        "id": record_id,
                        "input": (
                            f"{opening[split]} {category.replace('_', ' ')} "
                            f"with distinctive detail number {index + 1}."
                        ),
                        "target": target,
                    }
                )
                metadata.append(
                    {
                        "schema_version": 1,
                        "id": record_id,
                        "split": split.value,
                        "domain": DOMAINS[index % len(DOMAINS)],
                        "category": category,
                        "gold_decision": decision,
                        "minimal_pair_id": None,
                        "generation": {
                            "method": "agent",
                            "generator": "test-generator",
                            "model": None,
                        },
                        "review": {
                            "independent": [
                                {"reviewer": "logic-reviewer", "status": "accepted"},
                                {"reviewer": "quality-reviewer", "status": "accepted"},
                            ],
                            "adjudication": "not_needed",
                            "human_audit": "not_selected",
                        },
                    }
                )
            cls._write_jsonl(
                root / "data" / "splits" / f"{split.value}.jsonl",
                examples,
            )
            cls._write_jsonl(
                root / "data" / "metadata" / f"{split.value}.jsonl",
                metadata,
            )

    @staticmethod
    def _target(decision: str, index: int) -> dict[str, object]:
        if decision == "challenge":
            return {
                "decision": decision,
                "issue": f"Material issue number {index + 1} needs attention.",
                "message": f"This request has consequence number {index + 1}.",
                "question": None,
                "suggested_next_step": f"Resolve issue number {index + 1} first.",
            }
        return {
            "decision": decision,
            "issue": None,
            "message": f"Proceed with reasonable request number {index + 1}.",
            "question": None,
            "suggested_next_step": None,
        }

    @classmethod
    def _remove_record(cls, root: Path, split: Split, index: int) -> None:
        for subdirectory in ("splits", "metadata"):
            path = root / "data" / subdirectory / f"{split.value}.jsonl"
            records = cls._read_jsonl(path)
            records.pop(index)
            cls._write_jsonl(path, records)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, separators=(",", ":")) + "\n" for record in records
            ),
            encoding="utf-8",
        )


class ProductionExpectationsTest(unittest.TestCase):
    def test_production_expectations_are_fixed(self) -> None:
        self.assertEqual(
            freeze_data.EXPECTED_SPLITS,
            {
                Split.TRAIN: SplitExpectation(
                    400,
                    {"challenge": 240, "comply": 160},
                    80,
                    freeze_data.TRAIN_CATEGORY_DISTRIBUTION,
                ),
                Split.VALIDATION: SplitExpectation(
                    50,
                    {"challenge": 30, "comply": 20},
                    10,
                ),
                Split.TEST: SplitExpectation(
                    100,
                    {"challenge": 50, "comply": 50},
                    20,
                ),
            },
        )
        self.assertEqual(
            {category.value for category in Category},
            set(CHALLENGE_CATEGORIES + COMPLY_CATEGORIES),
        )

    def test_cli_passes_check_mode_and_prints_manifest(self) -> None:
        output = io.StringIO()
        argv = [
            "disagree-freeze-data",
            "--experiment-root",
            "/tmp/example-experiment",
            "--check",
        ]
        expected = {"schema_version": 1}

        with (
            patch.object(sys, "argv", argv),
            patch.object(
                freeze_data,
                "freeze_dataset",
                return_value=expected,
            ) as mocked_freeze,
            redirect_stdout(output),
        ):
            result = freeze_data.main()

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), expected)
        mocked_freeze.assert_called_once_with(
            Path("/tmp/example-experiment"),
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

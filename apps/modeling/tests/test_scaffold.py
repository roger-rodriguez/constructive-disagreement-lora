from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import disagree_contracts
import disagree_modeling
from disagree_contracts.schemas import Split
from disagree_modeling.data import DatasetValidationError, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class ScaffoldTest(unittest.TestCase):
    def test_local_packages_are_importable(self) -> None:
        self.assertEqual(disagree_modeling.__version__, "0.0.0")
        self.assertEqual(disagree_contracts.__version__, "0.0.0")

    def test_tiny_fixture_satisfies_dataset_contract(self) -> None:
        summary = validate_dataset(
            FIXTURES / "tiny-examples.jsonl",
            FIXTURES / "tiny-metadata.jsonl",
            expected_split=Split.FIXTURE,
        )

        self.assertEqual(summary.records, 10)
        self.assertEqual(summary.minimal_pairs, 3)
        self.assertEqual(summary.decisions, {"challenge": 6, "comply": 4})
        self.assertEqual(len(summary.domains), 7)
        self.assertEqual(summary.as_dict()["records"], 10)

    def test_rejects_mismatched_sidecar_ids(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        metadata[-1]["id"] = "fixture-999"

        with self.assertRaisesRegex(DatasetValidationError, "missing metadata"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_incomplete_minimal_pair(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        metadata[1]["minimal_pair_id"] = None

        with self.assertRaisesRegex(DatasetValidationError, "exactly two records"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_category_decision_mismatch(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        metadata[-1]["gold_decision"] = "challenge"

        with self.assertRaisesRegex(DatasetValidationError, "requires comply"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_wrong_expected_split(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")

        with self.assertRaisesRegex(DatasetValidationError, "expected split train"):
            self._validate_temporary(
                conversations,
                metadata,
                expected_split=Split.TRAIN,
            )

    def test_rejects_duplicate_ids(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        conversations[-1]["id"] = conversations[0]["id"]

        with self.assertRaisesRegex(DatasetValidationError, "duplicate id"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_duplicate_reviewers(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        review = metadata[0]["review"]
        review["independent"] = [
            {"reviewer": "reviewer-1", "status": "accepted"},
            {"reviewer": "reviewer-1", "status": "accepted"},
        ]

        with self.assertRaisesRegex(DatasetValidationError, "duplicate independent"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_duplicate_user_requests(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        conversations[-1]["input"] = conversations[0]["input"].upper()

        with self.assertRaisesRegex(DatasetValidationError, "duplicates user request"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_minimal_pair_with_same_decision(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        metadata[1]["category"] = "missing_material_constraint"
        metadata[1]["gold_decision"] = "challenge"
        conversations[1]["target"] = {
            "decision": "challenge",
            "issue": "A material constraint is missing.",
            "message": "The request needs one decision before proceeding.",
            "question": "Which constraint should govern the work?",
            "suggested_next_step": "Set the governing constraint first.",
        }

        with self.assertRaisesRegex(
            DatasetValidationError,
            "must contain challenge and comply",
        ):
            self._validate_temporary(conversations, metadata)

    def test_rejects_minimal_pair_across_domains(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        metadata[1]["domain"] = "project_planning"

        with self.assertRaisesRegex(DatasetValidationError, "must share a domain"):
            self._validate_temporary(conversations, metadata)

    def test_rejects_target_decision_that_disagrees_with_metadata(self) -> None:
        conversations = self._read_fixture("tiny-examples.jsonl")
        metadata = self._read_fixture("tiny-metadata.jsonl")
        conversations[0]["target"]["decision"] = "comply"
        conversations[0]["target"]["issue"] = None

        with self.assertRaisesRegex(DatasetValidationError, "target decision"):
            self._validate_temporary(conversations, metadata)

    @staticmethod
    def _read_fixture(name: str) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def _validate_temporary(
        conversations: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        *,
        expected_split: Split = Split.FIXTURE,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversations_path = root / "examples.jsonl"
            metadata_path = root / "metadata.jsonl"
            conversations_path.write_text(
                "\n".join(json.dumps(record) for record in conversations) + "\n",
                encoding="utf-8",
            )
            metadata_path.write_text(
                "\n".join(json.dumps(record) for record in metadata) + "\n",
                encoding="utf-8",
            )
            validate_dataset(
                conversations_path,
                metadata_path,
                expected_split=expected_split,
            )


if __name__ == "__main__":
    unittest.main()

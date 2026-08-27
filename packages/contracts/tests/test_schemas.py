from __future__ import annotations

import unittest

from disagree_contracts.identifiers import validate_identifier
from disagree_contracts.schemas import (
    Category,
    ConversationRecord,
    DecisionRecord,
    DecisionResponse,
    Domain,
    GoldDecision,
    MetadataRecord,
    SchemaError,
    Split,
)


class IdentifierTest(unittest.TestCase):
    def test_accepts_segmented_lowercase_identifier(self) -> None:
        self.assertEqual(
            validate_identifier("train-0001", field="id"),
            "train-0001",
        )

    def test_rejects_unstable_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase alphanumeric"):
            validate_identifier("Train_1", field="id")

    def test_rejects_non_string_identifier(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            validate_identifier(1, field="id")


class ConversationRecordTest(unittest.TestCase):
    def test_accepts_optional_system_and_multiple_turns(self) -> None:
        record = ConversationRecord.from_mapping(
            {
                "id": "fixture-001",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Question one."},
                    {"role": "assistant", "content": "Answer one."},
                    {"role": "user", "content": "Question two."},
                    {"role": "assistant", "content": "Answer two."},
                ],
            }
        )

        self.assertEqual(record.id, "fixture-001")
        self.assertEqual(len(record.messages), 5)

    def test_rejects_invalid_role_order(self) -> None:
        with self.assertRaisesRegex(SchemaError, "alternate user and assistant"):
            ConversationRecord.from_mapping(
                {
                    "id": "fixture-001",
                    "messages": [
                        {"role": "assistant", "content": "Out of order."},
                        {"role": "user", "content": "Still out of order."},
                    ],
                }
            )

    def test_rejects_invalid_identifier_as_schema_error(self) -> None:
        with self.assertRaisesRegex(SchemaError, "lowercase alphanumeric"):
            ConversationRecord.from_mapping(
                {
                    "id": "Fixture_1",
                    "messages": [
                        {"role": "user", "content": "Question."},
                        {"role": "assistant", "content": "Answer."},
                    ],
                }
            )

    def test_rejects_empty_conversation(self) -> None:
        with self.assertRaisesRegex(SchemaError, "must not be empty"):
            ConversationRecord.from_mapping({"id": "fixture-001", "messages": []})

    def test_rejects_unpaired_conversation(self) -> None:
        with self.assertRaisesRegex(SchemaError, "user/assistant pairs"):
            ConversationRecord.from_mapping(
                {
                    "id": "fixture-001",
                    "messages": [{"role": "user", "content": "Question."}],
                }
            )

    def test_rejects_non_array_messages(self) -> None:
        with self.assertRaisesRegex(SchemaError, "must be an array"):
            ConversationRecord.from_mapping(
                {"id": "fixture-001", "messages": "not-an-array"}
            )

    def test_rejects_blank_message_content(self) -> None:
        with self.assertRaisesRegex(SchemaError, "non-empty string"):
            ConversationRecord.from_mapping(
                {
                    "id": "fixture-001",
                    "messages": [
                        {"role": "user", "content": " "},
                        {"role": "assistant", "content": "Answer."},
                    ],
                }
            )

    def test_rejects_message_whitespace(self) -> None:
        with self.assertRaisesRegex(SchemaError, "leading or trailing"):
            ConversationRecord.from_mapping(
                {
                    "id": "fixture-001",
                    "messages": [
                        {"role": "user", "content": " Question."},
                        {"role": "assistant", "content": "Answer."},
                    ],
                }
            )


class MetadataRecordTest(unittest.TestCase):
    @staticmethod
    def _valid_mapping() -> dict[str, object]:
        return {
            "schema_version": 1,
            "id": "fixture-001",
            "split": "fixture",
            "domain": "engineering_estimates",
            "category": "implausible_estimate_or_schedule",
            "gold_decision": "challenge",
            "minimal_pair_id": "fixture-pair-001",
            "generation": {
                "method": "agent",
                "generator": "codex-agent",
                "model": None,
            },
            "review": {
                "independent": [],
                "adjudication": "not_started",
                "human_audit": "not_selected",
            },
        }

    def test_parses_complete_sidecar_contract(self) -> None:
        record = MetadataRecord.from_mapping(self._valid_mapping())

        self.assertEqual(record.split, Split.FIXTURE)
        self.assertEqual(record.domain, Domain.ENGINEERING_ESTIMATES)
        self.assertEqual(record.category, Category.IMPLAUSIBLE_ESTIMATE)
        self.assertEqual(record.gold_decision, GoldDecision.CHALLENGE)

    def test_parses_review_only_pilot_split(self) -> None:
        value = self._valid_mapping()
        value["split"] = "pilot"

        record = MetadataRecord.from_mapping(value)

        self.assertEqual(record.split, Split.PILOT)

    def test_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(SchemaError, "unknown keys"):
            MetadataRecord.from_mapping(
                {
                    "schema_version": 1,
                    "id": "fixture-001",
                    "split": "fixture",
                    "domain": "engineering_estimates",
                    "category": "implausible_estimate_or_schedule",
                    "gold_decision": "challenge",
                    "minimal_pair_id": None,
                    "generation": {
                        "method": "agent",
                        "generator": "codex-agent",
                        "model": None,
                    },
                    "review": {
                        "independent": [],
                        "adjudication": "not_started",
                        "human_audit": "not_selected",
                    },
                    "leaked_label": "unexpected",
                }
            )

    def test_rejects_wrong_schema_version(self) -> None:
        value = self._valid_mapping()
        value["schema_version"] = 2

        with self.assertRaisesRegex(SchemaError, "must equal 1"):
            MetadataRecord.from_mapping(value)

    def test_rejects_invalid_pair_identifier(self) -> None:
        value = self._valid_mapping()
        value["minimal_pair_id"] = "PAIR_1"

        with self.assertRaisesRegex(SchemaError, "lowercase alphanumeric"):
            MetadataRecord.from_mapping(value)

    def test_rejects_non_string_model(self) -> None:
        value = self._valid_mapping()
        generation = value["generation"]
        assert isinstance(generation, dict)
        generation["model"] = 42

        with self.assertRaisesRegex(SchemaError, "string or null"):
            MetadataRecord.from_mapping(value)

    def test_parses_independent_review(self) -> None:
        value = self._valid_mapping()
        review = value["review"]
        assert isinstance(review, dict)
        review["independent"] = [{"reviewer": "reviewer-1", "status": "accepted"}]

        record = MetadataRecord.from_mapping(value)

        self.assertEqual(record.review.independent[0].reviewer, "reviewer-1")

    def test_rejects_invalid_enum(self) -> None:
        value = self._valid_mapping()
        value["split"] = "training"

        with self.assertRaisesRegex(SchemaError, "must be one of"):
            MetadataRecord.from_mapping(value)

    def test_rejects_non_object(self) -> None:
        with self.assertRaisesRegex(SchemaError, "must be an object"):
            MetadataRecord.from_mapping([])

    def test_rejects_missing_fields(self) -> None:
        value = self._valid_mapping()
        del value["review"]

        with self.assertRaisesRegex(SchemaError, "is missing keys"):
            MetadataRecord.from_mapping(value)


class DecisionRecordTest(unittest.TestCase):
    @staticmethod
    def _challenge_target() -> dict[str, object]:
        return {
            "decision": "challenge",
            "issue": "The requirements conflict.",
            "message": "Those requirements pull in opposite directions.",
            "question": "Which guarantee matters more?",
            "suggested_next_step": "Choose one guarantee as the priority.",
        }

    def test_parses_structured_challenge(self) -> None:
        record = DecisionRecord.from_mapping(
            {
                "id": "pilot-001",
                "input": "Make both guarantees absolute.",
                "target": self._challenge_target(),
            }
        )

        self.assertEqual(record.target.decision, GoldDecision.CHALLENGE)
        self.assertEqual(record.target.question, "Which guarantee matters more?")

    def test_parses_structured_comply_response(self) -> None:
        response = DecisionResponse.from_mapping(
            {
                "decision": "comply",
                "issue": None,
                "message": "I’ll prepare the checklist.",
                "question": None,
                "suggested_next_step": None,
            }
        )

        self.assertEqual(response.decision, GoldDecision.COMPLY)
        self.assertIsNone(response.issue)

    def test_rejects_unclear_as_model_target(self) -> None:
        target = self._challenge_target()
        target["decision"] = "unclear"

        with self.assertRaisesRegex(SchemaError, "review-only"):
            DecisionResponse.from_mapping(target)

    def test_rejects_challenge_without_issue(self) -> None:
        target = self._challenge_target()
        target["issue"] = None

        with self.assertRaisesRegex(SchemaError, "requires a non-empty issue"):
            DecisionResponse.from_mapping(target)

    def test_rejects_challenge_without_next_step(self) -> None:
        target = self._challenge_target()
        target["suggested_next_step"] = None

        with self.assertRaisesRegex(SchemaError, "suggested_next_step"):
            DecisionResponse.from_mapping(target)

    def test_rejects_comply_with_issue(self) -> None:
        target = self._challenge_target()
        target["decision"] = "comply"

        with self.assertRaisesRegex(SchemaError, "requires issue to be null"):
            DecisionResponse.from_mapping(target)

    def test_rejects_multiple_questions(self) -> None:
        target = self._challenge_target()
        target["question"] = "Which guarantee matters? What is the deadline?"

        with self.assertRaisesRegex(SchemaError, "exactly one question"):
            DecisionResponse.from_mapping(target)

    def test_rejects_unknown_target_field(self) -> None:
        target = self._challenge_target()
        target["confidence"] = 0.9

        with self.assertRaisesRegex(SchemaError, "unknown keys"):
            DecisionResponse.from_mapping(target)


if __name__ == "__main__":
    unittest.main()

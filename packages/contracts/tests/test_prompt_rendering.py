from __future__ import annotations

import unittest

from disagree_contracts.prompt_rendering import (
    IGNORE_INDEX,
    assistant_only_labels,
    decision_messages_for_template,
    parse_decision_output,
    render_decision_training_conversation,
    render_training_conversation,
)
from disagree_contracts.schemas import ConversationRecord, DecisionRecord


class FakeTokenizer:
    def __init__(self) -> None:
        self.conversation: list[dict[str, str]] | None = None
        self.tokenize: bool | None = None
        self.add_generation_prompt: bool | None = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.conversation = conversation
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "<user>Test request</user><assistant>Test response</assistant>"


class EmptyTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        return ""


class PromptRenderingTest(unittest.TestCase):
    def test_renders_complete_training_conversation(self) -> None:
        record = ConversationRecord.from_mapping(
            {
                "id": "fixture-001",
                "messages": [
                    {"role": "user", "content": "Test request"},
                    {"role": "assistant", "content": "Test response"},
                ],
            }
        )
        tokenizer = FakeTokenizer()

        rendered = render_training_conversation(record, tokenizer)

        self.assertEqual(
            rendered,
            "<user>Test request</user><assistant>Test response</assistant>",
        )
        self.assertEqual(
            tokenizer.conversation,
            [
                {"role": "user", "content": "Test request"},
                {"role": "assistant", "content": "Test response"},
            ],
        )
        self.assertFalse(tokenizer.tokenize)
        self.assertFalse(tokenizer.add_generation_prompt)

    def test_renders_decision_target_as_canonical_json(self) -> None:
        record = DecisionRecord.from_mapping(
            {
                "id": "pilot-001",
                "input": "Make both guarantees absolute.",
                "target": {
                    "decision": "challenge",
                    "issue": "The requirements conflict.",
                    "message": "Those requirements pull in opposite directions.",
                    "question": "Which guarantee matters more?",
                    "suggested_next_step": "Choose one guarantee as the priority.",
                },
            }
        )

        messages = decision_messages_for_template(record)

        self.assertEqual(messages[0], {"role": "user", "content": record.input})
        self.assertEqual(
            messages[1]["content"],
            '{"decision":"challenge","issue":"The requirements conflict.",'
            '"message":"Those requirements pull in opposite directions.",'
            '"question":"Which guarantee matters more?",'
            '"suggested_next_step":"Choose one guarantee as the priority."}',
        )

    def test_renders_structured_decision_through_chat_template(self) -> None:
        record = DecisionRecord.from_mapping(
            {
                "id": "pilot-002",
                "input": "Prepare the approved checklist.",
                "target": {
                    "decision": "comply",
                    "issue": None,
                    "message": "I’ll prepare the checklist.",
                    "question": None,
                    "suggested_next_step": None,
                },
            }
        )
        tokenizer = FakeTokenizer()

        rendered = render_decision_training_conversation(record, tokenizer)

        self.assertEqual(
            rendered,
            "<user>Test request</user><assistant>Test response</assistant>",
        )
        assert tokenizer.conversation is not None
        self.assertEqual(tokenizer.conversation[0]["content"], record.input)
        self.assertIn('"decision":"comply"', tokenizer.conversation[1]["content"])

    def test_parses_valid_model_decision_output(self) -> None:
        response = parse_decision_output(
            '{"decision":"comply","issue":null,'
            '"message":"I’ll prepare the checklist.","question":null,'
            '"suggested_next_step":null}'
        )

        self.assertEqual(response.decision.value, "comply")

    def test_rejects_invalid_model_decision_json_without_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_decision_output("```json\n{}\n```")

    def test_masks_every_non_assistant_token(self) -> None:
        labels = assistant_only_labels(
            [10, 11, 12, 13, 14],
            [False, False, False, True, True],
        )

        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14])

    def test_rejects_mask_without_assistant_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one token"):
            assistant_only_labels([10, 11], [False, False])

    def test_rejects_invalid_template_output(self) -> None:
        record = ConversationRecord.from_mapping(
            {
                "id": "fixture-001",
                "messages": [
                    {"role": "user", "content": "Test request"},
                    {"role": "assistant", "content": "Test response"},
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "non-empty text"):
            render_training_conversation(record, EmptyTokenizer())

    def test_rejects_mask_with_different_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal length"):
            assistant_only_labels([10, 11], [True])

    def test_rejects_empty_token_sequence(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            assistant_only_labels([], [])


if __name__ == "__main__":
    unittest.main()

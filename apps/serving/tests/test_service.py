from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from disagree_serving.service import (
    inference_messages,
    read_verified_prompt,
    response_payload,
    validate_user_input,
)


class ServiceTest(unittest.TestCase):
    def test_reads_prompt_with_matching_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("  frozen prompt\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            self.assertEqual(read_verified_prompt(path, digest), "frozen prompt")

    def test_rejects_prompt_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("prompt", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash does not match"):
                read_verified_prompt(path, "0" * 64)

    def test_rejects_empty_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("  \n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "must not be empty"):
                read_verified_prompt(path, digest)

    def test_validates_input_without_rewriting_it(self) -> None:
        self.assertEqual(
            validate_user_input("  keep spacing  ", 20), "  keep spacing  "
        )

    def test_rejects_non_text_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be text"):
            validate_user_input(123, 20)

    def test_rejects_blank_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty or whitespace"):
            validate_user_input(" \n", 20)

    def test_rejects_input_over_character_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 3"):
            validate_user_input("four", 3)

    def test_builds_evaluation_prompt_shape(self) -> None:
        self.assertEqual(
            inference_messages("system", "request"),
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "request"},
            ],
        )

    def test_returns_only_validated_five_field_response(self) -> None:
        raw = json.dumps(
            {
                "decision": "challenge",
                "issue": "The guarantees conflict.",
                "message": "Pick which guarantee matters more.",
                "question": "Which guarantee is the priority?",
                "suggested_next_step": "Choose one guarantee before implementation.",
            }
        )
        self.assertEqual(
            response_payload(raw),
            {
                "decision": "challenge",
                "issue": "The guarantees conflict.",
                "message": "Pick which guarantee matters more.",
                "question": "Which guarantee is the priority?",
                "suggested_next_step": "Choose one guarantee before implementation.",
            },
        )

    def test_rejects_invalid_model_json_without_repair(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            response_payload("not json")


if __name__ == "__main__":
    unittest.main()

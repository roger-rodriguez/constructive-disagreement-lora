from __future__ import annotations

import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from disagree_serving.config import load_config
from disagree_serving.model_runtime import generate_response

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "model.toml")


class FakeTensor:
    def __init__(self, token_count: int) -> None:
        self.shape = (1, token_count)


class FakeEncoding(dict[str, FakeTensor]):
    def to(self, device: str) -> FakeEncoding:
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self, token_count: int, raw_output: str) -> None:
        self.token_count = token_count
        self.raw_output = raw_output
        self.messages: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.messages = messages
        if tokenize or not add_generation_prompt:
            raise AssertionError("unexpected chat-template options")
        return "rendered prompt"

    def __call__(self, rendered: str, *, return_tensors: str) -> FakeEncoding:
        if rendered != "rendered prompt" or return_tensors != "pt":
            raise AssertionError("unexpected tokenizer call")
        return FakeEncoding(
            input_ids=FakeTensor(self.token_count),
            attention_mask=FakeTensor(self.token_count),
        )

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        if tokens != [999] or not skip_special_tokens:
            raise AssertionError("unexpected decode call")
        return self.raw_output


class FakeModel:
    device = "cuda"

    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.generation_arguments: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generation_arguments = kwargs
        return [list(range(self.prompt_tokens)) + [999]]


class ModelRuntimeTest(unittest.TestCase):
    def test_generates_from_batch_encoding_and_parses_json(self) -> None:
        raw = json.dumps(
            {
                "decision": "comply",
                "issue": None,
                "message": "This request is feasible as written.",
                "question": None,
                "suggested_next_step": None,
            }
        )
        tokenizer = FakeTokenizer(12, raw)
        model = FakeModel(12)
        torch = SimpleNamespace(inference_mode=nullcontext)

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            system_prompt="system",
            user_input="request",
            config=CONFIG,
        )

        self.assertEqual(result["decision"], "comply")
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "request"},
            ],
        )
        arguments = model.generation_arguments
        self.assertIsNotNone(arguments)
        if arguments is None:
            self.fail("model.generate was not called")
        self.assertEqual(arguments["max_new_tokens"], 256)
        self.assertFalse(arguments["do_sample"])
        self.assertTrue(arguments["use_cache"])

    def test_rejects_rendered_input_over_token_limit(self) -> None:
        tokenizer = FakeTokenizer(769, "unused")
        model = FakeModel(769)
        torch = SimpleNamespace(inference_mode=nullcontext)

        with self.assertRaisesRegex(ValueError, "exceeds 768 tokens"):
            generate_response(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                system_prompt="system",
                user_input="request",
                config=CONFIG,
            )

    def test_rejects_invalid_generated_json(self) -> None:
        tokenizer = FakeTokenizer(12, "not json")
        model = FakeModel(12)
        torch = SimpleNamespace(inference_mode=nullcontext)

        with self.assertRaisesRegex(RuntimeError, "invalid structured JSON"):
            generate_response(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                system_prompt="system",
                user_input="request",
                config=CONFIG,
            )


if __name__ == "__main__":
    unittest.main()

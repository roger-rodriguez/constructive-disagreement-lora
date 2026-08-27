"""Contracts for model-specific chat rendering and assistant-only labels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from disagree_contracts.schemas import (
    ConversationRecord,
    DecisionRecord,
    DecisionResponse,
)

IGNORE_INDEX = -100


class ChatTemplateTokenizer(Protocol):
    """Small portion of the tokenizer interface used by modeling and serving."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def messages_for_template(record: ConversationRecord) -> list[dict[str, str]]:
    """Convert the immutable public record into a tokenizer-friendly value."""
    return [
        {"role": message.role.value, "content": message.content}
        for message in record.messages
    ]


def decision_messages_for_template(record: DecisionRecord) -> list[dict[str, str]]:
    """Render a structured decision target as canonical compact JSON."""
    target = json.dumps(
        record.target.as_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "user", "content": record.input},
        {"role": "assistant", "content": target},
    ]


def parse_decision_output(value: str) -> DecisionResponse:
    """Parse and validate one raw model response without repairing it."""
    if not isinstance(value, str) or not value:
        raise ValueError("model decision output must be non-empty text")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("model decision output must be valid JSON") from error
    return DecisionResponse.from_mapping(parsed)


def render_training_conversation(
    record: ConversationRecord, tokenizer: ChatTemplateTokenizer
) -> str:
    """Render a complete training example without an extra generation prompt."""
    rendered = tokenizer.apply_chat_template(
        messages_for_template(record),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template must return non-empty text")
    return rendered


def render_decision_training_conversation(
    record: DecisionRecord,
    tokenizer: ChatTemplateTokenizer,
) -> str:
    """Render one structured decision record through the model chat template."""
    rendered = tokenizer.apply_chat_template(
        decision_messages_for_template(record),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template must return non-empty text")
    return rendered


def assistant_only_labels(
    input_ids: Sequence[int],
    assistant_mask: Sequence[bool],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Mask every non-assistant token from supervised training loss."""
    if len(input_ids) != len(assistant_mask):
        raise ValueError("input_ids and assistant_mask must have equal length")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if not any(assistant_mask):
        raise ValueError("assistant_mask must select at least one token")
    return [
        token_id if is_assistant else ignore_index
        for token_id, is_assistant in zip(input_ids, assistant_mask, strict=True)
    ]

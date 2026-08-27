"""Dependency-free schemas for structured decisions and chat rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from disagree_contracts.identifiers import validate_identifier

SCHEMA_VERSION = 1


class SchemaError(ValueError):
    """Raised when a public dataset record violates its schema."""


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Split(StrEnum):
    FIXTURE = "fixture"
    PILOT = "pilot"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class GoldDecision(StrEnum):
    CHALLENGE = "challenge"
    COMPLY = "comply"
    UNCLEAR = "unclear"


class Domain(StrEnum):
    PRODUCT_REQUIREMENTS = "product_requirements"
    ENGINEERING_ESTIMATES = "engineering_estimates"
    PROJECT_PLANNING = "project_planning"
    CUSTOMER_REQUESTS = "customer_requests"
    OPERATIONS_INCIDENT_RESPONSE = "operations_incident_response"
    HIRING_TEAM_MANAGEMENT = "hiring_team_management"
    AI_AGENT_AUTHORIZATION = "ai_agent_authorization"


class Category(StrEnum):
    UNSUPPORTED_ASSUMPTION = "unsupported_assumption_or_missing_evidence"
    INTERNAL_CONTRADICTION = "internal_contradiction"
    MISSING_CONSTRAINT = "missing_material_constraint"
    IMPLAUSIBLE_ESTIMATE = "implausible_estimate_or_schedule"
    AUTHORIZATION_RISK = "authorization_privacy_security_or_operational_risk"
    MATERIAL_HARM = "material_harm_or_unethical_request"
    STRAIGHTFORWARD = "straightforward_reasonable_request"
    CONSTRAINED = "constrained_but_reasonable_request"
    SAFE_NEAR_NEIGHBOR = "safe_near_neighbor_to_flawed_request"


class GenerationMethod(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    MIXED = "mixed"


class ReviewStatus(StrEnum):
    ACCEPTED = "accepted"
    REVISE = "revise"
    REJECTED = "rejected"


class AdjudicationStatus(StrEnum):
    NOT_STARTED = "not_started"
    NOT_NEEDED = "not_needed"
    PENDING = "pending"
    RESOLVED = "resolved"


class HumanAuditStatus(StrEnum):
    NOT_SELECTED = "not_selected"
    PENDING = "pending"
    VERIFIED = "verified"
    REVISED = "revised"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    @classmethod
    def from_mapping(cls, value: object, *, context: str) -> Message:
        data = _mapping(value, context=context)
        _expect_keys(data, required={"role", "content"}, context=context)
        role = _enum(Role, data["role"], field=f"{context}.role")
        content = _nonempty_string(data["content"], field=f"{context}.content")
        return cls(role=role, content=content)


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: str
    messages: tuple[Message, ...]

    @classmethod
    def from_mapping(cls, value: object) -> ConversationRecord:
        data = _mapping(value, context="conversation")
        _expect_keys(data, required={"id", "messages"}, context="conversation")
        try:
            scenario_id = validate_identifier(data["id"], field="conversation.id")
        except (TypeError, ValueError) as error:
            raise SchemaError(str(error)) from error
        raw_messages = _sequence(data["messages"], field="conversation.messages")
        messages = tuple(
            Message.from_mapping(message, context=f"conversation.messages[{index}]")
            for index, message in enumerate(raw_messages)
        )
        _validate_role_order(messages)
        return cls(id=scenario_id, messages=messages)


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    decision: GoldDecision
    issue: str | None
    message: str
    question: str | None
    suggested_next_step: str | None

    @classmethod
    def from_mapping(cls, value: object) -> DecisionResponse:
        data = _mapping(value, context="decision.target")
        _expect_keys(
            data,
            required={
                "decision",
                "issue",
                "message",
                "question",
                "suggested_next_step",
            },
            context="decision.target",
        )
        decision = _enum(
            GoldDecision,
            data["decision"],
            field="decision.target.decision",
        )
        if decision is GoldDecision.UNCLEAR:
            raise SchemaError(
                "decision.target.decision must be challenge or comply; "
                "unclear is review-only"
            )
        issue = _optional_nonempty_string(
            data["issue"],
            field="decision.target.issue",
        )
        question = _optional_nonempty_string(
            data["question"],
            field="decision.target.question",
        )
        suggested_next_step = _optional_nonempty_string(
            data["suggested_next_step"],
            field="decision.target.suggested_next_step",
        )
        if question is not None and (
            not question.endswith("?") or question.count("?") != 1
        ):
            raise SchemaError(
                "decision.target.question must contain exactly one question"
            )
        if decision is GoldDecision.CHALLENGE:
            if issue is None:
                raise SchemaError("challenge target requires a non-empty issue")
            if suggested_next_step is None:
                raise SchemaError(
                    "challenge target requires a non-empty suggested_next_step"
                )
        elif issue is not None:
            raise SchemaError("comply target requires issue to be null")
        return cls(
            decision=decision,
            issue=issue,
            message=_nonempty_string(
                data["message"],
                field="decision.target.message",
            ),
            question=question,
            suggested_next_step=suggested_next_step,
        )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "decision": self.decision.value,
            "issue": self.issue,
            "message": self.message,
            "question": self.question,
            "suggested_next_step": self.suggested_next_step,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    id: str
    input: str
    target: DecisionResponse

    @classmethod
    def from_mapping(cls, value: object) -> DecisionRecord:
        data = _mapping(value, context="decision")
        _expect_keys(
            data,
            required={"id", "input", "target"},
            context="decision",
        )
        try:
            scenario_id = validate_identifier(data["id"], field="decision.id")
        except (TypeError, ValueError) as error:
            raise SchemaError(str(error)) from error
        return cls(
            id=scenario_id,
            input=_nonempty_string(data["input"], field="decision.input"),
            target=DecisionResponse.from_mapping(data["target"]),
        )


@dataclass(frozen=True, slots=True)
class GenerationProvenance:
    method: GenerationMethod
    generator: str
    model: str | None

    @classmethod
    def from_mapping(cls, value: object) -> GenerationProvenance:
        data = _mapping(value, context="metadata.generation")
        _expect_keys(
            data,
            required={"method", "generator", "model"},
            context="metadata.generation",
        )
        model_value = data["model"]
        if model_value is not None and not isinstance(model_value, str):
            raise SchemaError("metadata.generation.model must be a string or null")
        return cls(
            method=_enum(
                GenerationMethod,
                data["method"],
                field="metadata.generation.method",
            ),
            generator=_nonempty_string(
                data["generator"], field="metadata.generation.generator"
            ),
            model=model_value,
        )


@dataclass(frozen=True, slots=True)
class IndependentReview:
    reviewer: str
    status: ReviewStatus

    @classmethod
    def from_mapping(cls, value: object, *, index: int) -> IndependentReview:
        context = f"metadata.review.independent[{index}]"
        data = _mapping(value, context=context)
        _expect_keys(data, required={"reviewer", "status"}, context=context)
        return cls(
            reviewer=_nonempty_string(data["reviewer"], field=f"{context}.reviewer"),
            status=_enum(ReviewStatus, data["status"], field=f"{context}.status"),
        )


@dataclass(frozen=True, slots=True)
class ReviewProvenance:
    independent: tuple[IndependentReview, ...]
    adjudication: AdjudicationStatus
    human_audit: HumanAuditStatus

    @classmethod
    def from_mapping(cls, value: object) -> ReviewProvenance:
        data = _mapping(value, context="metadata.review")
        _expect_keys(
            data,
            required={"independent", "adjudication", "human_audit"},
            context="metadata.review",
        )
        reviews = _sequence(data["independent"], field="metadata.review.independent")
        return cls(
            independent=tuple(
                IndependentReview.from_mapping(review, index=index)
                for index, review in enumerate(reviews)
            ),
            adjudication=_enum(
                AdjudicationStatus,
                data["adjudication"],
                field="metadata.review.adjudication",
            ),
            human_audit=_enum(
                HumanAuditStatus,
                data["human_audit"],
                field="metadata.review.human_audit",
            ),
        )


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    schema_version: int
    id: str
    split: Split
    domain: Domain
    category: Category
    gold_decision: GoldDecision
    minimal_pair_id: str | None
    generation: GenerationProvenance
    review: ReviewProvenance

    @classmethod
    def from_mapping(cls, value: object) -> MetadataRecord:
        data = _mapping(value, context="metadata")
        _expect_keys(
            data,
            required={
                "schema_version",
                "id",
                "split",
                "domain",
                "category",
                "gold_decision",
                "minimal_pair_id",
                "generation",
                "review",
            },
            context="metadata",
        )
        schema_version = data["schema_version"]
        if schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"metadata.schema_version must equal {SCHEMA_VERSION}, got {schema_version!r}"
            )
        try:
            scenario_id = validate_identifier(data["id"], field="metadata.id")
            pair_value = data["minimal_pair_id"]
            pair_id = (
                None
                if pair_value is None
                else validate_identifier(pair_value, field="metadata.minimal_pair_id")
            )
        except (TypeError, ValueError) as error:
            raise SchemaError(str(error)) from error
        return cls(
            schema_version=SCHEMA_VERSION,
            id=scenario_id,
            split=_enum(Split, data["split"], field="metadata.split"),
            domain=_enum(Domain, data["domain"], field="metadata.domain"),
            category=_enum(Category, data["category"], field="metadata.category"),
            gold_decision=_enum(
                GoldDecision, data["gold_decision"], field="metadata.gold_decision"
            ),
            minimal_pair_id=pair_id,
            generation=GenerationProvenance.from_mapping(data["generation"]),
            review=ReviewProvenance.from_mapping(data["review"]),
        )


EnumType = TypeVar("EnumType", bound=StrEnum)


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{context} keys must be strings")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SchemaError(f"{field} must be an array")
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise SchemaError(f"{field} must not have leading or trailing whitespace")
    return value


def _optional_nonempty_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field=field)


def _enum(enum_type: type[EnumType], value: object, *, field: str) -> EnumType:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise SchemaError(f"{field} must be one of: {allowed}") from error


def _expect_keys(data: Mapping[str, Any], *, required: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise SchemaError(f"{context} is missing keys: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{context} has unknown keys: {', '.join(unknown)}")


def _validate_role_order(messages: tuple[Message, ...]) -> None:
    if not messages:
        raise SchemaError("conversation.messages must not be empty")
    start = 1 if messages[0].role is Role.SYSTEM else 0
    conversational = messages[start:]
    if not conversational or len(conversational) % 2 != 0:
        raise SchemaError(
            "conversation.messages must contain user/assistant pairs after an optional system message"
        )
    for index, message in enumerate(conversational):
        expected = Role.USER if index % 2 == 0 else Role.ASSISTANT
        if message.role is not expected:
            raise SchemaError(
                "conversation.messages must alternate user and assistant roles and end with assistant"
            )

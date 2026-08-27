"""Strict validation for structured decision JSONL and sidecar metadata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from disagree_contracts.schemas import (
    Category,
    DecisionRecord,
    GoldDecision,
    MetadataRecord,
    SchemaError,
    Split,
)


class DatasetValidationError(ValueError):
    """Raised when related dataset files violate a cross-record invariant."""


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    records: int
    minimal_pairs: int
    decisions: Mapping[str, int]
    domains: Mapping[str, int]
    categories: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "minimal_pairs": self.minimal_pairs,
            "decisions": dict(self.decisions),
            "domains": dict(self.domains),
            "categories": dict(self.categories),
        }


CHALLENGE_CATEGORIES = frozenset(
    {
        Category.UNSUPPORTED_ASSUMPTION,
        Category.INTERNAL_CONTRADICTION,
        Category.MISSING_CONSTRAINT,
        Category.IMPLAUSIBLE_ESTIMATE,
        Category.AUTHORIZATION_RISK,
        Category.MATERIAL_HARM,
    }
)
COMPLY_CATEGORIES = frozenset(
    {
        Category.STRAIGHTFORWARD,
        Category.CONSTRAINED,
        Category.SAFE_NEAR_NEIGHBOR,
    }
)


def validate_dataset(
    records_path: Path,
    metadata_path: Path,
    *,
    expected_split: Split | None = None,
) -> ValidationSummary:
    """Validate structured decision records and their metadata as one contract."""
    records = _parse_jsonl(
        records_path,
        DecisionRecord.from_mapping,
    )
    metadata = _parse_jsonl(metadata_path, MetadataRecord.from_mapping)
    record_by_id = _index_unique(records, source=records_path)
    metadata_by_id = _index_unique(metadata, source=metadata_path)

    record_ids = set(record_by_id)
    metadata_ids = set(metadata_by_id)
    if record_ids != metadata_ids:
        missing_metadata = sorted(record_ids - metadata_ids)
        missing_records = sorted(metadata_ids - record_ids)
        details = []
        if missing_metadata:
            details.append(f"missing metadata for: {', '.join(missing_metadata)}")
        if missing_records:
            details.append(f"missing examples for: {', '.join(missing_records)}")
        raise DatasetValidationError("; ".join(details))

    if expected_split is not None:
        wrong_split = sorted(
            record.id for record in metadata if record.split is not expected_split
        )
        if wrong_split:
            raise DatasetValidationError(
                f"records do not use expected split {expected_split.value}: "
                f"{', '.join(wrong_split)}"
            )

    _validate_decisions(metadata)
    _validate_target_decisions(record_by_id, metadata_by_id)
    _validate_reviewer_ids(metadata)
    _validate_unique_requests(records)
    minimal_pairs = _validate_minimal_pairs(record_by_id, metadata)

    decisions = Counter(record.gold_decision.value for record in metadata)
    domains = Counter(record.domain.value for record in metadata)
    categories = Counter(record.category.value for record in metadata)
    return ValidationSummary(
        records=len(records),
        minimal_pairs=minimal_pairs,
        decisions=dict(sorted(decisions.items())),
        domains=dict(sorted(domains.items())),
        categories=dict(sorted(categories.items())),
    )


ParsedRecord = TypeVar("ParsedRecord", DecisionRecord, MetadataRecord)


def _parse_jsonl(
    path: Path,
    parser: Callable[[object], ParsedRecord],
) -> tuple[ParsedRecord, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DatasetValidationError(f"cannot read {path}: {error}") from error
    if not lines:
        raise DatasetValidationError(f"{path} must not be empty")

    records: list[ParsedRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetValidationError(f"{path}:{line_number}: blank JSONL line")
        try:
            value = json.loads(line)
            records.append(parser(value))
        except (json.JSONDecodeError, SchemaError) as error:
            raise DatasetValidationError(f"{path}:{line_number}: {error}") from error
    return tuple(records)


def _index_unique(
    records: tuple[ParsedRecord, ...], *, source: Path
) -> dict[str, ParsedRecord]:
    indexed: dict[str, ParsedRecord] = {}
    for record in records:
        if record.id in indexed:
            raise DatasetValidationError(f"{source}: duplicate id {record.id}")
        indexed[record.id] = record
    return indexed


def _validate_decisions(metadata: tuple[MetadataRecord, ...]) -> None:
    for record in metadata:
        if (
            record.category in CHALLENGE_CATEGORIES
            and record.gold_decision is not GoldDecision.CHALLENGE
        ):
            raise DatasetValidationError(
                f"{record.id}: challenge category requires challenge decision"
            )
        if (
            record.category in COMPLY_CATEGORIES
            and record.gold_decision is not GoldDecision.COMPLY
        ):
            raise DatasetValidationError(
                f"{record.id}: reasonable category requires comply decision"
            )


def _validate_reviewer_ids(metadata: tuple[MetadataRecord, ...]) -> None:
    for record in metadata:
        reviewers = [review.reviewer for review in record.review.independent]
        if len(reviewers) != len(set(reviewers)):
            raise DatasetValidationError(f"{record.id}: duplicate independent reviewer")


def _validate_target_decisions(
    records: Mapping[str, DecisionRecord],
    metadata: Mapping[str, MetadataRecord],
) -> None:
    for record_id, record in records.items():
        expected = metadata[record_id].gold_decision
        if record.target.decision is not expected:
            raise DatasetValidationError(
                f"{record_id}: target decision does not match metadata gold decision"
            )


def _validate_unique_requests(
    records: tuple[DecisionRecord, ...],
) -> None:
    seen: dict[str, str] = {}
    for record in records:
        normalized = " ".join(record.input.casefold().split())
        duplicate = seen.get(normalized)
        if duplicate is not None:
            raise DatasetValidationError(
                f"{record.id}: duplicates user request from {duplicate}"
            )
        seen[normalized] = record.id


def _validate_minimal_pairs(
    records_by_id: Mapping[str, DecisionRecord],
    metadata: tuple[MetadataRecord, ...],
) -> int:
    groups: dict[str, list[MetadataRecord]] = defaultdict(list)
    for record in metadata:
        if record.minimal_pair_id is not None:
            groups[record.minimal_pair_id].append(record)

    for pair_id, records in groups.items():
        if len(records) != 2:
            raise DatasetValidationError(
                f"{pair_id}: minimal pair must contain exactly two records"
            )
        decisions = {record.gold_decision for record in records}
        if decisions != {GoldDecision.CHALLENGE, GoldDecision.COMPLY}:
            raise DatasetValidationError(
                f"{pair_id}: minimal pair must contain challenge and comply decisions"
            )
        domains = {record.domain for record in records}
        if len(domains) != 1:
            raise DatasetValidationError(
                f"{pair_id}: minimal pair records must share a domain"
            )
        requests = {records_by_id[record.id].input for record in records}
        if len(requests) != 2:
            raise DatasetValidationError(
                f"{pair_id}: minimal pair requests must be distinct"
            )
    return len(groups)

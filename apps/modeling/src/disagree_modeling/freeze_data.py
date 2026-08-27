"""Validate and freeze the reported dataset into a deterministic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TypeVar

from disagree_contracts.schemas import (
    AdjudicationStatus,
    Category,
    DecisionRecord,
    Domain,
    MetadataRecord,
    ReviewStatus,
    SchemaError,
    Split,
)

from disagree_modeling.data import (
    DatasetValidationError,
    ValidationSummary,
    validate_dataset,
)

MANIFEST_SCHEMA_VERSION = 1
NEAR_DUPLICATE_THRESHOLD = 0.90
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"


class FreezeDataError(ValueError):
    """Raised when reviewed splits are not ready to freeze."""


@dataclass(frozen=True, slots=True)
class SplitExpectation:
    records: int
    decisions: Mapping[str, int]
    minimal_pairs: int
    categories: Mapping[str, int] | None = None


TRAIN_CATEGORY_DISTRIBUTION: Mapping[str, int] = {
    Category.UNSUPPORTED_ASSUMPTION.value: 60,
    Category.INTERNAL_CONTRADICTION.value: 40,
    Category.MISSING_CONSTRAINT.value: 40,
    Category.IMPLAUSIBLE_ESTIMATE.value: 40,
    Category.AUTHORIZATION_RISK.value: 40,
    Category.MATERIAL_HARM.value: 20,
    Category.STRAIGHTFORWARD.value: 60,
    Category.CONSTRAINED.value: 60,
    Category.SAFE_NEAR_NEIGHBOR.value: 40,
}


EXPECTED_SPLITS: Mapping[Split, SplitExpectation] = {
    Split.TRAIN: SplitExpectation(
        records=400,
        decisions={"challenge": 240, "comply": 160},
        minimal_pairs=80,
        categories=TRAIN_CATEGORY_DISTRIBUTION,
    ),
    Split.VALIDATION: SplitExpectation(
        records=50,
        decisions={"challenge": 30, "comply": 20},
        minimal_pairs=10,
    ),
    Split.TEST: SplitExpectation(
        records=100,
        decisions={"challenge": 50, "comply": 50},
        minimal_pairs=20,
    ),
}

ParsedRecord = TypeVar("ParsedRecord", DecisionRecord, MetadataRecord)


@dataclass(frozen=True, slots=True)
class _IndexedInput:
    id: str
    split: Split
    normalized: str
    tokens: tuple[str, ...]


def build_manifest(experiment_root: Path) -> dict[str, Any]:
    """Validate all reported inputs and return their canonical manifest."""
    root = experiment_root.resolve()
    split_entries: dict[str, object] = {}
    all_inputs: list[_IndexedInput] = []
    all_ids: dict[str, Split] = {}

    for split, expectation in EXPECTED_SPLITS.items():
        examples_path = root / "data" / "splits" / f"{split.value}.jsonl"
        metadata_path = root / "data" / "metadata" / f"{split.value}.jsonl"
        try:
            summary = validate_dataset(
                examples_path,
                metadata_path,
                expected_split=split,
            )
        except DatasetValidationError as error:
            raise FreezeDataError(f"{split.value}: {error}") from error

        _validate_expected_distribution(split, summary, expectation)
        metadata = _load_jsonl(metadata_path, MetadataRecord.from_mapping)
        records = _load_jsonl(examples_path, DecisionRecord.from_mapping)
        _validate_review_completion(split, metadata)

        for record in records:
            previous_split = all_ids.get(record.id)
            if previous_split is not None:
                raise FreezeDataError(
                    f"{record.id}: duplicate id across {previous_split.value} and "
                    f"{split.value} splits"
                )
            all_ids[record.id] = split
            tokens = _input_tokens(record.input)
            all_inputs.append(
                _IndexedInput(
                    id=record.id,
                    split=split,
                    normalized=" ".join(tokens),
                    tokens=tokens,
                )
            )

        split_entries[split.value] = {
            "examples": _file_descriptor(
                examples_path,
                root=root,
                record_count=summary.records,
            ),
            "metadata": _file_descriptor(
                metadata_path,
                root=root,
                record_count=summary.records,
            ),
            "summary": summary.as_dict(),
        }

    _validate_cross_split_inputs(all_inputs)

    basic_prompt = root / "prompts" / "basic.txt"
    strong_prompt = root / "prompts" / "strong.txt"
    response_schema = root / "schema" / "decision-response-v1.schema.json"
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "splits": split_entries,
        "prompts": {
            "basic": _file_descriptor(basic_prompt, root=root),
            "strong": _file_descriptor(strong_prompt, root=root),
        },
        "response_schema": _file_descriptor(response_schema, root=root),
    }


def freeze_dataset(experiment_root: Path, *, check: bool = False) -> dict[str, Any]:
    """Write the canonical manifest, or verify an existing one with ``check``."""
    root = experiment_root.resolve()
    manifest = build_manifest(root)
    manifest_path = root / "data" / "manifest.json"

    if check:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise FreezeDataError(f"cannot read {manifest_path}: {error}") from error
        except json.JSONDecodeError as error:
            raise FreezeDataError(
                f"invalid JSON in {manifest_path}: {error}"
            ) from error
        if existing != manifest:
            raise FreezeDataError(
                "dataset manifest does not match the validated files; "
                "run disagree-freeze-data to regenerate it"
            )
        return manifest

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_expected_distribution(
    split: Split,
    summary: ValidationSummary,
    expectation: SplitExpectation,
) -> None:
    if summary.records != expectation.records:
        raise FreezeDataError(
            f"{split.value}: expected {expectation.records} records, "
            f"found {summary.records}"
        )
    if dict(summary.decisions) != dict(expectation.decisions):
        raise FreezeDataError(
            f"{split.value}: expected decisions {dict(expectation.decisions)}, "
            f"found {dict(summary.decisions)}"
        )
    if summary.minimal_pairs != expectation.minimal_pairs:
        raise FreezeDataError(
            f"{split.value}: expected {expectation.minimal_pairs} minimal pairs, "
            f"found {summary.minimal_pairs}"
        )

    expected_domains = {domain.value for domain in Domain}
    missing_domains = sorted(expected_domains - set(summary.domains))
    if missing_domains:
        raise FreezeDataError(
            f"{split.value}: missing domains: {', '.join(missing_domains)}"
        )

    expected_categories = {category.value for category in Category}
    missing_categories = sorted(expected_categories - set(summary.categories))
    if missing_categories:
        raise FreezeDataError(
            f"{split.value}: missing categories: {', '.join(missing_categories)}"
        )
    if expectation.categories is not None and dict(summary.categories) != dict(
        expectation.categories
    ):
        raise FreezeDataError(
            f"{split.value}: expected category distribution "
            f"{dict(expectation.categories)}, found {dict(summary.categories)}"
        )


def _validate_review_completion(
    split: Split,
    metadata: Sequence[MetadataRecord],
) -> None:
    unfinished = {
        AdjudicationStatus.NOT_STARTED,
        AdjudicationStatus.PENDING,
    }
    for record in metadata:
        accepted_reviewers = {
            review.reviewer
            for review in record.review.independent
            if review.status is ReviewStatus.ACCEPTED
        }
        if len(accepted_reviewers) < 2:
            raise FreezeDataError(
                f"{record.id}: {split.value} record requires at least two distinct "
                "accepted independent reviewers"
            )
        if record.review.adjudication in unfinished:
            raise FreezeDataError(
                f"{record.id}: adjudication must be not_needed or resolved before freeze"
            )


def _validate_cross_split_inputs(records: Sequence[_IndexedInput]) -> None:
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left.split is right.split:
                continue
            if left.normalized == right.normalized:
                raise FreezeDataError(
                    f"exact duplicate inputs across splits: {left.id} "
                    f"({left.split.value}) and {right.id} ({right.split.value})"
                )
            similarity = _near_duplicate_similarity(left.tokens, right.tokens)
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                raise FreezeDataError(
                    "high-confidence near-duplicate inputs across splits: "
                    f"{left.id} ({left.split.value}) and {right.id} "
                    f"({right.split.value}); similarity={similarity:.3f}"
                )


def _near_duplicate_similarity(
    left: Sequence[str],
    right: Sequence[str],
) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _input_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _file_descriptor(
    path: Path,
    *,
    root: Path,
    record_count: int | None = None,
) -> dict[str, object]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise FreezeDataError(f"cannot read {path}: {error}") from error
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as error:
        raise FreezeDataError(f"{path} must be inside {root}") from error

    descriptor: dict[str, object] = {
        "path": relative_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_size": len(content),
    }
    if record_count is not None:
        descriptor["record_count"] = record_count
    return descriptor


def _load_jsonl(
    path: Path,
    parser: Callable[[object], ParsedRecord],
) -> tuple[ParsedRecord, ...]:
    records: list[ParsedRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FreezeDataError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            records.append(parser(json.loads(line)))
        except (json.JSONDecodeError, SchemaError) as error:
            raise FreezeDataError(f"{path}:{line_number}: {error}") from error
    return tuple(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate reviewed splits and write or check their manifest."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="experiment directory containing data, prompts, and schema",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing manifest instead of writing it",
    )
    args = parser.parse_args()
    manifest = freeze_dataset(args.experiment_root, check=args.check)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

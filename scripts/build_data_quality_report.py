"""Build or check the deterministic public dataset quality report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "experiment" / "data"
SPLITS = ("test", "validation", "train")
NEAR_DUPLICATE_THRESHOLD = 0.90
MAX_TEXT_LENGTH = 600


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report() -> dict[str, Any]:
    indexed: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    all_messages: Counter[str] = Counter()
    sensitive_hits: list[dict[str, str]] = []
    overlong_fields: list[dict[str, Any]] = []

    for split in SPLITS:
        examples_path = DATA_ROOT / "splits" / f"{split}.jsonl"
        metadata_path = DATA_ROOT / "metadata" / f"{split}.jsonl"
        examples = _read_jsonl(examples_path)
        metadata = _read_jsonl(metadata_path)
        metadata_by_id = {record["id"]: record for record in metadata}
        decisions = Counter(record["target"]["decision"] for record in examples)
        questions = sum(record["target"]["question"] is not None for record in examples)
        openings: Counter[str] = Counter()

        for record in examples:
            record_id = str(record["id"])
            source = str(record["input"])
            target = record["target"]
            message = str(target["message"])
            all_messages[message.casefold()] += 1
            opening = " ".join(_tokens(message)[:2])
            openings[opening] += 1
            for field, value in (("input", source), *target.items()):
                if isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
                    overlong_fields.append(
                        {"id": record_id, "field": field, "length": len(value)}
                    )
                if isinstance(value, str) and re.search(
                    r"https?://|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
                    value,
                    flags=re.IGNORECASE,
                ):
                    sensitive_hits.append({"id": record_id, "field": field})
            pair_id = metadata_by_id[record_id]["minimal_pair_id"]
            indexed.append(
                {
                    "id": record_id,
                    "split": split,
                    "pair_id": pair_id,
                    "normalized": " ".join(_tokens(source)),
                    "tokens": _tokens(source),
                }
            )

        reviews_complete = all(
            len(
                {
                    review["reviewer"]
                    for review in record["review"]["independent"]
                    if review["status"] == "accepted"
                }
            )
            >= 2
            and record["review"]["adjudication"] in {"not_needed", "resolved"}
            for record in metadata
        )
        split_reports[split] = {
            "records": len(examples),
            "decisions": dict(sorted(decisions.items())),
            "questions": questions,
            "reviews_complete": reviews_complete,
            "top_message_openings": [
                {"opening": opening, "count": count}
                for opening, count in openings.most_common(10)
            ],
            "examples_sha256": _sha256(examples_path),
            "metadata_sha256": _sha256(metadata_path),
        }

    exact_duplicates: list[dict[str, str]] = []
    near_duplicates: list[dict[str, Any]] = []
    for left_index, left in enumerate(indexed):
        for right in indexed[left_index + 1 :]:
            if left["pair_id"] is not None and left["pair_id"] == right["pair_id"]:
                continue
            if left["normalized"] == right["normalized"]:
                exact_duplicates.append({"left": left["id"], "right": right["id"]})
                continue
            similarity = SequenceMatcher(
                None,
                left["tokens"],
                right["tokens"],
                autojunk=False,
            ).ratio()
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicates.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "similarity": round(similarity, 6),
                    }
                )

    repeated_messages = [
        {"message": message, "count": count}
        for message, count in sorted(all_messages.items())
        if count > 1
    ]
    checks = {
        "all_reviews_complete": all(
            report["reviews_complete"] for report in split_reports.values()
        ),
        "no_exact_duplicate_inputs": not exact_duplicates,
        "no_unpaired_near_duplicate_inputs": not near_duplicates,
        "no_exact_duplicate_messages": not repeated_messages,
        "no_overlong_fields": not overlong_fields,
        "no_url_or_email_markers": not sensitive_hits,
    }
    return {
        "schema_version": 1,
        "thresholds": {
            "near_duplicate_similarity": NEAR_DUPLICATE_THRESHOLD,
            "maximum_text_length": MAX_TEXT_LENGTH,
        },
        "splits": split_reports,
        "findings": {
            "exact_duplicate_inputs": exact_duplicates,
            "unpaired_near_duplicate_inputs": near_duplicates,
            "exact_duplicate_messages": repeated_messages,
            "overlong_fields": overlong_fields,
            "url_or_email_markers": sensitive_hits,
        },
        "checks": checks,
        "quality_gate_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    path = DATA_ROOT / "quality-report.json"
    if args.check:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit("quality report is stale; regenerate it")
    else:
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["quality_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

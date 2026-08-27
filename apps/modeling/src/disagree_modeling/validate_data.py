"""Command-line entry point for public dataset validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from disagree_contracts.schemas import Split

from disagree_modeling.data import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate structured decision JSONL and its metadata sidecar."
    )
    parser.add_argument("examples", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--split", choices=[split.value for split in Split])
    args = parser.parse_args()

    expected_split = None if args.split is None else Split(args.split)
    summary = validate_dataset(
        args.examples,
        args.metadata,
        expected_split=expected_split,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Print the primary-training run plan without contacting Modal."""

from __future__ import annotations

import json
from pathlib import Path

from disagree_modeling.training import DEFAULT_RUN_ID, build_training_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment"


def main() -> int:
    print(
        json.dumps(
            build_training_plan(EXPERIMENT_ROOT, DEFAULT_RUN_ID),
            indent=2,
            sort_keys=True,
        )
    )
    return 0

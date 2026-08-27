"""Print the model compatibility plan without contacting Modal."""

from __future__ import annotations

import json

from disagree_modeling.compatibility_check import build_compatibility_plan


def main() -> int:
    print(json.dumps(build_compatibility_plan(), indent=2, sort_keys=True))
    return 0

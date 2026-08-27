"""Run and enforce repository coverage across the isolated uv projects."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLOBAL_MINIMUM = 90.0
FILE_MINIMUM = 85.0
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Project:
    label: str
    directory: Path


PROJECTS = (
    Project("contracts", ROOT / "packages" / "contracts"),
    Project("modeling", ROOT / "apps" / "modeling"),
    Project("serving", ROOT / "apps" / "serving"),
)


def _run(command: list[str]) -> None:
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _opportunities(summary: dict[str, Any]) -> tuple[int, int]:
    covered = int(summary["covered_lines"]) + int(summary["covered_branches"])
    total = int(summary["num_statements"]) + int(summary["num_branches"])
    return covered, total


def _percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered / total * 100


def _collect_report(project: Project, temporary_root: Path) -> dict[str, Any]:
    coverage_data = temporary_root / f".coverage-{project.label}"
    coverage_json = temporary_root / f"coverage-{project.label}.json"
    common = ["uv", "run", "--directory", str(project.directory), "--frozen"]

    _run(
        [
            *common,
            "coverage",
            "run",
            f"--data-file={coverage_data}",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
        ]
    )
    _run(
        [
            *common,
            "coverage",
            "json",
            f"--data-file={coverage_data}",
            "-o",
            str(coverage_json),
        ]
    )
    return json.loads(coverage_json.read_text(encoding="utf-8"))


def main() -> int:
    file_results: list[tuple[str, int, int]] = []
    global_covered = 0
    global_total = 0

    with tempfile.TemporaryDirectory(prefix="fine-tuning-coverage-") as directory:
        temporary_root = Path(directory)
        for project in PROJECTS:
            report = _collect_report(project, temporary_root)
            for filename, details in sorted(report["files"].items()):
                covered, total = _opportunities(details["summary"])
                display_name = f"{project.directory.relative_to(ROOT)}/{filename}"
                file_results.append((display_name, covered, total))
                global_covered += covered
                global_total += total

    failures: list[str] = []
    print("\nCombined statement and branch coverage:")
    for filename, covered, total in file_results:
        percent = _percent(covered, total)
        print(f"  {percent:6.2f}%  {covered:>4}/{total:<4}  {filename}")
        if percent < FILE_MINIMUM:
            failures.append(
                f"{filename} is {percent:.2f}% (minimum {FILE_MINIMUM:.2f}%)"
            )

    global_percent = _percent(global_covered, global_total)
    print(f"\nGlobal: {global_percent:.2f}% ({global_covered}/{global_total})")
    if global_percent < GLOBAL_MINIMUM:
        failures.append(
            f"global coverage is {global_percent:.2f}% (minimum {GLOBAL_MINIMUM:.2f}%)"
        )

    if failures:
        print("\nCoverage gate failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"Coverage gate passed: global >= {GLOBAL_MINIMUM:.2f}% and "
        f"every file >= {FILE_MINIMUM:.2f}%."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

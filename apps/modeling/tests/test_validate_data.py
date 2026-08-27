from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from disagree_modeling.validate_data import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class ValidateDataCliTest(unittest.TestCase):
    def test_prints_validation_summary_for_expected_split(self) -> None:
        output = io.StringIO()
        argv = [
            "disagree-validate-data",
            str(FIXTURES / "tiny-examples.jsonl"),
            str(FIXTURES / "tiny-metadata.jsonl"),
            "--split",
            "fixture",
        ]

        with patch.object(sys, "argv", argv), redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["records"], 10)

    def test_accepts_validation_without_expected_split(self) -> None:
        output = io.StringIO()
        argv = [
            "disagree-validate-data",
            str(FIXTURES / "tiny-examples.jsonl"),
            str(FIXTURES / "tiny-metadata.jsonl"),
        ]

        with patch.object(sys, "argv", argv), redirect_stdout(output):
            result = main()

        self.assertEqual(result, 0)

from __future__ import annotations

import unittest

import disagree_contracts


class ScaffoldTest(unittest.TestCase):
    def test_package_version_matches_scaffold(self) -> None:
        self.assertEqual(disagree_contracts.__version__, "0.0.0")


if __name__ == "__main__":
    unittest.main()

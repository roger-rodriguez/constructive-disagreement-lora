from __future__ import annotations

import unittest

import disagree_contracts
import disagree_serving
from disagree_serving.modal_app import (
    APP_NAME,
    FASTAPI_VERSION,
    MIN_CONTAINERS,
    PYDANTIC_VERSION,
    health_payload,
)


class ScaffoldTest(unittest.TestCase):
    def test_local_packages_are_importable(self) -> None:
        self.assertEqual(disagree_serving.__version__, "0.0.0")
        self.assertEqual(disagree_contracts.__version__, "0.0.0")

    def test_deployment_configuration_is_stable(self) -> None:
        self.assertEqual(APP_NAME, "constructive-disagreement")
        self.assertEqual(FASTAPI_VERSION, "0.141.1")
        self.assertEqual(PYDANTIC_VERSION, "2.12.5")
        self.assertEqual(MIN_CONTAINERS, 0)
        self.assertEqual(
            health_payload(),
            {
                "service": "constructive-disagreement",
                "status": "ok",
                "version": "0.0.0",
            },
        )


if __name__ == "__main__":
    unittest.main()

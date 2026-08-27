from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from disagree_serving.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "model.toml"


class ConfigTest(unittest.TestCase):
    def test_loads_frozen_serving_contract(self) -> None:
        config = load_config(CONFIG_PATH)

        self.assertEqual(config.service.app_name, "constructive-disagreement")
        self.assertEqual(
            config.service.health_label,
            "constructive-disagreement-health",
        )
        self.assertEqual(
            config.service.decision_label,
            "constructive-disagreement",
        )
        self.assertEqual(config.model.base_id, "Qwen/Qwen3-4B-Instruct-2507")
        self.assertEqual(
            config.model.base_revision,
            "cdbee75f17c01a7cc42f958dc650907174af0554",
        )
        self.assertEqual(
            config.model.adapter_id,
            "rrod/qwen3-4b-constructive-disagreement-lora",
        )
        self.assertEqual(config.generation.max_input_tokens, 768)
        self.assertFalse(config.generation.do_sample)
        self.assertEqual(config.runtime.gpu, "L4")
        self.assertEqual(config.runtime.max_containers, 1)
        self.assertEqual(config.runtime.min_containers, 0)

    def test_rejects_unknown_top_level_section(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8") + "\n[unknown]\nvalue = 1\n"
        with self.assertRaisesRegex(ValueError, "config must contain exactly"):
            self._load_text(text)

    def test_rejects_non_table_section(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            """[service]
app_name = "constructive-disagreement"
health_label = "constructive-disagreement-health"
decision_label = "constructive-disagreement"
""",
            'service = "invalid"\n',
        )
        with self.assertRaisesRegex(TypeError, "service must be a table"):
            self._load_text(text)

    def test_rejects_missing_section_key(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            'health_label = "constructive-disagreement-health"\n', ""
        )
        with self.assertRaisesRegex(ValueError, "service must contain exactly"):
            self._load_text(text)

    def test_rejects_empty_text(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace('gpu = "L4"', 'gpu = ""')
        with self.assertRaisesRegex(TypeError, "gpu must be non-empty text"):
            self._load_text(text)

    def test_rejects_invalid_revision(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "cdbee75f17c01a7cc42f958dc650907174af0554", "main"
        )
        with self.assertRaisesRegex(ValueError, "base_revision"):
            self._load_text(text)

    def test_rejects_invalid_prompt_digest(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "638febed692b5c909932753654dd8d167282a775b2aa15fdf0d8a2d79dc2c3b0",
            "not-a-digest",
        )
        with self.assertRaisesRegex(ValueError, "sha256"):
            self._load_text(text)

    def test_rejects_boolean_where_positive_integer_is_required(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "max_new_tokens = 256", "max_new_tokens = true"
        )
        with self.assertRaisesRegex(TypeError, "max_new_tokens"):
            self._load_text(text)

    def test_rejects_negative_minimum_containers(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "min_containers = 0", "min_containers = -1"
        )
        with self.assertRaisesRegex(TypeError, "min_containers"):
            self._load_text(text)

    def test_rejects_boolean_cpu(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "cpu = 4.0", "cpu = true"
        )
        with self.assertRaisesRegex(TypeError, "cpu"):
            self._load_text(text)

    def test_rejects_non_boolean_sampling(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "do_sample = false", 'do_sample = "false"'
        )
        with self.assertRaisesRegex(TypeError, "do_sample"):
            self._load_text(text)

    def test_rejects_inverted_container_limits(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "min_containers = 0", "min_containers = 2"
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self._load_text(text)

    def test_rejects_total_token_budget_above_context(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "max_input_tokens = 768", "max_input_tokens = 769"
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed 1024"):
            self._load_text(text)

    def _load_text(self, text: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.toml"
            path.write_text(text, encoding="utf-8")
            load_config(path)


if __name__ == "__main__":
    unittest.main()

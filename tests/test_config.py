import json
import tempfile
import unittest
from pathlib import Path

from swe_agent.benchmark.config import load_benchmark_config


class BenchmarkConfigTests(unittest.TestCase):
    def _write_config(self, payload: object) -> str:
        temporary_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        with temporary_file:
            json.dump(payload, temporary_file)
        self.addCleanup(Path(temporary_file.name).unlink)
        return temporary_file.name

    def test_loads_valid_config(self) -> None:
        config_path = self._write_config(
            {"acme/widget": {"issue_number": 42, "test_command": ["pytest", "-q"]}}
        )

        entries = load_benchmark_config(config_path)

        self.assertEqual(entries[0].owner, "acme")
        self.assertEqual(entries[0].repository, "widget")
        self.assertEqual(entries[0].issue_number, 42)
        self.assertEqual(entries[0].test_command, ["pytest", "-q"])

    def test_rejects_missing_required_field(self) -> None:
        config_path = self._write_config({"acme/widget": {"issue_number": 42}})

        with self.assertRaisesRegex(ValueError, "test_command"):
            load_benchmark_config(config_path)

    def test_rejects_malformed_repository_key(self) -> None:
        config_path = self._write_config(
            {"not-a-repository": {"issue_number": 42, "test_command": ["pytest"]}}
        )

        with self.assertRaisesRegex(ValueError, "Malformed benchmark repository key"):
            load_benchmark_config(config_path)


if __name__ == "__main__":
    unittest.main()

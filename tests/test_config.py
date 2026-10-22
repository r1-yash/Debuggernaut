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
            {
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "a" * 40,
                    "test_command": ["pytest", "-q"],
                }
            }
        )

        entries = load_benchmark_config(config_path)

        self.assertEqual(entries[0].owner, "acme")
        self.assertEqual(entries[0].repository, "widget")
        self.assertEqual(entries[0].issue_number, 42)
        self.assertEqual(entries[0].commit, "a" * 40)
        self.assertEqual(entries[0].test_command, ["pytest", "-q"])

    def test_rejects_missing_required_field(self) -> None:
        config_path = self._write_config({"acme/widget": {"issue_number": 42}})

        with self.assertRaisesRegex(ValueError, "test_command"):
            load_benchmark_config(config_path)

    def test_rejects_missing_commit(self) -> None:
        config_path = self._write_config(
            {"acme/widget": {"issue_number": 42, "test_command": ["pytest"]}}
        )

        with self.assertRaisesRegex(ValueError, "pinned commit"):
            load_benchmark_config(config_path)

    def test_rejects_short_commit(self) -> None:
        config_path = self._write_config(
            {
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "abc1234",
                    "test_command": ["pytest"],
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "pinned commit"):
            load_benchmark_config(config_path)

    def test_rejects_non_hex_commit(self) -> None:
        config_path = self._write_config(
            {
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "z" * 40,
                    "test_command": ["pytest"],
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "pinned commit"):
            load_benchmark_config(config_path)

    def test_rejects_malformed_repository_key(self) -> None:
        config_path = self._write_config(
            {"not-a-repository": {"issue_number": 42, "test_command": ["pytest"]}}
        )

        with self.assertRaisesRegex(ValueError, "Malformed benchmark repository key"):
            load_benchmark_config(config_path)

    def test_test_patch_defaults_to_none(self) -> None:
        config_path = self._write_config(
            {
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "a" * 40,
                    "test_command": ["pytest"],
                }
            }
        )

        entries = load_benchmark_config(config_path)

        self.assertIsNone(entries[0].test_patch)

    def test_loads_relative_test_patch_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            patches = config_dir / "patches"
            patches.mkdir()
            patch_path = patches / "widget.diff"
            patch_path.write_text("")
            config_path = config_dir / "repos.json"
            config_path.write_text(
                json.dumps(
                    {
                        "acme/widget": {
                            "issue_number": 42,
                            "commit": "a" * 40,
                            "test_command": ["pytest"],
                            "test_patch": "patches/widget.diff",
                        }
                    }
                )
            )

            entries = load_benchmark_config(str(config_path))

            self.assertEqual(entries[0].test_patch, str(patch_path.resolve()))

    def test_rejects_missing_test_patch_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "repos.json"
            config_path.write_text(
                json.dumps(
                    {
                        "acme/widget": {
                            "issue_number": 42,
                            "commit": "a" * 40,
                            "test_command": ["pytest"],
                            "test_patch": "patches/missing.diff",
                        }
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "test_patch file does not exist"):
                load_benchmark_config(str(config_path))

    def test_rejects_absolute_test_patch_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "repos.json"
            patch_path = Path(directory) / "widget.diff"
            patch_path.write_text("")
            config_path.write_text(
                json.dumps(
                    {
                        "acme/widget": {
                            "issue_number": 42,
                            "commit": "a" * 40,
                            "test_command": ["pytest"],
                            "test_patch": str(patch_path),
                        }
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "must be relative"):
                load_benchmark_config(str(config_path))

    def test_rejects_test_patch_outside_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "config"
            config_dir.mkdir()
            patch_path = root / "widget.diff"
            patch_path.write_text("")
            config_path = config_dir / "repos.json"
            config_path.write_text(
                json.dumps(
                    {
                        "acme/widget": {
                            "issue_number": 42,
                            "commit": "a" * 40,
                            "test_command": ["pytest"],
                            "test_patch": "../widget.diff",
                        }
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "must stay within"):
                load_benchmark_config(str(config_path))

    def test_rejects_non_string_test_patch(self) -> None:
        config_path = self._write_config(
            {
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "a" * 40,
                    "test_command": ["pytest"],
                    "test_patch": 7,
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "non-empty string"):
            load_benchmark_config(config_path)


if __name__ == "__main__":
    unittest.main()

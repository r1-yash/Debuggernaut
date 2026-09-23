import tempfile
import unittest
from pathlib import Path

from swe_agent.agent.paths import repository_root, resolve_repository_path


class PathsTests(unittest.TestCase):
    def test_repository_root_rejects_non_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "existing directory"):
            repository_root("does-not-exist")

    def test_resolve_repository_path_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir) / "repository"
            repository.mkdir()
            outside_file = Path(temporary_dir) / "outside.txt"
            outside_file.write_text("private")

            with self.assertRaisesRegex(ValueError, "within the repository"):
                resolve_repository_path(str(repository), "../outside.txt")


if __name__ == "__main__":
    unittest.main()

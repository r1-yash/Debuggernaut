import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from swe_agent.ingestion.clone import clone_repository


class CloneRepositoryTests(unittest.TestCase):
    @patch("swe_agent.ingestion.clone.subprocess.run")
    def test_rejects_missing_owner_or_repository(self, run: Mock) -> None:
        for owner, repository in (("", "widget"), ("acme", "   ")):
            with self.subTest(owner=owner, repository=repository):
                with self.assertRaises(ValueError):
                    clone_repository(owner, repository)

        run.assert_not_called()

    @patch("swe_agent.ingestion.clone.subprocess.run")
    def test_clones_repository_and_returns_absolute_path(self, run: Mock) -> None:
        run.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace_dir = os.path.join(temporary_dir, "workspace")
            cloned_path = clone_repository("acme", "widget", workspace_dir)

            target_path = os.path.join(workspace_dir, "acme__widget")
            self.assertEqual(cloned_path, os.path.abspath(target_path))
            run.assert_called_once_with(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/acme/widget.git",
                    target_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )

    @patch("swe_agent.ingestion.clone.shutil.rmtree")
    @patch("swe_agent.ingestion.clone.subprocess.run")
    def test_removes_existing_clone_before_cloning(
        self,
        run: Mock,
        rmtree: Mock,
    ) -> None:
        run.return_value = Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace_dir = os.path.join(temporary_dir, "workspace")
            target_path = os.path.join(workspace_dir, "acme__widget")
            os.makedirs(target_path)

            clone_repository("acme", "widget", workspace_dir)

            rmtree.assert_called_once_with(target_path)

    @patch("swe_agent.ingestion.clone.subprocess.run")
    def test_reports_git_error_when_clone_fails(self, run: Mock) -> None:
        run.return_value = Mock(
            returncode=128,
            stdout="",
            stderr="fatal: repository not found",
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "fatal: repository not found"):
                clone_repository("missing", "repository", temporary_dir)

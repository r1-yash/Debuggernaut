import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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


def _git(repo_path: Path, *arguments: str) -> str:
    """Run a git command in a repository and return its stripped stdout."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_upstream_with_bug_then_fix(upstream: Path) -> tuple[str, str]:
    """Create a two-commit repository, returning its (bug commit, fix commit) SHAs."""
    upstream.mkdir()
    _git(upstream, "init")
    _git(upstream, "config", "user.email", "test@example.com")
    _git(upstream, "config", "user.name", "Test")
    (upstream / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(upstream, "add", "calc.py")
    _git(upstream, "commit", "-m", "introduce bug")
    bug_commit = _git(upstream, "rev-parse", "HEAD")
    (upstream / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(upstream, "commit", "-am", "fix bug")
    fix_commit = _git(upstream, "rev-parse", "HEAD")
    return bug_commit, fix_commit


def test_checks_out_pinned_commit(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    bug_commit, fix_commit = _make_upstream_with_bug_then_fix(upstream)

    cloned = clone_repository(
        "acme",
        "widget",
        str(tmp_path / "workspace"),
        commit=bug_commit,
        remote_url=str(upstream),
    )

    assert _git(Path(cloned), "rev-parse", "HEAD") == bug_commit
    assert (Path(cloned) / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert bug_commit != fix_commit


def test_unfetchable_commit_raises_readable_error(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    _make_upstream_with_bug_then_fix(upstream)
    missing_commit = "f" * 40

    with pytest.raises(ValueError) as error:
        clone_repository(
            "acme",
            "widget",
            str(tmp_path / "workspace"),
            commit=missing_commit,
            remote_url=str(upstream),
        )

    assert missing_commit in str(error.value)

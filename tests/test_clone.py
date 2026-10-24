import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import pytest

from swe_agent.git_runner import DEFAULT_GIT_TIMEOUT_SECONDS, GitRunner
from swe_agent.ingestion.clone import clone_repository
from swe_agent.process import CommandResult


class RecordingCommandRunner:
    """A deterministic command boundary for clone unit tests."""

    def __init__(self, outcomes: Sequence[CommandResult]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[list[str], str, dict[str, str], float | None]] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandResult:
        self.calls.append(
            (list(command), os.fspath(cwd), dict(env), timeout)
        )
        if not self._outcomes:
            raise AssertionError(f"Unexpected command: {list(command)!r}")
        outcome = self._outcomes.pop(0)
        return outcome.model_copy(update={"command": list(command)})


class CloneRepositoryTests(unittest.TestCase):
    def test_rejects_missing_owner_or_repository(self) -> None:
        for owner, repository in (("", "widget"), ("acme", "   ")):
            with self.subTest(owner=owner, repository=repository):
                with self.assertRaises(ValueError):
                    clone_repository(owner, repository)

    def test_clones_repository_through_timed_git_runner(self) -> None:
        command_runner = RecordingCommandRunner(
            [
                CommandResult(
                    command=[],
                    return_code=0,
                )
            ]
        )
        git_runner = GitRunner(
            command_runner=command_runner,
            timeout_seconds=DEFAULT_GIT_TIMEOUT_SECONDS,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace_dir = os.path.join(temporary_dir, "workspace")
            cloned_path = clone_repository(
                "acme",
                "widget",
                workspace_dir,
                git_runner=git_runner,
            )

            target_path = os.path.join(workspace_dir, "acme__widget")
            self.assertEqual(cloned_path, os.path.abspath(target_path))
            self.assertEqual(len(command_runner.calls), 1)
            command, cwd, env, timeout = command_runner.calls[0]
            self.assertEqual(
                command,
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/acme/widget.git",
                    target_path,
                ],
            )
            self.assertEqual(cwd, os.getcwd())
            self.assertEqual(timeout, DEFAULT_GIT_TIMEOUT_SECONDS)
            self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")

    @patch("swe_agent.ingestion.clone.shutil.rmtree")
    def test_removes_existing_clone_before_cloning(self, rmtree) -> None:
        command_runner = RecordingCommandRunner(
            [CommandResult(command=[], return_code=0)]
        )
        git_runner = GitRunner(command_runner=command_runner)

        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace_dir = os.path.join(temporary_dir, "workspace")
            target_path = os.path.join(workspace_dir, "acme__widget")
            os.makedirs(target_path)

            clone_repository(
                "acme",
                "widget",
                workspace_dir,
                git_runner=git_runner,
            )

            rmtree.assert_called_once_with(target_path)

    def test_reports_git_error_when_clone_fails(self) -> None:
        command_runner = RecordingCommandRunner(
            [
                CommandResult(
                    command=[],
                    return_code=128,
                    stderr="fatal: repository not found",
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(
                ValueError, "fatal: repository not found"
            ):
                clone_repository(
                    "missing",
                    "repository",
                    temporary_dir,
                    git_runner=GitRunner(command_runner=command_runner),
                )

    def test_reports_timeout_when_clone_hangs(self) -> None:
        command_runner = RecordingCommandRunner(
            [
                CommandResult(
                    command=[],
                    return_code=None,
                    timed_out=True,
                )
            ]
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "timed out"):
                clone_repository(
                    "acme",
                    "widget",
                    temporary_dir,
                    git_runner=GitRunner(command_runner=command_runner),
                )


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

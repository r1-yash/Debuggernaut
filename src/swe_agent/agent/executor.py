"""Apply proposed fixes and run repository test commands."""

from __future__ import annotations

import os
from typing import Callable

from pydantic import BaseModel

from swe_agent.agent.fixer import FixResult
from swe_agent.agent.guardrails import RejectedProposalError, proposal_rejection
from swe_agent.agent.paths import repository_root, resolve_repository_path
from swe_agent.process import CommandResult, CommandRunner, SubprocessRunner


DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
_TEST_TIMEOUT_SECONDS = DEFAULT_TEST_TIMEOUT_SECONDS

TestCommandRunner = Callable[[list[str]], CommandResult]


class TestOutcome(BaseModel):
    """The captured result of running a repository test command."""

    passed: bool
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    started: bool = True
    timeout_seconds: float | None = None

    @classmethod
    def from_command_result(
        cls,
        result: CommandResult,
        timeout_seconds: float,
    ) -> TestOutcome:
        """Convert a shared command result into an agent test outcome."""
        return cls(
            passed=result.succeeded,
            return_code=-1 if result.timed_out else result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            started=not (result.return_code is None and not result.timed_out),
            timeout_seconds=timeout_seconds if result.timed_out else None,
        )


def apply_fix(
    repo_path: str,
    fix_result: FixResult,
    test_command: list[str] | None = None,
    protected_paths: list[str] | None = None,
) -> None:
    """Overwrite the proposed file, refusing missing and protected test files."""
    rejection = proposal_rejection(
        repo_path, fix_result.file_path, test_command, protected_paths
    )
    if rejection is not None:
        kind, reason = rejection
        raise RejectedProposalError(fix_result.file_path, kind, reason)
    path = resolve_repository_path(repo_path, fix_result.file_path)
    path.write_text(fix_result.new_content, encoding="utf-8")


def revert_fix(repo_path: str, fix_result: FixResult) -> None:
    """Restore the original contents for a previously applied proposed fix."""
    path = resolve_repository_path(repo_path, fix_result.file_path)
    path.write_text(fix_result.original_content, encoding="utf-8")


def run_tests(
    repo_path: str,
    test_command: list[str],
    *,
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    test_runner: TestCommandRunner | None = None,
    command_runner: CommandRunner | None = None,
) -> TestOutcome:
    """Run a test command and return its captured result without raising.

    Benchmark callers inject ``test_runner`` so the command runs inside the
    entry's isolated environment.  The host fallback also uses the shared
    process-group runner so a timeout kills the entire test process tree.
    """
    repository_root(repo_path)
    if not test_command:
        raise ValueError("Test command must not be empty.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")

    try:
        if test_runner is not None:
            result = test_runner(list(test_command))
        else:
            runner = command_runner or SubprocessRunner()
            result = runner.run(
                list(test_command),
                cwd=repo_path,
                env=dict(os.environ),
                timeout=timeout_seconds,
            )
    except OSError as error:
        return TestOutcome(
            passed=False,
            return_code=None,
            stdout="",
            stderr=f"{type(error).__name__}: {error}",
            started=False,
            timeout_seconds=None,
        )

    return TestOutcome.from_command_result(result, timeout_seconds)


def apply_and_test(
    repo_path: str,
    fix_result: FixResult,
    test_command: list[str],
    protected_paths: list[str] | None = None,
    *,
    timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    test_runner: TestCommandRunner | None = None,
) -> TestOutcome:
    """Apply a fix, reverting it whenever its test command does not pass."""
    apply_fix(repo_path, fix_result, test_command, protected_paths)
    outcome = run_tests(
        repo_path,
        test_command,
        timeout_seconds=timeout_seconds,
        test_runner=test_runner,
    )
    if not outcome.passed:
        revert_fix(repo_path, fix_result)
    return outcome

"""Apply proposed fixes and run repository test commands."""

from __future__ import annotations

import subprocess

from pydantic import BaseModel

from swe_agent.agent.fixer import FixResult, _repository_root, _resolve_repository_path


_TEST_TIMEOUT_SECONDS = 120


class TestOutcome(BaseModel):
    """The captured result of running a repository test command."""

    passed: bool
    return_code: int
    stdout: str
    stderr: str


def apply_fix(repo_path: str, fix_result: FixResult) -> None:
    """Overwrite the proposed repository file with its fixed contents."""
    path = _resolve_repository_path(repo_path, fix_result.file_path)
    path.write_text(fix_result.new_content, encoding="utf-8")


def revert_fix(repo_path: str, fix_result: FixResult) -> None:
    """Restore the original contents for a previously applied proposed fix."""
    path = _resolve_repository_path(repo_path, fix_result.file_path)
    path.write_text(fix_result.original_content, encoding="utf-8")


def run_tests(repo_path: str, test_command: list[str]) -> TestOutcome:
    """Run a test command and return its captured result without raising on timeout."""
    _repository_root(repo_path)
    if not test_command:
        raise ValueError("Test command must not be empty.")

    try:
        result = subprocess.run(
            test_command,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=_TEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        return TestOutcome(
            passed=False,
            return_code=-1,
            stdout=stdout,
            stderr=(
                f"Test command timed out after {_TEST_TIMEOUT_SECONDS} seconds."
            ),
        )

    return TestOutcome(
        passed=result.returncode == 0,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def apply_and_test(
    repo_path: str,
    fix_result: FixResult,
    test_command: list[str],
) -> TestOutcome:
    """Apply a fix, reverting it whenever its test command does not pass."""
    apply_fix(repo_path, fix_result)
    outcome = run_tests(repo_path, test_command)
    if not outcome.passed:
        revert_fix(repo_path, fix_result)
    return outcome

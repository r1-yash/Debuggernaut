"""Apply benchmark test patches and record a local diff baseline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from swe_agent.agent.paths import repository_root
from swe_agent.git_runner import (
    DEFAULT_PATCH_GIT_TIMEOUT_SECONDS,
    GitRunner,
    git_failure_message,
)
from swe_agent.process import CommandResult


_BASELINE_MESSAGE = "Apply benchmark test patch"
_BASELINE_AUTHOR_NAME = "Debuggernaut Benchmark"
_BASELINE_AUTHOR_EMAIL = "benchmark@invalid"


class TestPatchResult(BaseModel):
    """The outcome of applying one benchmark entry's test patch."""

    applied: bool
    file_paths: list[str] = []
    baseline_commit: str | None = None


def _run_git(
    repo_path: str,
    arguments: list[str],
    git_runner: GitRunner,
) -> CommandResult:
    """Run a git command in a repository through the timed runner."""
    return git_runner.run(arguments, cwd=repo_path)


def _failure(label: str, result: CommandResult) -> ValueError:
    """Return a consistent test-patch failure while preserving timeouts."""
    return ValueError(f"Test patch failed to apply: {git_failure_message(label, result)}")


def apply_test_patch(
    repo_path: str,
    patch_path: str,
    *,
    git_runner: GitRunner | None = None,
    timeout_seconds: float = DEFAULT_PATCH_GIT_TIMEOUT_SECONDS,
) -> TestPatchResult:
    """Apply a test patch and commit it as the baseline for agent changes.

    The patch must apply cleanly; there is no three-way or whitespace-fuzzy
    fallback. The staged patch is committed locally so git-based checks compare
    agent changes against the post-patch tree rather than the pinned commit.

    Every git command runs through the shared timed, process-group-aware
    runner.
    """
    repository_root(repo_path)
    resolved_patch = Path(patch_path).resolve()
    if not resolved_patch.is_file():
        raise ValueError(f"Test patch file does not exist: {patch_path}")
    if not resolved_patch.read_text(encoding="utf-8", errors="replace").strip():
        raise ValueError("Test patch did not change any files.")

    runner = git_runner or GitRunner(timeout_seconds=timeout_seconds)
    check = _run_git(
        repo_path,
        ["apply", "--check", "--whitespace=nowarn", "--", str(resolved_patch)],
        runner,
    )
    if not check.succeeded:
        raise _failure("git apply --check", check)

    apply = _run_git(
        repo_path,
        [
            "apply", "--index", "--whitespace=nowarn", "--",
            str(resolved_patch),
        ],
        runner,
    )
    if not apply.succeeded:
        raise _failure("git apply", apply)

    staged = _run_git(repo_path, ["diff", "--cached", "--name-only", "-z"], runner)
    if not staged.succeeded:
        raise ValueError(
            "Could not list test patch files: "
            f"{git_failure_message('git diff --cached', staged)}"
        )
    file_paths = [path for path in staged.stdout.split("\0") if path]
    if not file_paths:
        raise ValueError("Test patch did not change any files.")

    commit = _run_git(
        repo_path,
        [
            "-c", f"user.name={_BASELINE_AUTHOR_NAME}",
            "-c", f"user.email={_BASELINE_AUTHOR_EMAIL}",
            "-c", "commit.gpgsign=false",
            "commit", "--no-verify", "-m", _BASELINE_MESSAGE,
        ],
        runner,
    )
    if not commit.succeeded:
        raise ValueError(
            "Could not commit the test patch baseline: "
            f"{git_failure_message('git commit', commit)}"
        )

    baseline = _run_git(repo_path, ["rev-parse", "HEAD"], runner)
    if not baseline.succeeded:
        raise ValueError(
            "Could not resolve the test patch baseline commit: "
            f"{git_failure_message('git rev-parse', baseline)}"
        )
    return TestPatchResult(
        applied=True,
        file_paths=file_paths,
        baseline_commit=baseline.stdout.strip(),
    )

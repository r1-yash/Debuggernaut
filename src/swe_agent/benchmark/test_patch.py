"""Apply benchmark test patches and record a local diff baseline."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from swe_agent.agent.paths import repository_root


_BASELINE_MESSAGE = "Apply benchmark test patch"
_BASELINE_AUTHOR_NAME = "Debuggernaut Benchmark"
_BASELINE_AUTHOR_EMAIL = "benchmark@invalid"


class TestPatchResult(BaseModel):
    """The outcome of applying one benchmark entry's test patch."""

    applied: bool
    file_paths: list[str] = []
    baseline_commit: str | None = None


def _run_git(
    repo_path: str, arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run a git command in a repository without raising."""
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )


def apply_test_patch(repo_path: str, patch_path: str) -> TestPatchResult:
    """Apply a test patch and commit it as the baseline for agent changes.

    The patch must apply cleanly; there is no three-way or whitespace-fuzzy
    fallback. The staged patch is committed locally so git-based checks compare
    agent changes against the post-patch tree rather than the pinned commit.
    """
    repository_root(repo_path)
    resolved_patch = Path(patch_path).resolve()
    if not resolved_patch.is_file():
        raise ValueError(f"Test patch file does not exist: {patch_path}")
    if not resolved_patch.read_text(encoding="utf-8", errors="replace").strip():
        raise ValueError("Test patch did not change any files.")

    check = _run_git(
        repo_path,
        ["apply", "--check", "--whitespace=nowarn", "--", str(resolved_patch)],
    )
    if check.returncode != 0:
        raise ValueError(
            f"Test patch failed to apply: {check.stderr.strip()}"
        )

    apply = _run_git(
        repo_path,
        [
            "apply", "--index", "--whitespace=nowarn", "--",
            str(resolved_patch),
        ],
    )
    if apply.returncode != 0:
        raise ValueError(
            f"Test patch failed to apply: {apply.stderr.strip()}"
        )

    staged = _run_git(repo_path, ["diff", "--cached", "--name-only", "-z"])
    if staged.returncode != 0:
        raise ValueError(
            f"Could not list test patch files: {staged.stderr.strip()}"
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
    )
    if commit.returncode != 0:
        raise ValueError(
            f"Could not commit the test patch baseline: {commit.stderr.strip()}"
        )

    baseline = _run_git(repo_path, ["rev-parse", "HEAD"])
    if baseline.returncode != 0:
        raise ValueError(
            "Could not resolve the test patch baseline commit: "
            f"{baseline.stderr.strip()}"
        )
    return TestPatchResult(
        applied=True,
        file_paths=file_paths,
        baseline_commit=baseline.stdout.strip(),
    )

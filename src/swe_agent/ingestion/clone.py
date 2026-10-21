"""Clone GitHub repositories into a local workspace for ingestion."""

from __future__ import annotations

import os
import shutil
import subprocess


def _run_git(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command without raising, capturing its output."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )


def _clone_head(target_path: str, remote_url: str) -> None:
    """Shallow-clone the remote's default branch at its latest commit."""
    result = _run_git(
        ["git", "clone", "--depth", "1", remote_url, target_path]
    )
    if result.returncode != 0:
        raise ValueError(
            f"Git clone failed with exit code {result.returncode}: {result.stderr.strip()}"
        )


def _checkout_commit(target_path: str, remote_url: str, commit: str) -> None:
    """Fetch exactly one commit and check it out, preferring a shallow fetch."""
    os.makedirs(target_path, exist_ok=True)
    if _run_git(["git", "init", target_path]).returncode != 0:
        raise ValueError(f"Could not initialize a git repository at {target_path}.")
    _run_git(["git", "-C", target_path, "remote", "add", "origin", remote_url])

    result = _run_git(
        ["git", "-C", target_path, "fetch", "--depth", "1", "origin", commit]
    )
    if result.returncode != 0:
        # Some remotes refuse shallow fetches of a bare SHA; fall back to a
        # full fetch, which always contains the requested commit.
        result = _run_git(["git", "-C", target_path, "fetch", "origin", commit])
    if result.returncode != 0:
        raise ValueError(
            f"Could not fetch commit {commit} from {remote_url} "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )

    result = _run_git(["git", "-C", target_path, "checkout", "--force", commit])
    if result.returncode != 0:
        raise ValueError(
            f"Could not check out commit {commit} "
            f"(exit code {result.returncode}): {result.stderr.strip()}"
        )


def clone_repository(
    owner: str,
    repository: str,
    workspace_dir: str = "workspace",
    commit: str | None = None,
    remote_url: str | None = None,
) -> str:
    """Clone a GitHub repository and return its absolute local path.

    When ``commit`` is given, the repository is checked out at exactly that
    commit (a shallow single-commit fetch, falling back to a full fetch);
    otherwise the default branch is shallow-cloned at its latest commit.
    ``remote_url`` overrides the derived GitHub URL, primarily for tests.
    """
    if not owner.strip() or not repository.strip():
        raise ValueError("GitHub owner and repository must be provided.")

    if remote_url is None:
        remote_url = f"https://github.com/{owner}/{repository}.git"
    target_path = os.path.join(workspace_dir, f"{owner}__{repository}")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    os.makedirs(workspace_dir, exist_ok=True)
    if commit is None:
        _clone_head(target_path, remote_url)
    else:
        _checkout_commit(target_path, remote_url, commit)

    return os.path.abspath(target_path)

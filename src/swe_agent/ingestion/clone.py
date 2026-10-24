"""Clone GitHub repositories into a local workspace for ingestion."""

from __future__ import annotations

import os
import shutil

from swe_agent.git_runner import (
    DEFAULT_GIT_TIMEOUT_SECONDS,
    GitRunner,
    git_failure_message,
)


def _clone_head(
    target_path: str,
    remote_url: str,
    git_runner: GitRunner,
) -> None:
    """Shallow-clone the remote's default branch at its latest commit."""
    result = git_runner.run(
        ["clone", "--depth", "1", remote_url, target_path],
        cwd=os.getcwd(),
    )
    if not result.succeeded:
        raise ValueError(git_failure_message("Git clone", result))


def _checkout_commit(
    target_path: str,
    remote_url: str,
    commit: str,
    git_runner: GitRunner,
) -> None:
    """Fetch exactly one commit and check it out, preferring a shallow fetch."""
    os.makedirs(target_path, exist_ok=True)
    init = git_runner.run(["init", target_path], cwd=os.getcwd())
    if not init.succeeded:
        raise ValueError(git_failure_message("Git init", init))
    remote = git_runner.run(
        ["-C", target_path, "remote", "add", "origin", remote_url],
        cwd=os.getcwd(),
    )
    if not remote.succeeded:
        raise ValueError(git_failure_message("Git remote add", remote))

    result = git_runner.run(
        ["-C", target_path, "fetch", "--depth", "1", "origin", commit],
        cwd=os.getcwd(),
    )
    if not result.succeeded:
        # Some remotes refuse shallow fetches of a bare SHA; fall back to a
        # full fetch, which always contains the requested commit.
        result = git_runner.run(
            ["-C", target_path, "fetch", "origin", commit],
            cwd=os.getcwd(),
        )
    if not result.succeeded:
        raise ValueError(
            f"Could not fetch commit {commit} from {remote_url} "
            f"({git_failure_message('Git fetch', result)}): "
            f"{result.stderr.strip()}"
        )

    checkout = git_runner.run(
        ["-C", target_path, "checkout", "--force", commit],
        cwd=os.getcwd(),
    )
    if not checkout.succeeded:
        raise ValueError(
            f"Could not check out commit {commit} "
            f"({git_failure_message('Git checkout', checkout)}): "
            f"{checkout.stderr.strip()}"
        )


def clone_repository(
    owner: str,
    repository: str,
    workspace_dir: str = "workspace",
    commit: str | None = None,
    remote_url: str | None = None,
    *,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
    git_runner: GitRunner | None = None,
) -> str:
    """Clone a GitHub repository and return its absolute local path.

    When ``commit`` is given, the repository is checked out at exactly that
    commit (a shallow single-commit fetch, falling back to a full fetch);
    otherwise the default branch is shallow-cloned at its latest commit.
    ``remote_url`` overrides the derived GitHub URL, primarily for tests.

    Git commands run through a timed, process-group-aware runner so a hung
    clone cannot keep the benchmark alive.
    """
    if not owner.strip() or not repository.strip():
        raise ValueError("GitHub owner and repository must be provided.")

    if remote_url is None:
        remote_url = f"https://github.com/{owner}/{repository}.git"
    target_path = os.path.join(workspace_dir, f"{owner}__{repository}")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    os.makedirs(workspace_dir, exist_ok=True)
    runner = git_runner or GitRunner(timeout_seconds=timeout_seconds)
    if commit is None:
        _clone_head(target_path, remote_url, runner)
    else:
        _checkout_commit(target_path, remote_url, commit, runner)

    return os.path.abspath(target_path)

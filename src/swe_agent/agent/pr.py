"""Create GitHub pull requests for successful SWE Agent fixes."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import requests
from pydantic import BaseModel

from swe_agent.agent.loop import LoopResult


_GITHUB_API_URL = "https://api.github.com"
_REQUEST_TIMEOUT_SECONDS = 10


class PullRequestResult(BaseModel):
    """The GitHub pull request created for a successful fix."""

    url: str
    number: int
    branch_name: str


def _github_headers() -> dict[str, str]:
    """Build GitHub request headers, adding authentication when configured."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_error_message(response: requests.Response) -> str:
    """Extract GitHub's error message, falling back to response text."""
    message = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"]
    return message


def _run_git(repo_path: str, command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command and raise a readable error when it fails."""
    result = subprocess.run(
        command,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or f"Git command failed: {' '.join(command)}")
    return result


def _remove_fork_remote(repo_path: str) -> None:
    """Remove a prior SWE Agent fork remote without treating absence as an error."""
    subprocess.run(
        ["git", "remote", "remove", "swe-agent-fork"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )


def _validate_worktree(repo_path: str, fixed_file_path: str) -> None:
    """Ensure the only outstanding change is the successful fixed file."""
    status = _run_git(repo_path, ["git", "status", "--porcelain"])
    changed_paths = [line[3:] for line in status.stdout.splitlines() if line]
    if not changed_paths or any(path != fixed_file_path for path in changed_paths):
        raise ValueError(
            "Repository must be clean except for the already-applied fixed file."
        )


def _ensure_fork(owner: str, repository: str) -> str:
    """Create or reuse a fork and return the fork owner's GitHub login."""
    response = requests.post(
        f"{_GITHUB_API_URL}/repos/{owner}/{repository}/forks",
        headers=_github_headers(),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code not in (200, 202):
        raise ValueError(
            f"GitHub API request failed with status {response.status_code}: "
            f"{_github_error_message(response)}"
        )

    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    fork_owner = payload.get("owner") if isinstance(payload, dict) else None
    if isinstance(fork_owner, dict) and isinstance(fork_owner.get("login"), str):
        return fork_owner["login"]

    user_response = requests.get(
        f"{_GITHUB_API_URL}/user",
        headers=_github_headers(),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if user_response.status_code != 200:
        raise ValueError(
            f"GitHub API request failed with status {user_response.status_code}: "
            f"{_github_error_message(user_response)}"
        )
    user_payload: Any = user_response.json()
    if not isinstance(user_payload, dict) or not isinstance(user_payload.get("login"), str):
        raise ValueError("GitHub API returned an unexpected response format.")
    return user_payload["login"]


def _get_default_branch(owner: str, repository: str) -> str:
    """Fetch the repository's default branch from GitHub."""
    response = requests.get(
        f"{_GITHUB_API_URL}/repos/{owner}/{repository}",
        headers=_github_headers(),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise ValueError(
            f"GitHub API request failed with status {response.status_code}: "
            f"{_github_error_message(response)}"
        )
    payload: Any = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("default_branch"), str):
        raise ValueError("GitHub API returned an unexpected response format.")
    return payload["default_branch"]


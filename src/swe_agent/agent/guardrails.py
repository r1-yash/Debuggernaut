"""Protect test files from agent modifications."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from swe_agent.agent.paths import repository_root, resolve_repository_path


RejectionKind = Literal["test_file", "nonexistent_path", "test_patch_file"]

_TEST_DIRECTORY_NAMES = {"test", "tests"}


class RejectedProposalError(ValueError):
    """A proposal rejected before any repository file was modified."""

    def __init__(self, file_path: str, kind: RejectionKind, reason: str) -> None:
        super().__init__(reason)
        self.file_path = file_path
        self.kind = kind
        self.reason = reason


def _normalized(path: Path) -> str:
    """Return a lowercase POSIX form for case-insensitive path comparison."""
    return path.as_posix().lower()


def _candidate_paths(repo_path: str, file_path: str) -> list[Path]:
    """Return resolved and textual forms of a proposed repository path."""
    root = repository_root(repo_path)
    resolved = resolve_repository_path(repo_path, file_path)
    candidates = [resolved.relative_to(root)]
    textual = Path(file_path.replace("\\", "/"))
    if not textual.is_absolute():
        candidates.append(textual)
    return candidates


def _matches_protected_path(
    candidate_paths: list[Path],
    protected_paths: list[str] | None,
) -> bool:
    """Return whether a proposal targets an explicitly protected file."""
    candidates = {_normalized(path) for path in candidate_paths}
    for protected_path in protected_paths or []:
        protected = Path(protected_path.replace("\\", "/"))
        if _normalized(protected) in candidates:
            return True
    return False


def _is_test_path(relative_path: Path) -> bool:
    """Return whether a repository-relative path names a test file."""
    parts = [part.lower() for part in relative_path.parts]
    if any(part in _TEST_DIRECTORY_NAMES for part in parts[:-1]):
        return True
    name = parts[-1] if parts else ""
    return (
        name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def _command_paths(
    repo_path: str,
    test_command: list[str] | None,
) -> list[Path]:
    """Return resolved repository paths named by a test command."""
    root = repository_root(repo_path)
    paths: list[Path] = []
    for token in test_command or []:
        candidate_text = token.split("::", 1)[0]
        if (
            not candidate_text
            or candidate_text.startswith("-")
            or "=" in candidate_text
        ):
            continue
        candidate = (root / candidate_text).resolve()
        if not candidate.exists():
            continue
        try:
            paths.append(candidate.relative_to(root))
        except ValueError:
            continue
    return paths


def _matches_command_path(
    repo_path: str,
    resolved_relative: Path,
    test_command: list[str] | None,
) -> bool:
    """Return whether a path was named by the test command (file or subtree)."""
    root = repository_root(repo_path)
    resolved = _normalized(resolved_relative)
    for command_path in _command_paths(repo_path, test_command):
        candidate = _normalized(command_path)
        if resolved == candidate:
            return True
        if (root / command_path).is_dir() and resolved.startswith(f"{candidate}/"):
            return True
    return False


def protected_path_reason(
    repo_path: str,
    file_path: str,
    test_command: list[str] | None = None,
    protected_paths: list[str] | None = None,
) -> str | None:
    """Return why a proposal may not modify a file, or ``None`` when allowed.

    Protection derived from the test command is the primary rule; directory
    and filename patterns are a backup heuristic. Explicit ``protected_paths``
    (the benchmark test patch's files) are blocked regardless of their name or
    location. A ``testing/`` directory is only protected when the test command
    names it, because it is often a real source package.
    """
    candidate_paths = _candidate_paths(repo_path, file_path)
    if _matches_protected_path(candidate_paths, protected_paths):
        return (
            f"{file_path} is part of the benchmark test patch and may not be "
            "modified; propose a change to non-test source instead."
        )

    if any(_is_test_path(path) for path in candidate_paths):
        return (
            f"{file_path} is a test file and may not be modified; "
            "propose a change to non-test source instead."
        )
    if _matches_command_path(repo_path, candidate_paths[0], test_command):
        return (
            f"{file_path} is named by the test command and may not be modified; "
            "propose a change to non-test source instead."
        )
    return None


def proposal_rejection(
    repo_path: str,
    file_path: str,
    test_command: list[str] | None = None,
    protected_paths: list[str] | None = None,
) -> tuple[RejectionKind, str] | None:
    """Return a rejection kind and reason when a proposal must not be applied."""
    resolved = resolve_repository_path(repo_path, file_path)
    if not resolved.is_file():
        return "nonexistent_path", f"Proposed file does not exist: {file_path}"
    if _matches_protected_path(
        _candidate_paths(repo_path, file_path), protected_paths
    ):
        return "test_patch_file", (
            f"{file_path} is part of the benchmark test patch and may not be "
            "modified; propose a change to non-test source instead."
        )
    reason = protected_path_reason(repo_path, file_path, test_command)
    if reason is not None:
        return "test_file", reason
    return None

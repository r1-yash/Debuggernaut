"""Validate and resolve paths within a cloned repository."""

from __future__ import annotations

from pathlib import Path


def repository_root(repo_path: str) -> Path:
    """Return a validated, resolved repository root."""
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise ValueError("Repository path must be an existing directory.")
    return root

#guardrail
#ensure that the requested path is within the repository root and does not escape to parent directories
#fixer.py also uses this function to ensure that file paths are safe and do not allow path traversal outside the repository.
#exector.py also uses this function to ensure that file paths are safe and do not allow path traversal outside the repository.

def resolve_repository_path(repo_path: str, requested_path: str) -> Path:
    """Resolve a repository-relative path, rejecting paths outside the root."""
    root = repository_root(repo_path)
    candidate = (root / requested_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Path must stay within the repository.") from error
    return candidate

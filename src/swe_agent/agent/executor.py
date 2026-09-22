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



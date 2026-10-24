"""Classify whether an entry's pinned tests reproduce the expected failure."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.environment import EnvironmentSummary


BaselineStatus = Literal[
    "fails_as_expected",
    "passes_unexpectedly",
    "could_not_install",
    "test_timeout",
    "test_patch_failed",
    "could_not_run",
]

BASELINE_STATUS_IS_INFRASTRUCTURE: dict[BaselineStatus, bool] = {
    "fails_as_expected": False,
    "passes_unexpectedly": True,
    "could_not_install": True,
    "test_timeout": True,
    "test_patch_failed": True,
    "could_not_run": True,
}

_OUTPUT_TAIL_LINES = 20
_MODULE_NOT_FOUND = re.compile(
    r"ModuleNotFoundError:\s+No module named\s+['\"]([^'\"]+)['\"]"
)
_SKIPPED_PACKAGE_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    "docs",
    "examples",
    "requirements",
    "test",
    "tests",
}


class BaselineResult(BaseModel):
    """The result of checking one benchmark entry's pinned failing tests."""

    entry: BenchmarkEntry
    status: BaselineStatus
    detail: str
    return_code: int | None
    output_tail: str
    environment: EnvironmentSummary | None = None


def repository_top_level_modules(repo_path: str | Path) -> set[str]:
    """Infer importable top-level modules from the repository layout.

    Packaging metadata is deliberately not parsed.  Top-level packages, direct
    top-level modules, and packages under a conventional ``src`` or ``lib``
    directory are enough to distinguish the repository's own imports from a
    missing third-party dependency.
    """
    repository = Path(repo_path)
    if not repository.is_dir():
        return set()

    modules: set[str] = set()

    def add_directory(path: Path) -> None:
        if not path.is_dir():
            return
        if path.name in _SKIPPED_PACKAGE_DIRECTORIES or path.name.startswith("."):
            return
        if (path / "__init__.py").is_file() or any(path.rglob("*.py")):
            modules.add(path.name)

    for child in repository.iterdir():
        if child.is_dir():
            if child.name in {"src", "lib"}:
                for candidate in child.iterdir():
                    add_directory(candidate)
                    if candidate.is_file() and candidate.suffix == ".py":
                        modules.add(candidate.stem)
            else:
                add_directory(child)
        elif child.is_file() and child.suffix == ".py":
            modules.add(child.stem)
    return modules


def _combined_output(outcome: TestOutcome) -> str:
    return "\n".join(
        part for part in (outcome.stdout.strip(), outcome.stderr.strip()) if part
    )


def output_tail(outcome: TestOutcome) -> str:
    """Return the last few lines of a test run's combined output."""
    return "\n".join(_combined_output(outcome).splitlines()[-_OUTPUT_TAIL_LINES:])


def _timeout_detail(outcome: TestOutcome) -> str:
    if outcome.timeout_seconds is None:
        return "The baseline test command timed out."
    return (
        "The baseline test command timed out after "
        f"{outcome.timeout_seconds:g} s."
    )


def _missing_module(outcome: TestOutcome) -> str | None:
    match = _MODULE_NOT_FOUND.search(_combined_output(outcome))
    return match.group(1) if match is not None else None


def classify_baseline(
    outcome: TestOutcome,
    *,
    own_top_level_modules: Iterable[str] = (),
) -> tuple[BaselineStatus, str]:
    """Classify one baseline test outcome without rerunning any command.

    A missing third-party module means the environment was not installed well
    enough to test the entry.  A missing module that belongs to the repository
    itself is an ordinary expected failure because the pinned commit may be
    intentionally incomplete or tests may rely on the fix being applied.
    """
    if outcome.timed_out:
        return "test_timeout", _timeout_detail(outcome)
    if not outcome.started or outcome.return_code is None:
        detail = outcome.stderr.strip() or "The test process could not be started."
        return "could_not_install", detail
    if outcome.passed:
        return (
            "passes_unexpectedly",
            "The pinned tests already pass; there is nothing to fix.",
        )

    missing_module = _missing_module(outcome)
    if missing_module is not None:
        top_level = missing_module.split(".", maxsplit=1)[0]
        if top_level not in set(own_top_level_modules):
            return (
                "could_not_install",
                "The baseline test run could not import a dependency: "
                f"ModuleNotFoundError: No module named {missing_module!r}.",
            )

    return (
        "fails_as_expected",
        f"The test command exited with code {outcome.return_code}.",
    )


def baseline_from_outcome(
    entry: BenchmarkEntry,
    outcome: TestOutcome,
    repo_path: str | Path,
    *,
    environment: EnvironmentSummary | None = None,
) -> BaselineResult:
    """Build a baseline result for a test outcome from one repository."""
    status, detail = classify_baseline(
        outcome,
        own_top_level_modules=repository_top_level_modules(repo_path),
    )
    return BaselineResult(
        entry=entry,
        status=status,
        detail=detail,
        return_code=outcome.return_code,
        output_tail=output_tail(outcome),
        environment=environment,
    )

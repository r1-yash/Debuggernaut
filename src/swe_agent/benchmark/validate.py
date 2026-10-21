"""Validate benchmark entries by reproducing their pinned tests without an LLM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome, run_tests
from swe_agent.benchmark.config import BenchmarkEntry, load_benchmark_config
from swe_agent.ingestion.clone import clone_repository


_OUTPUT_TAIL_LINES = 10

BaselineStatus = Literal["fails_as_expected", "passes_unexpectedly", "could_not_run"]


class BaselineResult(BaseModel):
    """The result of checking one benchmark entry's pinned failing tests."""

    entry: BenchmarkEntry
    status: BaselineStatus
    detail: str
    return_code: int | None
    output_tail: str


def _is_pytest_command(test_command: list[str]) -> bool:
    """Return whether a test command invokes pytest."""
    return any("pytest" in argument for argument in test_command)


def _output_tail(outcome: TestOutcome) -> str:
    """Return the last few lines of a test run's combined output."""
    combined = "\n".join(
        part for part in (outcome.stdout.strip(), outcome.stderr.strip()) if part
    )
    return "\n".join(combined.splitlines()[-_OUTPUT_TAIL_LINES:])


def _classify(outcome: TestOutcome, test_command: list[str]) -> tuple[BaselineStatus, str]:
    """Decide what a captured test run says about the entry's baseline."""
    if outcome.return_code == -1:
        return "could_not_run", outcome.stderr
    if outcome.passed:
        return (
            "passes_unexpectedly",
            "The pinned tests already pass; there is nothing to fix.",
        )
    if _is_pytest_command(test_command):
        if outcome.return_code == 1:
            return "fails_as_expected", f"Pytest exited with code {outcome.return_code}."
        return "could_not_run", (
            f"Pytest exited with code {outcome.return_code}, which reports a "
            "test-run problem rather than failing tests."
        )
    return (
        "fails_as_expected",
        f"The test command exited with code {outcome.return_code}.",
    )


def validate_entry(
    entry: BenchmarkEntry,
    workspace_dir: str = "workspace",
) -> BaselineResult:
    """Check one entry's pinned tests without making any LLM calls."""
    try:
        repo_path = clone_repository(
            entry.owner, entry.repository, workspace_dir, commit=entry.commit
        )
    except Exception as error:
        return BaselineResult(
            entry=entry,
            status="could_not_run",
            detail=str(error),
            return_code=None,
            output_tail="",
        )

    try:
        outcome = run_tests(repo_path, entry.test_command)
    except FileNotFoundError:
        return BaselineResult(
            entry=entry,
            status="could_not_run",
            detail=f"Test command not found: {entry.test_command[0]!r}.",
            return_code=None,
            output_tail="",
        )

    status, detail = _classify(outcome, entry.test_command)
    return BaselineResult(
        entry=entry,
        status=status,
        detail=detail,
        return_code=outcome.return_code,
        output_tail=_output_tail(outcome),
    )


def validate_benchmark(
    config_path: str,
    workspace_dir: str = "workspace",
) -> list[BaselineResult]:
    """Validate every entry in a benchmark configuration."""
    return [
        validate_entry(entry, workspace_dir)
        for entry in load_benchmark_config(config_path)
    ]


def summarize_baselines(results: list[BaselineResult]) -> str:
    """Return a human-readable report with exit codes and output tails."""
    reproducible = sum(result.status == "fails_as_expected" for result in results)
    lines = [
        f"Total entries: {len(results)}",
        f"Reproduce as failing: {reproducible}",
        f"Not reproducible: {len(results) - reproducible}",
    ]
    for result in results:
        identifier = (
            f"{result.entry.owner}/{result.entry.repository}"
            f"#{result.entry.issue_number}"
        )
        exit_code = "unknown" if result.return_code is None else str(result.return_code)
        lines.append(f"{identifier}: {result.status} (exit code {exit_code})")
        lines.append(f"  {result.detail}")
        if result.output_tail:
            lines.append("  Last test output lines:")
            lines.extend(f"    {line}" for line in result.output_tail.splitlines())
    return "\n".join(lines)

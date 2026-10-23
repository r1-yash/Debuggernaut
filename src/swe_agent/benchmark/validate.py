"""Validate benchmark entries by reproducing their pinned tests without an LLM."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome, run_tests
from swe_agent.benchmark.config import BenchmarkEntry, load_benchmark_config
from swe_agent.benchmark.test_patch import apply_test_patch
from swe_agent.ingestion.clone import clone_repository


_OUTPUT_TAIL_LINES = 20
_SUMMARY_TAIL_LINES = 10
_RESULTS_TAIL_LINES = 20

BaselineStatus = Literal[
    "fails_as_expected",
    "passes_unexpectedly",
    "could_not_run",
    "test_patch_failed",
]


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
    """Check one entry's pinned tests without making any LLM calls.

    When the entry defines a test patch, it is applied before the test command
    runs. A patch that cannot be applied is an infrastructure failure, not a
    statement about whether the pinned tests reproduce the bug.
    """
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

    if entry.test_patch is not None:
        try:
            apply_test_patch(repo_path, entry.test_patch)
        except Exception as error:
            return BaselineResult(
                entry=entry,
                status="test_patch_failed",
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


class ValidationInterrupted(Exception):
    """Carry the completed results out of an interrupted validation run."""

    def __init__(self, results: list[BaselineResult]) -> None:
        super().__init__("Validation interrupted.")
        self.results = results


def _format_progress(
    index: int,
    total: int,
    result: BaselineResult,
    elapsed: float,
) -> str:
    """Format the one-line progress update for a finished entry."""
    entry = result.entry
    return (
        f"[{index}/{total}] {entry.owner}/{entry.repository} "
        f"{result.status} ({elapsed:.1f}s)"
    )


def _last_lines(text: str, count: int) -> str:
    """Keep only the last ``count`` lines of text."""
    return "\n".join(text.splitlines()[-count:])


def _result_record(result: BaselineResult) -> dict[str, object]:
    """Build the JSON-lines record for one completed validation result."""
    entry = result.entry
    return {
        "repo": f"{entry.owner}/{entry.repository}",
        "status": result.status,
        "return_code": result.return_code,
        "detail": result.detail,
        "output_tail": _last_lines(result.output_tail, _RESULTS_TAIL_LINES),
    }


def validate_benchmark(
    config_path: str,
    workspace_dir: str = "workspace",
    *,
    limit: int | None = None,
    results_file: str | Path | None = None,
) -> list[BaselineResult]:
    """Validate benchmark entries, reporting progress after each one."""
    if limit is not None and limit < 1:
        raise ValueError("Validation limit must be positive.")

    entries = load_benchmark_config(config_path)
    if limit is not None:
        entries = entries[:limit]
    total = len(entries)
    results: list[BaselineResult] = []
    results_handle = (
        open(results_file, "a", encoding="utf-8", newline="\n")
        if results_file is not None
        else None
    )
    try:
        for index, entry in enumerate(entries, start=1):
            started = time.monotonic()
            result = validate_entry(entry, workspace_dir)
            elapsed = time.monotonic() - started
            results.append(result)
            if results_handle is not None:
                results_handle.write(
                    json.dumps(_result_record(result), ensure_ascii=False) + "\n"
                )
                results_handle.flush()
            print(_format_progress(index, total, result, elapsed), flush=True)
    except KeyboardInterrupt:
        raise ValidationInterrupted(results) from None
    finally:
        if results_handle is not None:
            results_handle.close()
    return results


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
        tail_lines = result.output_tail.splitlines()[-_SUMMARY_TAIL_LINES:]
        if tail_lines:
            lines.append("  Last test output lines:")
            lines.extend(f"    {line}" for line in tail_lines)
    return "\n".join(lines)

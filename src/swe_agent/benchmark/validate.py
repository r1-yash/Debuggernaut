"""Validate benchmark entries by reproducing their pinned tests without an LLM."""

from __future__ import annotations

import json
import time
from pathlib import Path

from swe_agent.agent.executor import TestOutcome
from swe_agent.benchmark.baseline import (
    BASELINE_STATUS_IS_INFRASTRUCTURE,
    BaselineResult,
    BaselineStatus,
    baseline_from_outcome,
)
from swe_agent.benchmark.config import BenchmarkEntry, load_benchmark_config
from swe_agent.benchmark.environment import (
    DEFAULT_INSTALL_TIMEOUT_SECONDS,
    DEFAULT_SETUP_DEADLINE_SECONDS,
    EntryEnvironment,
    EnvironmentSetupError,
    sweep_stale_workspaces,
)
from swe_agent.benchmark.reporting import SCHEMA_VERSION
from swe_agent.benchmark.test_patch import apply_test_patch
from swe_agent.ingestion.clone import clone_repository


_SUMMARY_TAIL_LINES = 10
_RESULTS_TAIL_LINES = 20
DEFAULT_VALIDATION_TEST_TIMEOUT_SECONDS = 300.0

__all__ = [
    "BaselineResult",
    "BaselineStatus",
    "DEFAULT_VALIDATION_TEST_TIMEOUT_SECONDS",
    "ValidationInterrupted",
    "summarize_baselines",
    "validate_benchmark",
    "validate_entry",
]


def _clone_path(workspace_dir: str, entry: BenchmarkEntry) -> Path:
    """Return the configured clone destination for one entry."""
    return Path(workspace_dir) / f"{entry.owner}__{entry.repository}"


def _next_baseline_result(
    entry: BenchmarkEntry,
    environment: EntryEnvironment,
) -> BaselineResult:
    """Run an entry's pinned tests inside its prepared environment."""
    result = environment.run_test_command(entry.test_command)
    outcome = TestOutcome.from_command_result(
        result,
        environment.test_timeout_seconds,
    )
    return baseline_from_outcome(
        entry,
        outcome,
        environment.repo_path,
        environment=environment.summary,
    )


def validate_entry(
    entry: BenchmarkEntry,
    workspace_dir: str = "workspace",
    *,
    install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    test_timeout_seconds: float = DEFAULT_VALIDATION_TEST_TIMEOUT_SECONDS,
    setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
) -> BaselineResult:
    """Check one entry's pinned tests without making any LLM calls.

    The clone and isolated environment are always removed.  A failure to
    prepare the environment is reported as ``could_not_install`` rather than
    as a statement about whether the pinned tests reproduce the bug.
    """
    clone_path = _clone_path(workspace_dir, entry)
    try:
        repo_path = clone_repository(
            entry.owner, entry.repository, workspace_dir, commit=entry.commit
        )
    except Exception as error:
        sweep_stale_workspaces(workspace_dir, clone_paths=[clone_path])
        return BaselineResult(
            entry=entry,
            status="could_not_run",
            detail=str(error),
            return_code=None,
            output_tail="",
        )

    environment: EntryEnvironment | None = None
    result: BaselineResult | None = None
    try:
        try:
            environment = EntryEnvironment.create(
                entry_id=(
                    f"{entry.owner}/{entry.repository}#{entry.issue_number}"
                ),
                repo_path=repo_path,
                workspace_dir=workspace_dir,
                install_timeout_seconds=install_timeout_seconds,
                test_timeout_seconds=test_timeout_seconds,
                setup_deadline_seconds=setup_deadline_seconds,
            )
        except EnvironmentSetupError as error:
            result = BaselineResult(
                entry=entry,
                status="could_not_install",
                detail=(
                    "Could not prepare the benchmark environment "
                    f"({error.status}): {error}"
                ),
                return_code=None,
                output_tail="",
                environment=error.summary,
            )

        if environment is not None:
            try:
                if entry.test_patch is not None:
                    apply_test_patch(repo_path, entry.test_patch)
            except Exception as error:
                result = BaselineResult(
                    entry=entry,
                    status="test_patch_failed",
                    detail=str(error),
                    return_code=None,
                    output_tail="",
                    environment=environment.summary,
                )
            else:
                try:
                    result = _next_baseline_result(entry, environment)
                except Exception as error:
                    result = BaselineResult(
                        entry=entry,
                        status="could_not_install",
                        detail=(
                            "The isolated environment could not run the tests: "
                            f"{type(error).__name__}: {error}"
                        ),
                        return_code=None,
                        output_tail="",
                        environment=environment.summary,
                    )
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as cleanup_error:
                environment.summary.cleanup_errors.append(
                    "Could not close the benchmark environment: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            if result is not None:
                result.environment = environment.summary

    if result is None:  # pragma: no cover - defensive invariant.
        result = BaselineResult(
            entry=entry,
            status="could_not_run",
            detail="Validation ended without a test result.",
            return_code=None,
            output_tail="",
        )
    return result


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
    """Build the schema-v3 JSON-lines record for one validation result."""
    entry = result.entry
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": f"{entry.owner}/{entry.repository}",
        "status": result.status,
        "failure_class": (
            "infrastructure"
            if BASELINE_STATUS_IS_INFRASTRUCTURE[result.status]
            else None
        ),
        "return_code": result.return_code,
        "detail": result.detail,
        "output_tail": _last_lines(result.output_tail, _RESULTS_TAIL_LINES),
        "environment": (
            result.environment.model_dump(mode="json")
            if result.environment is not None
            else None
        ),
    }


def validate_benchmark(
    config_path: str,
    workspace_dir: str = "workspace",
    *,
    limit: int | None = None,
    results_file: str | Path | None = None,
    install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    test_timeout_seconds: float = DEFAULT_VALIDATION_TEST_TIMEOUT_SECONDS,
    setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
) -> list[BaselineResult]:
    """Validate benchmark entries, reporting progress after each one."""
    if limit is not None and limit < 1:
        raise ValueError("Validation limit must be positive.")

    entries = load_benchmark_config(config_path)
    sweep_errors = sweep_stale_workspaces(
        workspace_dir,
        clone_paths=[_clone_path(workspace_dir, entry) for entry in entries],
    )
    if sweep_errors:
        raise ValueError(
            "Could not clear stale validation workspaces: "
            + "; ".join(sweep_errors)
        )
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
            result = validate_entry(
                entry,
                workspace_dir,
                install_timeout_seconds=install_timeout_seconds,
                test_timeout_seconds=test_timeout_seconds,
                setup_deadline_seconds=setup_deadline_seconds,
            )
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
        if result.environment is not None and result.environment.cleanup_errors:
            lines.append("  Cleanup errors:")
            lines.extend(
                f"    {error}" for error in result.environment.cleanup_errors
            )
    return "\n".join(lines)

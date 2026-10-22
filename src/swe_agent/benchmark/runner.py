"""Run benchmark entries through cloning, fixing, and optional PR creation."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel

from swe_agent.agent.loop import LoopResult, resolve_issue
from swe_agent.agent.pr import PullRequestResult, create_fix_pull_request
from swe_agent.benchmark.config import BenchmarkEntry, load_benchmark_config
from swe_agent.benchmark.reporting import (
    RunStatus,
    RunTiming,
    write_benchmark_summary,
    write_run_report,
)
from swe_agent.benchmark.test_patch import TestPatchResult, apply_test_patch
from swe_agent.ingestion.clone import clone_repository
from swe_agent.ingestion.issues import fetch_issue
from swe_agent.llm import UsageTracker, coerce_provider


class BenchmarkOutcome(BaseModel):
    """The result of processing one benchmark entry."""

    entry: BenchmarkEntry
    loop_result: LoopResult
    pr_result: PullRequestResult | None
    error: str | None
    report_path: str | None = None
    test_patch: TestPatchResult = TestPatchResult(applied=False)


def _error_outcome(entry: BenchmarkEntry, error: Exception) -> BenchmarkOutcome:
    """Return a failure outcome when setup or resolution raises an exception."""
    return BenchmarkOutcome(
        entry=entry,
        loop_result=LoopResult(succeeded=False, final_attempt=None, attempts=[]),
        pr_result=None,
        error=str(error),
    )


def _status(outcome: BenchmarkOutcome) -> RunStatus:
    """Classify an outcome as success, failed, or an infrastructure error."""
    if outcome.loop_result.succeeded:
        return "success"
    return "errored" if outcome.error is not None else "failed"


def run_single_entry(
    entry: BenchmarkEntry,
    client: object,
    workspace_dir: str = "workspace",
    open_pr: bool = False,
    report_dir: str = "runs",
) -> BenchmarkOutcome:
    """Run one benchmark entry, retaining errors instead of raising them."""
    tracker = UsageTracker()
    started_at = datetime.now(UTC)
    started = time.monotonic()
    patch_result = TestPatchResult(applied=False)
    try:
        repo_path = clone_repository(
            entry.owner, entry.repository, workspace_dir, commit=entry.commit
        )
        if entry.test_patch is not None:
            patch_result = apply_test_patch(repo_path, entry.test_patch)
        issue = fetch_issue(entry.owner, entry.repository, entry.issue_number)
        loop_result = resolve_issue(
            repo_path,
            issue.title,
            issue.body or "",
            entry.test_command,
            client,
            usage_tracker=tracker,
            protected_paths=patch_result.file_paths,
        )
        pr_result = None
        if loop_result.succeeded and open_pr:
            pr_result = create_fix_pull_request(
                repo_path,
                entry.owner,
                entry.repository,
                entry.issue_number,
                loop_result,
            )
        outcome = BenchmarkOutcome(
            entry=entry,
            loop_result=loop_result,
            pr_result=pr_result,
            error=None,
            test_patch=patch_result,
        )
    except Exception as error:
        outcome = _error_outcome(entry, error)
        outcome.loop_result.llm_calls = tracker.calls
        outcome.test_patch = patch_result
    timing = RunTiming(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_seconds=time.monotonic() - started,
    )
    provider = coerce_provider(client)
    path = write_run_report(
        report_dir, entry,
        provider.provider, provider.model, outcome.loop_result,
        _status(outcome),
        outcome.pr_result.url if outcome.pr_result else None,
        timing,
        error=outcome.error,
        test_patch=patch_result,
    )
    outcome.report_path = str(path)
    return outcome


def run_benchmark(
    config_path: str,
    client: object,
    workspace_dir: str = "workspace",
    open_pr: bool = False,
    report_dir: str = "runs",
) -> list[BenchmarkOutcome]:
    """Run all configured benchmark entries in sequence and write a summary."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    outcomes = [
        run_single_entry(entry, client, workspace_dir, open_pr, report_dir)
        for entry in load_benchmark_config(config_path)
    ]
    timing = RunTiming(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_seconds=time.monotonic() - started,
    )
    write_benchmark_summary(
        report_dir,
        [
            (outcome.report_path, _status(outcome))
            for outcome in outcomes
            if outcome.report_path is not None
        ],
        timing,
    )
    return outcomes


def summarize_outcomes(outcomes: list[BenchmarkOutcome]) -> str:
    """Return a concise human-readable report for benchmark outcomes."""
    succeeded = sum(outcome.loop_result.succeeded for outcome in outcomes)
    errored = sum(outcome.error is not None for outcome in outcomes)
    lines = [
        f"Total entries: {len(outcomes)}",
        f"Succeeded: {succeeded}",
        f"Errored: {errored}",
    ]
    for outcome in outcomes:
        identifier = (
            f"{outcome.entry.owner}/{outcome.entry.repository}"
            f"#{outcome.entry.issue_number}"
        )
        if outcome.error is not None:
            result = f"errored: {outcome.error}"
        elif outcome.loop_result.succeeded:
            result = "succeeded"
        else:
            result = f"failed after {len(outcome.loop_result.attempts)} attempts"
        lines.append(f"{identifier}: {result}")
    return "\n".join(lines)

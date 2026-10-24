"""Run benchmark entries through cloning, fixing, and optional PR creation."""

from __future__ import annotations

import time
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.loop import LoopResult, resolve_issue
from swe_agent.agent.pr import PullRequestResult, create_fix_pull_request
from swe_agent.benchmark.baseline import BaselineResult, baseline_from_outcome
from swe_agent.benchmark.config import BenchmarkEntry, load_benchmark_config
from swe_agent.benchmark.environment import (
    DEFAULT_INSTALL_TIMEOUT_SECONDS,
    DEFAULT_SETUP_DEADLINE_SECONDS,
    DEFAULT_TEST_TIMEOUT_SECONDS,
    EntryEnvironment,
    EnvironmentSetupError,
    EnvironmentSummary,
    sweep_stale_workspaces,
)
from swe_agent.benchmark.reporting import (
    FailureClass,
    RunStatus,
    RunTiming,
    write_benchmark_summary,
    write_run_report,
)
from swe_agent.benchmark.test_patch import TestPatchResult, apply_test_patch
from swe_agent.ingestion.clone import clone_repository
from swe_agent.ingestion.issues import fetch_issue
from swe_agent.llm import (
    ContextTooLarge,
    MalformedProviderResponse,
    ProviderUnavailable,
    UsageTracker,
    coerce_provider,
)


class BenchmarkOutcome(BaseModel):
    """The result of processing one benchmark entry."""

    entry: BenchmarkEntry
    loop_result: LoopResult
    pr_result: PullRequestResult | None
    error: str | None
    status: RunStatus = "errored"
    failure_class: FailureClass | None = None
    status_reason: str | None = None
    baseline: BaselineResult | None = None
    environment: EnvironmentSummary | None = None
    report_path: str | None = None
    test_patch: TestPatchResult = TestPatchResult(applied=False)


def _status(outcome: BenchmarkOutcome) -> RunStatus:
    """Return the explicit status assigned while processing an entry."""
    return outcome.status


def _empty_loop_result() -> LoopResult:
    return LoopResult(succeeded=False, final_attempt=None, attempts=[])


def _baseline_outcome(
    baseline: BaselineResult,
) -> tuple[RunStatus, FailureClass, str]:
    """Translate a non-proceeding baseline gate into a run status and reason."""
    if baseline.status == "passes_unexpectedly":
        return (
            "errored",
            "infrastructure",
            f"Baseline passed unexpectedly: {baseline.detail}",
        )
    if baseline.status == "could_not_install":
        return "could_not_install", "infrastructure", baseline.detail
    if baseline.status == "test_timeout":
        return (
            "errored",
            "infrastructure",
            f"Baseline tests timed out: {baseline.detail}",
        )
    return (
        "errored",
        "infrastructure",
        f"Baseline did not reproduce the expected failure ({baseline.status}): "
        f"{baseline.detail}",
    )


def _agent_timeout_reason(loop_result: LoopResult) -> str | None:
    """Describe the last timed-out agent test run, if there was one."""
    for attempt in reversed(loop_result.attempts):
        outcome = attempt.outcome
        if not outcome.timed_out:
            continue
        if outcome.timeout_seconds is None:
            return "Agent tests timed out."
        return f"Agent tests timed out after {outcome.timeout_seconds:g} s."
    return None


def _clone_path(workspace_dir: str, entry: BenchmarkEntry) -> Path:
    """Return the configured clone destination for one entry."""
    return Path(workspace_dir) / f"{entry.owner}__{entry.repository}"


def run_single_entry(
    entry: BenchmarkEntry,
    client: object,
    workspace_dir: str = "workspace",
    open_pr: bool = False,
    report_dir: str = "runs",
    *,
    install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
) -> BenchmarkOutcome:
    """Run one benchmark entry, retaining errors instead of raising them.

    Lifecycle: clone, create the isolated environment, apply any test patch,
    check the baseline, fetch the issue, run the agent, optionally open a PR,
    then close the environment before writing the report.
    """
    if open_pr and entry.test_patch is not None:
        raise ValueError(
            "A benchmark entry with a test patch cannot open a pull request: "
            f"{entry.owner}/{entry.repository}#{entry.issue_number}. "
            "Run without --open-pr."
        )
    tracker = UsageTracker()
    started_at = datetime.now(UTC)
    started = time.monotonic()
    patch_result = TestPatchResult(applied=False)
    clone_path = _clone_path(workspace_dir, entry)
    clone_attempted = False
    environment: EntryEnvironment | None = None
    environment_summary: EnvironmentSummary | None = None
    baseline: BaselineResult | None = None
    loop_result = _empty_loop_result()
    pr_result: PullRequestResult | None = None
    error: str | None = None
    status: RunStatus = "errored"
    failure_class: FailureClass | None = "infrastructure"
    status_reason: str | None = None

    try:
        clone_attempted = True
        repo_path = clone_repository(
            entry.owner, entry.repository, workspace_dir, commit=entry.commit
        )
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
        if entry.test_patch is not None:
            patch_result = apply_test_patch(repo_path, entry.test_patch)

        baseline_command = environment.run_test_command(entry.test_command)
        baseline = baseline_from_outcome(
            entry,
            TestOutcome.from_command_result(
                baseline_command,
                environment.test_timeout_seconds,
            ),
            repo_path,
            environment=environment.summary,
        )
        if baseline.status != "fails_as_expected":
            status, failure_class, status_reason = _baseline_outcome(baseline)
        else:
            issue = fetch_issue(
                entry.owner, entry.repository, entry.issue_number
            )
            loop_result = resolve_issue(
                repo_path,
                issue.title,
                issue.body or "",
                entry.test_command,
                client,
                usage_tracker=tracker,
                protected_paths=patch_result.file_paths,
                test_runner=environment.run_test_command,
                test_timeout_seconds=environment.test_timeout_seconds,
            )
            if loop_result.succeeded:
                status = "success"
                failure_class = None
                if open_pr:
                    pr_result = create_fix_pull_request(
                        repo_path,
                        entry.owner,
                        entry.repository,
                        entry.issue_number,
                        loop_result,
                        test_patch_applied=patch_result.applied,
                    )
            else:
                timeout_reason = _agent_timeout_reason(loop_result)
                if timeout_reason is not None:
                    status = "test_timeout"
                    failure_class = "agent"
                    status_reason = timeout_reason
                else:
                    status = "failed"
                    failure_class = "agent"
    except ContextTooLarge as caught_error:
        error = str(caught_error)
        status = "context_too_large"
        failure_class = "agent"
        status_reason = str(caught_error)
        loop_result.llm_calls = tracker.calls
    except MalformedProviderResponse as caught_error:
        error = str(caught_error)
        status = "failed"
        failure_class = "agent"
        status_reason = str(caught_error)
        loop_result.llm_calls = tracker.calls
    except ProviderUnavailable as caught_error:
        error = str(caught_error)
        status = "provider_unavailable"
        failure_class = "infrastructure"
        status_reason = str(caught_error)
        loop_result.llm_calls = tracker.calls
    except EnvironmentSetupError as setup_error:
        environment_summary = setup_error.summary
        error = str(setup_error)
        status = (
            "could_not_install"
            if setup_error.status == "could_not_install"
            else "errored"
        )
        failure_class = "infrastructure"
        status_reason = setup_error.reason
        loop_result.llm_calls = tracker.calls
    except Exception as caught_error:
        error = str(caught_error)
        status = "errored"
        failure_class = "infrastructure"
        status_reason = str(caught_error)
        loop_result.llm_calls = tracker.calls
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as cleanup_error:
                environment.summary.cleanup_errors.append(
                    "Could not close the benchmark environment: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            environment_summary = environment.summary
            if baseline is not None:
                baseline.environment = environment.summary
        elif clone_attempted:
            # A failed clone or environment setup can leave a partial clone.
            sweep_stale_workspaces(
                workspace_dir,
                clone_paths=[clone_path],
            )

    timing = RunTiming(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_seconds=time.monotonic() - started,
    )
    provider = coerce_provider(client)
    path = write_run_report(
        report_dir,
        entry,
        provider.provider,
        provider.model,
        loop_result,
        status,
        pr_result.url if pr_result else None,
        timing,
        error=error,
        test_patch=patch_result,
        base_url_host=getattr(provider, "base_url_host", None),
        status_reason=status_reason,
        baseline=baseline,
        environment=environment_summary,
        failure_class=failure_class,
    )
    return BenchmarkOutcome(
        entry=entry,
        loop_result=loop_result,
        pr_result=pr_result,
        error=error,
        status=status,
        failure_class=failure_class,
        status_reason=status_reason,
        baseline=baseline,
        environment=environment_summary,
        report_path=str(path),
        test_patch=patch_result,
    )


def run_benchmark(
    config_path: str,
    client: object,
    workspace_dir: str = "workspace",
    open_pr: bool = False,
    report_dir: str = "runs",
    *,
    install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
    retry_errored: bool = False,
) -> list[BenchmarkOutcome]:
    """Run all configured benchmark entries in sequence and write a summary."""
    entries = load_benchmark_config(config_path)
    if open_pr:
        patched = [entry for entry in entries if entry.test_patch is not None]
        if patched:
            identifiers = ", ".join(
                f"{entry.owner}/{entry.repository}#{entry.issue_number}"
                for entry in patched
            )
            raise ValueError(
                "open_pr is not supported when any benchmark entry defines a "
                f"test patch ({identifiers}); benchmark patches must never be "
                "pushed to an external repository. Run without --open-pr."
            )
    sweep_errors = sweep_stale_workspaces(
        workspace_dir,
        clone_paths=[_clone_path(workspace_dir, entry) for entry in entries],
    )
    if sweep_errors:
        raise ValueError(
            "Could not clear stale benchmark workspaces: "
            + "; ".join(sweep_errors)
        )
    provider = coerce_provider(client)
    previous_results = _resumable_results(
        report_dir, entries, provider, retry_errored=retry_errored
    )
    entries = [
        entry for entry in entries
        if _entry_key(entry) not in previous_results
    ]
    if provider.provider == "tokenharbor":
        try:
            provider.preflight()
        except ProviderUnavailable:
            # There is no entry record to write: no entry was started.
            return []
    started_at = datetime.now(UTC)
    started = time.monotonic()
    outcomes: list[BenchmarkOutcome] = []
    for entry in entries:
        outcome = run_single_entry(
            entry,
            client,
            workspace_dir,
            open_pr,
            report_dir,
            install_timeout_seconds=install_timeout_seconds,
            test_timeout_seconds=test_timeout_seconds,
            setup_deadline_seconds=setup_deadline_seconds,
        )
        outcomes.append(outcome)
        if outcome.status == "provider_unavailable":
            break
    timing = RunTiming(
        started_at=started_at,
        finished_at=datetime.now(UTC),
        duration_seconds=time.monotonic() - started,
    )
    write_benchmark_summary(
        report_dir,
        [
            *previous_results.values(),
            *[
                (outcome.report_path, outcome.status)
                for outcome in outcomes
                if outcome.report_path is not None
            ],
        ],
        timing,
    )
    return outcomes


def _entry_key(entry: BenchmarkEntry) -> tuple[str, int, str]:
    return (f"{entry.owner}/{entry.repository}", entry.issue_number, entry.commit)


def _resumable_results(
    report_dir: str,
    entries: list[BenchmarkEntry],
    provider: object,
    *,
    retry_errored: bool,
) -> dict[tuple[str, int, str], tuple[str, RunStatus]]:
    """Return prior terminal records matching this exact provider configuration."""
    configured = {_entry_key(entry) for entry in entries}
    terminal = {
        "success", "failed", "test_timeout", "context_too_large",
        "errored", "could_not_install", "provider_unavailable",
    }
    if retry_errored:
        terminal -= {"errored", "could_not_install", "provider_unavailable"}
    results: dict[tuple[str, int, str], tuple[str, RunStatus]] = {}
    for path in sorted(Path(report_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (record["repository"], record["issue"], record["commit"])
            status = record["status"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (
            key in configured
            and status in terminal
            and record.get("provider") == getattr(provider, "provider", None)
            and record.get("model") == getattr(provider, "model", None)
            and record.get("base_url_host") == getattr(provider, "base_url_host", None)
        ):
            results[key] = (str(path), status)
    return results


def summarize_outcomes(outcomes: list[BenchmarkOutcome]) -> str:
    """Return a concise human-readable report for benchmark outcomes."""
    counts = {
        status: sum(outcome.status == status for outcome in outcomes)
        for status in (
            "success",
            "failed",
            "test_timeout",
            "context_too_large",
            "could_not_install",
            "provider_unavailable",
            "errored",
        )
    }
    lines = [
        f"Total entries: {len(outcomes)}",
        f"Succeeded: {counts['success']}",
        f"Failed: {counts['failed']}",
        f"Test timeouts: {counts['test_timeout']}",
        f"Context too large: {counts['context_too_large']}",
        f"Could not install: {counts['could_not_install']}",
        f"Provider unavailable: {counts['provider_unavailable']}",
        f"Errored: {counts['errored']}",
    ]
    for outcome in outcomes:
        identifier = (
            f"{outcome.entry.owner}/{outcome.entry.repository}"
            f"#{outcome.entry.issue_number}"
        )
        if outcome.status == "success":
            result = "succeeded"
        else:
            reason = outcome.status_reason or outcome.error
            result = outcome.status
            if reason:
                result = f"{result}: {reason}"
        lines.append(f"{identifier}: {result}")
    return "\n".join(lines)

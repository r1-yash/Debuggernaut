"""Write safe machine-readable results for benchmark runs."""
from __future__ import annotations
import difflib
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult, ProposalResponse
from swe_agent.agent.loop import Attempt, LoopResult
from swe_agent.benchmark.baseline import BaselineResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.environment import EnvironmentSummary
from swe_agent.benchmark.test_patch import TestPatchResult


SCHEMA_VERSION = 4
_ARTIFACT_SCHEMA_VERSION = 1
_DIFF_MAX_CHARS = 8_000
_OUTPUT_TAIL_MAX_LINES = 40
_OUTPUT_TAIL_MAX_CHARS = 8_000
_EXPLORATION_MAX_STEPS = 100
_PROPOSAL_ERROR_MAX_CHARS = 500

RunStatus = Literal[
    "success",
    "failed",
    "test_timeout",
    "context_too_large",
    "could_not_install",
    "provider_unavailable",
    "errored",
]
FailureClass = Literal["agent", "infrastructure"]

RUN_STATUS_FAILURE_CLASS: dict[RunStatus, FailureClass | None] = {
    "success": None,
    "failed": "agent",
    "test_timeout": "agent",
    "context_too_large": "agent",
    "could_not_install": "infrastructure",
    "provider_unavailable": "infrastructure",
    "errored": "infrastructure",
}


class RunTiming(BaseModel):
    """Wall-clock bounds and monotonic duration for one benchmark run."""

    started_at: datetime
    finished_at: datetime
    duration_seconds: float


def _token_total(calls: list[Any], field: str) -> int | None:
    values = [getattr(call, field) for call in calls]
    return sum(values) if values and all(value is not None for value in values) else None


def _diff(fix_result: FixResult) -> tuple[str, bool]:
    """Return a size-capped unified diff of one proposed rewrite."""
    diff = "".join(difflib.unified_diff(
        fix_result.original_content.splitlines(keepends=True),
        fix_result.new_content.splitlines(keepends=True),
        fromfile=f"a/{fix_result.file_path}",
        tofile=f"b/{fix_result.file_path}",
    ))
    if len(diff) <= _DIFF_MAX_CHARS:
        return diff, False
    return diff[:_DIFF_MAX_CHARS], True


def _output_tail(outcome: TestOutcome) -> tuple[str, bool]:
    """Return the last lines of combined test output within line and size caps."""
    combined = "\n".join(
        part for part in (outcome.stdout.strip(), outcome.stderr.strip()) if part
    )
    lines = combined.splitlines()
    truncated = len(lines) > _OUTPUT_TAIL_MAX_LINES
    tail = "\n".join(lines[-_OUTPUT_TAIL_MAX_LINES:])
    if len(tail) > _OUTPUT_TAIL_MAX_CHARS:
        return tail[-_OUTPUT_TAIL_MAX_CHARS:], True
    return tail, truncated


def _bounded_text(text: str, limit: int) -> str:
    """Cap report text while keeping a visible truncation marker."""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _proposal_attempt_summary(response: ProposalResponse) -> dict[str, Any]:
    """Describe one proposal response without embedding its raw content."""
    error = response.validation_error
    return {
        "attempt": response.attempt,
        "valid": response.valid,
        "raw_response_present": response.raw_response is not None,
        "parsed_proposal_present": response.parsed_proposal is not None,
        "validation_error": (
            _bounded_text(error, _PROPOSAL_ERROR_MAX_CHARS) if error else None
        ),
    }


def _new_artifact_directory(report_dir: Path, stem: str) -> Path:
    """Create a unique artifact directory without overwriting an earlier run."""
    base = report_dir / "artifacts"
    candidate = base / stem
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _write_json_artifact(
    artifact_dir: Path, filename: str, payload: dict[str, Any]
) -> Path:
    """Write one complete artifact as pretty JSON."""
    path = artifact_dir / filename
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _artifact_reference(
    path: Path, report_dir: Path, includes: list[str]
) -> dict[str, Any]:
    """Return compact metadata pointing at one separate artifact."""
    data = path.read_bytes()
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "path": path.relative_to(report_dir).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "includes": includes,
    }


def _proposal_artifact_payload(
    entry: BenchmarkEntry, attempt_number: int, attempt: Attempt
) -> dict[str, Any]:
    """Build the separate artifact holding one complete proposed fix."""
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "kind": "fix_proposal",
        "repository": f"{entry.owner}/{entry.repository}",
        "issue": entry.issue_number,
        "commit": entry.commit,
        "attempt": attempt_number,
        "file_path": attempt.fix_result.file_path,
        "reasoning": attempt.fix_result.reasoning,
        "exploration_rounds": attempt.fix_result.exploration_rounds,
        "original_content": attempt.fix_result.original_content,
        "new_content": attempt.fix_result.new_content,
        "proposal_attempts": [
            response.model_dump(mode="json")
            for response in attempt.fix_result.proposal_attempts
        ],
    }


def _guard_artifact_payload(
    entry: BenchmarkEntry, event: Any
) -> dict[str, Any]:
    """Build the separate artifact for a guard-rejected proposal."""
    return {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "kind": "rejected_fix_proposal",
        "repository": f"{entry.owner}/{entry.repository}",
        "issue": entry.issue_number,
        "commit": entry.commit,
        "attempt": event.attempt,
        "file_path": event.file_path,
        "rejection_kind": event.kind,
        "reason": event.reason,
        "proposal_attempts": [
            response.model_dump(mode="json")
            for response in event.proposal_attempts
        ],
    }


def _attempt_record(
    attempt: Attempt, proposal_artifact: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Serialize one attempt without its full file contents."""
    diff, diff_truncated = _diff(attempt.fix_result)
    output_tail, output_truncated = _output_tail(attempt.outcome)
    return {
        "file_path": attempt.fix_result.file_path,
        "reasoning": attempt.fix_result.reasoning,
        "diff": diff,
        "diff_truncated": diff_truncated,
        "test": {
            "passed": attempt.outcome.passed,
            "return_code": attempt.outcome.return_code,
            "timed_out": attempt.outcome.timed_out,
            "started": attempt.outcome.started,
            "timeout_seconds": attempt.outcome.timeout_seconds,
            "output_tail": output_tail,
            "output_truncated": output_truncated,
        },
        "diagnosis": (
            attempt.diagnosis.model_dump(mode="json")
            if attempt.diagnosis is not None
            else None
        ),
        "diagnosis_error": attempt.diagnosis_error,
        "avoid_file_paths_after": list(attempt.avoid_file_paths_after),
        "exploration_rounds": attempt.fix_result.exploration_rounds,
        "exploration": [
            step.model_dump(mode="json")
            for step in attempt.exploration_steps[:_EXPLORATION_MAX_STEPS]
        ],
        "exploration_truncated": (
            len(attempt.exploration_steps) > _EXPLORATION_MAX_STEPS
        ),
        "proposal_attempts": [
            _proposal_attempt_summary(response)
            for response in attempt.fix_result.proposal_attempts
        ],
        "proposal_artifact": proposal_artifact,
    }


def _baseline_record(baseline: BaselineResult) -> dict[str, Any]:
    """Serialize baseline evidence without duplicating the full entry."""
    tail, truncated = _cap_text(baseline.output_tail)
    return {
        "status": baseline.status,
        "detail": baseline.detail,
        "return_code": baseline.return_code,
        "output_tail": tail,
        "output_truncated": truncated,
    }


def _cap_text(text: str) -> tuple[str, bool]:
    """Apply the run report's output-tail line and character caps."""
    lines = text.splitlines()
    truncated = len(lines) > _OUTPUT_TAIL_MAX_LINES
    tail = "\n".join(lines[-_OUTPUT_TAIL_MAX_LINES:])
    if len(tail) > _OUTPUT_TAIL_MAX_CHARS:
        return tail[-_OUTPUT_TAIL_MAX_CHARS:], True
    return tail, truncated


def write_run_report(
    report_dir: str,
    entry: BenchmarkEntry,
    provider: str,
    model: str,
    loop_result: LoopResult,
    status: RunStatus,
    pr_url: str | None,
    timing: RunTiming,
    error: str | None = None,
    test_patch: TestPatchResult | None = None,
    *,
    base_url_host: str | None = None,
    status_reason: str | None = None,
    baseline: BaselineResult | None = None,
    environment: EnvironmentSummary | None = None,
    failure_class: FailureClass | None = None,
) -> Path:
    """Persist the compact run record and separate complete proposal artifacts."""
    calls = loop_result.llm_calls
    resolved_failure_class = (
        failure_class
        if failure_class is not None
        else RUN_STATUS_FAILURE_CLASS[status]
    )
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{entry.owner}-{entry.repository}")
    path = target / f"{timing.started_at:%Y%m%dT%H%M%SZ}_{slug}_{entry.issue_number}.json"

    has_proposals = bool(
        loop_result.attempts
        or any(event.proposal_attempts for event in loop_result.guard_events)
    )
    artifact_dir = (
        _new_artifact_directory(target, path.stem) if has_proposals else None
    )

    attempt_artifacts: dict[int, dict[str, Any]] = {}
    if artifact_dir is not None:
        for number, attempt in enumerate(loop_result.attempts, start=1):
            artifact_path = _write_json_artifact(
                artifact_dir,
                f"attempt-{number}-proposal.json",
                _proposal_artifact_payload(entry, number, attempt),
            )
            attempt_artifacts[number] = _artifact_reference(
                artifact_path,
                target,
                ["original_content", "new_content", "proposal_attempts"],
            )

    guard_records: list[dict[str, Any]] = []
    for number, event in enumerate(loop_result.guard_events, start=1):
        artifact_reference = None
        if artifact_dir is not None and event.proposal_attempts:
            artifact_path = _write_json_artifact(
                artifact_dir,
                f"guard-{number}-proposal.json",
                _guard_artifact_payload(entry, event),
            )
            artifact_reference = _artifact_reference(
                artifact_path,
                target,
                ["proposal_attempts"],
            )
        record = event.model_dump(
            mode="json", exclude={"proposal_attempts"}
        )
        record["proposal_attempts"] = [
            _proposal_attempt_summary(response)
            for response in event.proposal_attempts
        ]
        record["proposal_artifact"] = artifact_reference
        guard_records.append(record)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "repository": f"{entry.owner}/{entry.repository}",
        "issue": entry.issue_number,
        "commit": entry.commit,
        "provider": provider,
        "base_url_host": base_url_host,
        "model": model,
        "status": status,
        "failure_class": resolved_failure_class,
        "status_reason": status_reason,
        "error": error,
        "started_at": timing.started_at.isoformat(),
        "finished_at": timing.finished_at.isoformat(),
        "duration_seconds": timing.duration_seconds,
        "attempts": len(loop_result.attempts),
        "exploration_rounds": sum(
            attempt.fix_result.exploration_rounds
            for attempt in loop_result.attempts
        ),
        "llm_calls": len(calls),
        "input_tokens": _token_total(calls, "input_tokens"),
        "output_tokens": _token_total(calls, "output_tokens"),
        "total_tokens": _token_total(calls, "total_tokens"),
        "estimated_input_tokens": max(
            (call.estimated_input_tokens or 0 for call in calls), default=0
        ) or None,
        "input_token_limit": next(
            (call.input_token_limit for call in calls if call.input_token_limit is not None),
            None,
        ),
        "input_token_count_method": next(
            (
                call.input_token_count_method
                for call in calls
                if call.input_token_count_method is not None
            ),
            None,
        ),
        "input_token_count_reason": next(
            (
                call.input_token_count_reason
                for call in calls
                if call.input_token_count_reason is not None
            ),
            None,
        ),
        "pr": pr_url,
        "calls": [call.model_dump(mode="json") for call in calls],
        "attempt_records": [
            {
                "attempt": number,
                **_attempt_record(
                    attempt, attempt_artifacts.get(number)
                ),
            }
            for number, attempt in enumerate(loop_result.attempts, start=1)
        ],
        "guard_events": guard_records,
        "test_patch": {
            "applied": test_patch.applied if test_patch is not None else False,
            "file_paths": (
                list(test_patch.file_paths) if test_patch is not None else []
            ),
        },
        "baseline": (
            _baseline_record(baseline) if baseline is not None else None
        ),
        "environment": (
            environment.model_dump(mode="json")
            if environment is not None
            else None
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_benchmark_summary(
    report_dir: str,
    results: list[tuple[str, RunStatus]],
    timing: RunTiming,
) -> Path:
    """Persist benchmark-level counts and the per-entry report paths."""
    counts = {
        "total": len(results),
        "succeeded": 0,
        "failed": 0,
        "test_timeout": 0,
        "context_too_large": 0,
        "could_not_install": 0,
        "provider_unavailable": 0,
        "errored": 0,
    }
    count_keys: dict[RunStatus, str] = {
        "success": "succeeded",
        "failed": "failed",
        "test_timeout": "test_timeout",
        "context_too_large": "context_too_large",
        "could_not_install": "could_not_install",
        "provider_unavailable": "provider_unavailable",
        "errored": "errored",
    }
    for _, status in results:
        counts[count_keys[status]] += 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "started_at": timing.started_at.isoformat(),
        "finished_at": timing.finished_at.isoformat(),
        "duration_seconds": timing.duration_seconds,
        "counts": counts,
        "reports": [path for path, _ in results],
    }
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{timing.started_at:%Y%m%dT%H%M%SZ}_benchmark_summary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def format_run_summary(report: dict[str, Any]) -> str:
    def display(value: Any) -> Any: return "UNKNOWN" if value is None else value
    lines = [
        "=" * 50,
        "DEBUGGERNAUT RUN SUMMARY",
        "=" * 50,
        f"Repository:  {report['repository']}",
        f"Issue:       #{report['issue']}",
        f"Provider:    {report['provider']}",
        f"Base URL host: {report.get('base_url_host') or 'N/A'}",
        f"Model:       {report['model']}",
        "",
        f"Status:      {report['status'].upper()}",
        f"Failure:     {str(report.get('failure_class') or 'NONE').upper()}",
    ]
    baseline = report.get("baseline")
    if baseline is not None:
        lines.append(f"Baseline:    {baseline['status']}")
    if report.get("status_reason"):
        lines.append(f"Reason:      {report['status_reason']}")
    lines.extend(
        [
            f"Attempts:    {report['attempts']}",
            f"LLM calls:   {report['llm_calls']}",
            "",
            f"Input tokens:   {display(report['input_tokens'])}",
            f"Output tokens:  {display(report['output_tokens'])}",
            f"Total tokens:   {display(report['total_tokens'])}",
            f"Input token count: {display(report.get('estimated_input_tokens'))} "
            f"({report.get('input_token_count_method') or 'method unknown'})",
            f"Input token limit: {display(report.get('input_token_limit'))}",
            "",
            f"PR:          {report['pr'] or 'NONE'}",
            "=" * 50,
        ]
    )
    return "\n".join(lines)

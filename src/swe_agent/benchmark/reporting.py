"""Write safe machine-readable results for benchmark runs."""
from __future__ import annotations
import difflib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import Attempt, LoopResult
from swe_agent.benchmark.config import BenchmarkEntry


SCHEMA_VERSION = 2
_DIFF_MAX_CHARS = 8_000
_OUTPUT_TAIL_MAX_LINES = 40
_OUTPUT_TAIL_MAX_CHARS = 8_000
_EXPLORATION_MAX_STEPS = 100

RunStatus = Literal["success", "failed", "errored"]


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


def _attempt_record(attempt: Attempt) -> dict[str, Any]:
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
            "output_tail": output_tail,
            "output_truncated": output_truncated,
        },
        "diagnosis": (
            attempt.diagnosis.model_dump(mode="json")
            if attempt.diagnosis is not None
            else None
        ),
        "avoid_file_paths_after": list(attempt.avoid_file_paths_after),
        "exploration": [
            step.model_dump(mode="json")
            for step in attempt.exploration_steps[:_EXPLORATION_MAX_STEPS]
        ],
        "exploration_truncated": (
            len(attempt.exploration_steps) > _EXPLORATION_MAX_STEPS
        ),
    }


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
) -> Path:
    """Persist the full run record; prompts, file contents, and keys are excluded."""
    calls = loop_result.llm_calls
    payload = {
        "schema_version": SCHEMA_VERSION,
        "repository": f"{entry.owner}/{entry.repository}",
        "issue": entry.issue_number,
        "commit": entry.commit,
        "provider": provider,
        "model": model,
        "status": status,
        "error": error,
        "started_at": timing.started_at.isoformat(),
        "finished_at": timing.finished_at.isoformat(),
        "duration_seconds": timing.duration_seconds,
        "attempts": len(loop_result.attempts),
        "llm_calls": len(calls),
        "input_tokens": _token_total(calls, "input_tokens"),
        "output_tokens": _token_total(calls, "output_tokens"),
        "total_tokens": _token_total(calls, "total_tokens"),
        "pr": pr_url,
        "calls": [call.model_dump(mode="json") for call in calls],
        "attempt_records": [
            {"attempt": number, **_attempt_record(attempt)}
            for number, attempt in enumerate(loop_result.attempts, start=1)
        ],
        "guard_events": [
            event.model_dump(mode="json") for event in loop_result.guard_events
        ],
    }
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{entry.owner}-{entry.repository}")
    path = target / f"{timing.started_at:%Y%m%dT%H%M%SZ}_{slug}_{entry.issue_number}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_benchmark_summary(
    report_dir: str,
    results: list[tuple[str, RunStatus]],
    timing: RunTiming,
) -> Path:
    """Persist benchmark-level counts and the per-entry report paths."""
    counts = {"total": len(results), "succeeded": 0, "failed": 0, "errored": 0}
    count_keys = {"success": "succeeded", "failed": "failed", "errored": "errored"}
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
    return "\n".join(["=" * 50, "DEBUGGERNAUT RUN SUMMARY", "=" * 50, f"Repository:  {report['repository']}", f"Issue:       #{report['issue']}", f"Provider:    {report['provider']}", f"Model:       {report['model']}", "", f"Status:      {report['status'].upper()}", f"Attempts:    {report['attempts']}", f"LLM calls:   {report['llm_calls']}", "", f"Input tokens:   {display(report['input_tokens'])}", f"Output tokens:  {display(report['output_tokens'])}", f"Total tokens:   {display(report['total_tokens'])}", "", f"PR:          {report['pr'] or 'NONE'}", "=" * 50])

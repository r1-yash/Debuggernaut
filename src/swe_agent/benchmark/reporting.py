"""Write safe machine-readable results for a single benchmark entry."""
from __future__ import annotations
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from swe_agent.agent.loop import LoopResult

def _token_total(calls: list[Any], field: str) -> int | None:
    values = [getattr(call, field) for call in calls]
    return sum(values) if values and all(value is not None for value in values) else None

def write_run_report(report_dir: str, owner: str, repository: str, issue: int, provider: str, model: str, loop_result: LoopResult, status: str, pr_url: str | None) -> Path:
    """Persist operational metadata only; prompts, file content, and keys are excluded."""
    calls = loop_result.llm_calls
    payload = {"repository": f"{owner}/{repository}", "issue": issue, "provider": provider, "model": model, "status": status, "attempts": len(loop_result.attempts), "llm_calls": len(calls), "input_tokens": _token_total(calls, "input_tokens"), "output_tokens": _token_total(calls, "output_tokens"), "total_tokens": _token_total(calls, "total_tokens"), "pr": pr_url, "calls": [call.model_dump(mode="json") for call in calls]}
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{owner}-{repository}")
    path = target / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{slug}_{issue}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

def format_run_summary(report: dict[str, Any]) -> str:
    def display(value: Any) -> Any: return "UNKNOWN" if value is None else value
    return "\n".join(["=" * 50, "DEBUGGERNAUT RUN SUMMARY", "=" * 50, f"Repository:  {report['repository']}", f"Issue:       #{report['issue']}", f"Provider:    {report['provider']}", f"Model:       {report['model']}", "", f"Status:      {report['status'].upper()}", f"Attempts:    {report['attempts']}", f"LLM calls:   {report['llm_calls']}", "", f"Input tokens:   {display(report['input_tokens'])}", f"Output tokens:  {display(report['output_tokens'])}", f"Total tokens:   {display(report['total_tokens'])}", "", f"PR:          {report['pr'] or 'NONE'}", "=" * 50])

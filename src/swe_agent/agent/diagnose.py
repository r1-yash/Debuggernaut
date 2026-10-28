"""Diagnose why a proposed fix did not pass its test command."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.structured import parse_structured_response
from swe_agent.llm import coerce_provider


_SYSTEM_INSTRUCTION = """You diagnose failed software-maintenance fixes.
Classify the failure as `wrong_file` only when the proposed file was not the
right target at all. Classify it as `wrong_fix` when the file was relevant but
the proposed change's logic was incorrect or incomplete. Explain concisely."""
_DIAGNOSIS_OUTPUT_TAIL_CHARS = 4_000
_DIAGNOSIS_ERROR_MAX_CHARS = 300
_DIAGNOSIS_ERROR_BODY_MAX_CHARS = 200


class Diagnosis(BaseModel):
    """Gemini's classification of a failed fix attempt."""

    reason: Literal["wrong_file", "wrong_fix"]
    explanation: str


def _timeout_message(outcome: TestOutcome) -> str | None:
    """Return an explicit timeout explanation for a timed-out test run."""
    if not outcome.timed_out:
        return None
    if outcome.timeout_seconds is None:
        return "tests timed out after an unknown timeout"
    seconds = f"{outcome.timeout_seconds:g}"
    return f"tests timed out after {seconds} s"


def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cap text at ``limit`` characters."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def describe_diagnosis_error(error: Exception) -> str:
    """Describe a failed diagnosis call without keys, headers, or file contents."""
    status = getattr(error, "status_code", None)
    label = type(error).__name__
    if isinstance(status, int) and not isinstance(status, bool):
        label = f"{label} (HTTP {status})"
    message = _truncate(str(error), _DIAGNOSIS_ERROR_MAX_CHARS) or "no message"
    description = f"{label}: {message}"
    body = getattr(error, "provider_body", None)
    if isinstance(body, str):
        body_text = _truncate(body, _DIAGNOSIS_ERROR_BODY_MAX_CHARS)
        if body_text:
            description += f"; provider body: {body_text}"
    return description


def _output_tail(outcome: TestOutcome) -> str:
    """Return a bounded combined output tail, including an empty-output marker."""
    combined = "\n".join(
        part for part in (outcome.stdout.strip(), outcome.stderr.strip()) if part
    )
    tail = combined[-_DIAGNOSIS_OUTPUT_TAIL_CHARS:]
    return tail if tail else "(no output)"


def build_diagnosis_prompt(fix_result: FixResult, outcome: TestOutcome) -> str:
    """Build the diagnosis prompt, making timeouts explicit to the model."""
    parts = [
        f"Proposed file path: {fix_result.file_path}",
        f"Proposal reasoning:\n{fix_result.reasoning}",
        f"Test stdout:\n{outcome.stdout}",
        f"Test stderr:\n{outcome.stderr}",
        f"Test output tail:\n{_output_tail(outcome)}",
    ]
    timeout_message = _timeout_message(outcome)
    if timeout_message is not None:
        parts.append(timeout_message)
    return "\n\n".join(parts)


def diagnose_failure(
    fix_result: FixResult,
    outcome: TestOutcome,
    client: object,
) -> Diagnosis:
    """Classify a failed fix using its proposal and captured test output."""
    response = coerce_provider(client).structured(
        _SYSTEM_INSTRUCTION,
        build_diagnosis_prompt(fix_result, outcome),
        Diagnosis,
        "diagnosis",
    )
    return parse_structured_response(response, Diagnosis)

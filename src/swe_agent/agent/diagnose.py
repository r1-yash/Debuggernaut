"""Diagnose why a proposed fix did not pass its test command."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.structured import parse_structured_response
from swe_agent.llm import coerce_provider


_SYSTEM_INSTRUCTION = """You diagnose failed software-maintenance fixes.
Classify the failure as `wrong_file` only when the proposed file was not the
right target at all. Classify it as `wrong_fix` when the file was relevant but
the proposed change's logic was incorrect or incomplete. Explain concisely.
Return only a JSON object with `reason` and `explanation` keys."""
_DIAGNOSIS_OUTPUT_TAIL_CHARS = 4_000
_DIAGNOSIS_ERROR_MAX_CHARS = 300
_DIAGNOSIS_ERROR_BODY_MAX_CHARS = 200
_DIAGNOSIS_REPLY_MAX_CHARS = 2_000


class Diagnosis(BaseModel):
    """The model's classification of a failed fix attempt."""

    reason: Literal["wrong_file", "wrong_fix", "unknown"]
    explanation: str


class DiagnosisUnavailable(ValueError):
    """A diagnosis reply stayed invalid after one repair attempt."""

    def __init__(self, message: str, diagnosis: Diagnosis) -> None:
        super().__init__(message)
        self.diagnosis = diagnosis


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
    parts.append(
        "Return the diagnosis as a JSON object with `reason` set to "
        "`wrong_file` or `wrong_fix`, plus an `explanation` string."
    )
    return "\n\n".join(parts)


def _response_text(response: Any) -> str:
    """Return a bounded text form of a diagnosis response for a repair prompt."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    if isinstance(parsed, dict):
        try:
            return json.dumps(parsed, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(parsed)
    if isinstance(parsed, str):
        return parsed
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    choices = getattr(response, "choices", None)
    if isinstance(choices, (list, tuple)) and choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if isinstance(content, str):
            return content
    return ""


def _repair_prompt(prompt: str, response: Any, error: Exception) -> str:
    """Ask once for a schema-valid diagnosis, showing the failed answer."""
    previous = _truncate(_response_text(response), _DIAGNOSIS_REPLY_MAX_CHARS)
    if not previous:
        previous = "(no response text)"
    return (
        f"{prompt}\n\n"
        "Your previous answer could not be validated: "
        f"{_truncate(str(error), _DIAGNOSIS_ERROR_MAX_CHARS)}\n\n"
        f"Previous answer:\n{previous}\n\n"
        "Return only a JSON object with `reason` set to `wrong_file` or "
        "`wrong_fix` and an `explanation` string."
    )


def _parse_diagnosis(response: Any) -> Diagnosis:
    """Parse one provider response as a schema-valid diagnosis."""
    return parse_structured_response(response, Diagnosis)


def diagnose_failure(
    fix_result: FixResult,
    outcome: TestOutcome,
    client: object,
) -> Diagnosis:
    """Classify a failed fix, repairing one invalid reply before giving up."""
    prompt = build_diagnosis_prompt(fix_result, outcome)
    provider = coerce_provider(client)
    response = provider.structured(
        _SYSTEM_INSTRUCTION,
        prompt,
        Diagnosis,
        "diagnosis",
    )
    try:
        return _parse_diagnosis(response)
    except ValueError as first_error:
        repaired = provider.structured(
            _SYSTEM_INSTRUCTION,
            _repair_prompt(prompt, response, first_error),
            Diagnosis,
            "diagnosis_repair",
        )
        try:
            return _parse_diagnosis(repaired)
        except ValueError as second_error:
            message = (
                "Diagnosis response failed schema validation after one repair "
                f"attempt: {_truncate(str(second_error), _DIAGNOSIS_ERROR_MAX_CHARS)}"
            )
            unknown = Diagnosis(
                reason="unknown",
                explanation=_truncate(message, _DIAGNOSIS_ERROR_MAX_CHARS),
            )
            raise DiagnosisUnavailable(message, unknown) from second_error

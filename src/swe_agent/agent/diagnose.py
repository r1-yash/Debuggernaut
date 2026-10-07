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


class Diagnosis(BaseModel):
    """Gemini's classification of a failed fix attempt."""

    reason: Literal["wrong_file", "wrong_fix"]
    explanation: str


def diagnose_failure(
    fix_result: FixResult,
    outcome: TestOutcome,
    client: object,
) -> Diagnosis:
    """Classify a failed fix using its proposal and captured test output."""
    response = coerce_provider(client).structured(
        _SYSTEM_INSTRUCTION,
        (
            f"Proposed file path: {fix_result.file_path}\n\n"
            f"Proposal reasoning:\n{fix_result.reasoning}\n\n"
            f"Test stdout:\n{outcome.stdout}\n\n"
            f"Test stderr:\n{outcome.stderr}"
        ), Diagnosis, "diagnosis",
    )
    return parse_structured_response(response, Diagnosis)

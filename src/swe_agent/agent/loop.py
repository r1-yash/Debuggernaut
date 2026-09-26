"""Propose, test, diagnose, and retry repository fixes with LangGraph."""

from __future__ import annotations

from typing import TypedDict

from google import genai
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from swe_agent.agent.diagnose import diagnose_failure
from swe_agent.agent.executor import TestOutcome, apply_and_test
from swe_agent.agent.fixer import FixResult, propose_fix


class Attempt(BaseModel):
    """A proposed fix and the outcome of testing it."""

    fix_result: FixResult
    outcome: TestOutcome


class LoopResult(BaseModel):
    """The final result and full history of attempts to resolve an issue."""

    succeeded: bool
    final_attempt: Attempt | None
    attempts: list[Attempt]


class _LoopState(TypedDict):
    """Mutable state carried between issue-resolution graph nodes."""

    repo_path: str
    issue_title: str
    issue_body: str
    test_command: list[str]
    client: genai.Client
    max_attempts: int
    attempts: list[Attempt]
    avoid_file_paths: list[str]
    proposed_fix: FixResult | None
    succeeded: bool


## so it is used to provide with the reason of previous failed attempts and to avoid proposing fixes for the same files again

def _issue_body_with_failed_attempts(
    issue_body: str,
    attempts: list[Attempt],
    avoid_file_paths: list[str] | None = None,
) -> str:
    """Append prior failure details and file-targeting guidance to an issue."""
    additions: list[str] = []
    if attempts:
        summaries = []
        for number, attempt in enumerate(attempts, start=1):
            summaries.append(
                f"Attempt {number}:\n"
                f"File path: {attempt.fix_result.file_path}\n"
                f"Reasoning: {attempt.fix_result.reasoning}\n"
                f"Test stdout: {attempt.outcome.stdout}\n"
                f"Test stderr: {attempt.outcome.stderr}"
            )
        additions.append(
            "Previous attempted fixes failed. Use their results to choose a "
            "different solution; do not repeat the same approach.\n\n"
            + "\n\n".join(summaries)
        )
    if avoid_file_paths:
        additions.append(
            "Do not target these files again: "
            + ", ".join(avoid_file_paths)
            + ". Explore other relevant files instead."
        )
    return issue_body if not additions else f"{issue_body}\n\n" + "\n\n".join(additions)

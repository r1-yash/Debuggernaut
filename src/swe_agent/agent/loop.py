"""Propose, test, and retry repository fixes for a reported issue."""

from __future__ import annotations

from google import genai
from pydantic import BaseModel

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


def _issue_body_with_failed_attempts(
    issue_body: str,
    attempts: list[Attempt],
) -> str:
    """Append prior failed attempt details for the next proposed fix."""
    if not attempts:
        return issue_body

    summaries = []
    for number, attempt in enumerate(attempts, start=1):
        summaries.append(
            f"Attempt {number}:\n"
            f"File path: {attempt.fix_result.file_path}\n"
            f"Reasoning: {attempt.fix_result.reasoning}\n"
            f"Test stdout: {attempt.outcome.stdout}\n"
            f"Test stderr: {attempt.outcome.stderr}"
        )
    return (
        f"{issue_body}\n\n"
        "Previous attempted fixes failed. Use their results to choose a "
        "different solution; do not repeat the same approach.\n\n"
        + "\n\n".join(summaries)
    )


def resolve_issue(
    repo_path: str,
    issue_title: str,
    issue_body: str,
    test_command: list[str],
    client: genai.Client,
    max_attempts: int = 3,
) -> LoopResult:
    """Propose and test fixes until one passes or attempts are exhausted."""
    if max_attempts < 1:
        raise ValueError("Maximum attempts must be at least 1.")

    attempts: list[Attempt] = []
    for _ in range(max_attempts):
        fix_result = propose_fix(
            repo_path,
            issue_title,
            _issue_body_with_failed_attempts(issue_body, attempts),
            client,
        )
        outcome = apply_and_test(repo_path, fix_result, test_command)
        attempt = Attempt(fix_result=fix_result, outcome=outcome)
        attempts.append(attempt)
        if outcome.passed:
            return LoopResult(
                succeeded=True,
                final_attempt=attempt,
                attempts=attempts,
            )

    return LoopResult(
        succeeded=False,
        final_attempt=attempts[-1],
        attempts=attempts,
    )

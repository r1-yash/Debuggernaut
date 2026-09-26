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


def _explore_and_propose(state: _LoopState) -> dict[str, FixResult]:
    """Propose a fix using prior attempts and diagnosed wrong-file targets."""
    return {
        "proposed_fix": propose_fix(
            state["repo_path"],
            state["issue_title"],
            _issue_body_with_failed_attempts(
                state["issue_body"], state["attempts"], state["avoid_file_paths"]
            ),
            state["client"],
        )
    }


def _apply_and_test(state: _LoopState) -> dict[str, list[Attempt] | bool]:
    """Apply the proposed fix, test it, and preserve the resulting attempt."""
    fix_result = state["proposed_fix"]
    if fix_result is None:
        raise ValueError("A proposed fix is required before testing.")
    outcome = apply_and_test(state["repo_path"], fix_result, state["test_command"])
    return {
        "attempts": [
            *state["attempts"], Attempt(fix_result=fix_result, outcome=outcome)
        ],
        "succeeded": outcome.passed,
    }


def _diagnose(state: _LoopState) -> dict[str, list[str]]:
    """Diagnose the most recent failed attempt and retain wrong-file targets."""
    latest_attempt = state["attempts"][-1]
    diagnosis = diagnose_failure(
        latest_attempt.fix_result, latest_attempt.outcome, state["client"]
    )
    if (
        diagnosis.reason == "wrong_file"
        and latest_attempt.fix_result.file_path not in state["avoid_file_paths"]
    ):
        return {
            "avoid_file_paths": [
                *state["avoid_file_paths"], latest_attempt.fix_result.file_path
            ]
        }
    return {}


def _after_apply_and_test(state: _LoopState) -> str:
    """Choose whether resolution is complete, exhausted, or needs diagnosis."""
    if state["attempts"][-1].outcome.passed:
        return "passed"
    if len(state["attempts"]) >= state["max_attempts"]:
        return "fail"
    return "diagnose"


def _build_graph() -> StateGraph:
    """Build the state graph used for one issue-resolution run."""
    graph = StateGraph(_LoopState)
    graph.add_node("explore_and_propose", _explore_and_propose)
    graph.add_node("apply_and_test", _apply_and_test)
    graph.add_node("diagnose", _diagnose)
    graph.add_edge(START, "explore_and_propose")
    graph.add_edge("explore_and_propose", "apply_and_test")
    graph.add_conditional_edges(
        "apply_and_test",
        _after_apply_and_test,
        {"passed": END, "fail": END, "diagnose": "diagnose"},
    )
    graph.add_edge("diagnose", "explore_and_propose")
    return graph


def resolve_issue(
    repo_path: str,
    issue_title: str,
    issue_body: str,
    test_command: list[str],
    client: genai.Client,
    max_attempts: int = 3,
) -> LoopResult:
    """Resolve an issue through a LangGraph proposal, test, and diagnosis loop."""
    if max_attempts < 1:
        raise ValueError("Maximum attempts must be at least 1.")

    final_state = _build_graph().compile().invoke(
        {
            "repo_path": repo_path,
            "issue_title": issue_title,
            "issue_body": issue_body,
            "test_command": test_command,
            "client": client,
            "max_attempts": max_attempts,
            "attempts": [],
            "avoid_file_paths": [],
            "proposed_fix": None,
            "succeeded": False,
        }
    )
    attempts = final_state["attempts"]
    return LoopResult(
        succeeded=final_state["succeeded"],
        final_attempt=attempts[-1] if attempts else None,
        attempts=attempts,
    )

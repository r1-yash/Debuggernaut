"""Propose, test, diagnose, and retry repository fixes with LangGraph."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from swe_agent.agent.diagnose import Diagnosis, diagnose_failure
from swe_agent.agent.executor import TestOutcome, apply_and_test
from swe_agent.agent.fixer import ExplorationStep, FixResult, propose_fix
from swe_agent.agent.guardrails import RejectedProposalError
from swe_agent.llm import LLMCall, UsageTracker, usage_run


_MAX_GUARD_REJECTIONS = 3


class Attempt(BaseModel):
    """A proposed fix and the outcome of testing it."""

    fix_result: FixResult
    outcome: TestOutcome
    diagnosis: Diagnosis | None = None
    avoid_file_paths_after: list[str] = []
    exploration_steps: list[ExplorationStep] = []


class GuardEvent(BaseModel):
    """A proposal rejected before any repository file was modified."""

    attempt: int
    file_path: str
    kind: Literal["test_file", "nonexistent_path"]
    reason: str


class LoopResult(BaseModel):
    """The final result and full history of attempts to resolve an issue."""

    succeeded: bool
    final_attempt: Attempt | None
    attempts: list[Attempt]
    llm_calls: list[LLMCall] = []
    guard_events: list[GuardEvent] = []


class _LoopState(TypedDict):
    """Mutable state carried between issue-resolution graph nodes."""

    repo_path: str
    issue_title: str
    issue_body: str
    test_command: list[str]
    client: object
    max_attempts: int
    usage_tracker: UsageTracker
    attempts: list[Attempt]
    avoid_file_paths: list[str]
    proposed_fix: FixResult | None
    exploration_steps: list[ExplorationStep]
    guard_events: list[GuardEvent]
    succeeded: bool


def _issue_body_with_failed_attempts(
    issue_body: str,
    attempts: list[Attempt],
    avoid_file_paths: list[str] | None = None,
    guard_events: list[GuardEvent] | None = None,
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
    if guard_events:
        rejections = []
        for event in guard_events:
            label = (
                "is a test file and may not be modified"
                if event.kind == "test_file"
                else "does not exist in the repository"
            )
            rejections.append(
                f"Proposal for attempt {event.attempt} was rejected: "
                f"{event.file_path} {label}."
            )
        additions.append(
            "Some proposals were rejected before any file was modified; do not "
            "repeat them.\n\n" + "\n".join(rejections)
        )
    return issue_body if not additions else f"{issue_body}\n\n" + "\n\n".join(additions)


def _explore_and_propose(
    state: _LoopState,
) -> dict[str, FixResult | list[ExplorationStep] | list[GuardEvent] | None]:
    """Propose a fix, or record a rejected proposal for a retry."""
    state["usage_tracker"].set_attempt(len(state["attempts"]) + 1)
    exploration_steps: list[ExplorationStep] = []
    try:
        fix_result = propose_fix(
            state["repo_path"],
            state["issue_title"],
            _issue_body_with_failed_attempts(
                state["issue_body"],
                state["attempts"],
                state["avoid_file_paths"],
                state["guard_events"],
            ),
            state["client"],
            test_command=state["test_command"],
            exploration_steps=exploration_steps,
        )
    except RejectedProposalError as error:
        return {
            "proposed_fix": None,
            "exploration_steps": exploration_steps,
            "guard_events": [
                *state["guard_events"],
                GuardEvent(
                    attempt=len(state["attempts"]) + 1,
                    file_path=error.file_path,
                    kind=error.kind,
                    reason=error.reason,
                ),
            ],
        }
    return {"proposed_fix": fix_result, "exploration_steps": exploration_steps}


def _apply_and_test(state: _LoopState) -> dict[str, list[Attempt] | bool]:
    """Apply the proposed fix, test it, and preserve the resulting attempt."""
    fix_result = state["proposed_fix"]
    if fix_result is None:
        raise ValueError("A proposed fix is required before testing.")
    outcome = apply_and_test(state["repo_path"], fix_result, state["test_command"])
    return {
        "attempts": [
            *state["attempts"],
            Attempt(
                fix_result=fix_result,
                outcome=outcome,
                avoid_file_paths_after=list(state["avoid_file_paths"]),
                exploration_steps=state["exploration_steps"],
            ),
        ],
        "succeeded": outcome.passed,
    }


def _diagnose(state: _LoopState) -> dict[str, list[str] | list[Attempt]]:
    """Diagnose the most recent failed attempt and retain wrong-file targets."""
    latest_attempt = state["attempts"][-1]
    diagnosis = diagnose_failure(
        latest_attempt.fix_result, latest_attempt.outcome, state["client"]
    )
    avoid_file_paths = state["avoid_file_paths"]
    if (
        diagnosis.reason == "wrong_file"
        and latest_attempt.fix_result.file_path not in avoid_file_paths
    ):
        avoid_file_paths = [
            *avoid_file_paths, latest_attempt.fix_result.file_path
        ]
    diagnosed_attempt = latest_attempt.model_copy(
        update={
            "diagnosis": diagnosis,
            "avoid_file_paths_after": avoid_file_paths,
        }
    )
    return {
        "attempts": [*state["attempts"][:-1], diagnosed_attempt],
        "avoid_file_paths": avoid_file_paths,
    }


def _after_explore_and_propose(state: _LoopState) -> str:
    """Apply a proposal, retry after a rejection, or stop at the guard cap."""
    if state["proposed_fix"] is not None:
        return "apply"
    if len(state["guard_events"]) >= _MAX_GUARD_REJECTIONS:
        return "fail"
    return "retry"


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
    graph.add_conditional_edges(
        "explore_and_propose",
        _after_explore_and_propose,
        {"apply": "apply_and_test", "retry": "explore_and_propose", "fail": END},
    )
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
    client: object,
    max_attempts: int = 3,
    usage_tracker: UsageTracker | None = None,
) -> LoopResult:
    """Resolve an issue through a LangGraph proposal, test, and diagnosis loop."""
    if max_attempts < 1:
        raise ValueError("Maximum attempts must be at least 1.")

    tracker = usage_tracker or UsageTracker()
    with usage_run(tracker):
        final_state = _build_graph().compile().invoke(
            {
                "repo_path": repo_path,
                "issue_title": issue_title,
                "issue_body": issue_body,
                "test_command": test_command,
                "client": client,
                "max_attempts": max_attempts,
                "usage_tracker": tracker,
                "attempts": [],
                "avoid_file_paths": [],
                "proposed_fix": None,
                "exploration_steps": [],
                "guard_events": [],
                "succeeded": False,
            }
        )
    attempts = final_state["attempts"]
    return LoopResult(
        succeeded=final_state["succeeded"],
        final_attempt=attempts[-1] if attempts else None,
        attempts=attempts,
        llm_calls=tracker.calls,
        guard_events=final_state["guard_events"],
    )

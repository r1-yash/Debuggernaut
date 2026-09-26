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


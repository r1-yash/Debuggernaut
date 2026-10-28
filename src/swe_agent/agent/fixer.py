"""Explore a cloned repository and propose a full-file bug fix with Gemini."""

from __future__ import annotations

import subprocess
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel

from swe_agent.llm import coerce_provider
from swe_agent.agent.guardrails import RejectedProposalError, proposal_rejection
from swe_agent.agent.paths import repository_root, resolve_repository_path
from swe_agent.agent.structured import parse_structured_response

DEFAULT_TOOL_RESULT_MAX_CHARS = 20_000
_TOOL_RESULT_INPUT_FRACTION = 0.10
# The provider's conservative estimate budgets two bytes per input token.
_CHARS_PER_ESTIMATED_TOKEN = 2

#BUG FIX - Prompt
_SYSTEM_INSTRUCTION = """You are a senior software engineer working in an unfamiliar production
repository. Investigate before modifying. Your goal is to make the smallest,
safest, well-tested change that fixes the reported issue.

REPOSITORY DISCOVERY
- Treat the current repository checkout as the source of truth.
- Never assume a file path mentioned in a GitHub issue is still valid.
- Issue references may be stale because files can be renamed, moved, or deleted.
- Before proposing a file_path, verify that it exists using repository tools.
- If a referenced path does not exist, use repository_search_code,
  repository_list_directory, or other available tools to locate the actual
  implementation.
- Never invent, infer, or hallucinate a file path.

UNDERSTAND BEFORE MODIFYING
- Inspect the relevant implementation and tests before changing code.
- Trace the relevant execution path and identify the root cause.
- Do not patch based solely on the issue description or an assumption.
- Prefer: search → inspect → understand → modify → test.

MINIMAL DIFF
- Make the smallest change that correctly fixes the root cause.
- Do not refactor, reformat, rename, reorder, or rewrite unrelated code.
- Do not rewrite an entire function when a localized change is sufficient.
- Do not modify files that are not necessary for the fix.
- Preserve existing APIs, behavior, and project conventions unless the issue
  explicitly requires changing them.

COMMENTS AND DOCUMENTATION — STRICT
- Never delete, rewrite, relocate, or clean up existing comments.
- Preserve existing comments exactly whenever possible.
- Do not remove TODOs, FIXMEs, documentation, or commented-out code unrelated
  to the fix.
- Only modify a comment if the code change makes it factually incorrect, and
  then change only what is necessary.

TESTING
- Inspect existing relevant tests before modifying code.
- Never modify, create, rename, or delete test files.
- Run the narrowest relevant tests first, then the broader test suite when
  practical.
- Do not declare success based solely on code inspection.

PROTECTED TEST FILES — STRICT
- Test files must never be changed by the agent. This includes anything under
  a tests/ or test/ directory, files named test_*.py or *_test.py,
  conftest.py, and any file or directory named in the repository's test
  command.
- Reading test files to understand expected behavior is allowed.
- Propose changes only to non-test source code.

FINAL DIFF CHECK
Before finalizing:
- Verify every changed file is necessary.
- Verify the diff contains only changes relevant to the issue.
- Verify no unrelated formatting/refactoring was introduced.
- Verify no existing comments were unnecessarily changed or removed.
- Verify the selected target file exists.
- Verify the root cause is addressed.
- Verify relevant tests pass.

PRINCIPLE
Produce the smallest, safest, maintainable patch that an experienced
open-source maintainer would reasonably accept."""


class FixResult(BaseModel):
    """A complete, un-applied file rewrite proposed for an issue."""

    file_path: str
    original_content: str
    new_content: str
    reasoning: str
    exploration_rounds: int = 0


class ExplorationStep(BaseModel):
    """One repository tool call made while exploring for a fix."""

    tool: Literal["read_file", "list_directory", "search_code"]
    arguments: dict[str, str]
    ok: bool
    result_size: int
    result_unit: Literal["characters", "entries", "lines"]
    error: str | None = None


def _record_exploration(
    steps: list[ExplorationStep],
    tool: Literal["read_file", "list_directory", "search_code"],
    arguments: dict[str, str],
    result_size: int,
    result_unit: Literal["characters", "entries", "lines"],
    error: str | None = None,
) -> None:
    """Append one exploration step to the caller's trajectory list."""
    steps.append(
        ExplorationStep(
            tool=tool,
            arguments=arguments,
            ok=error is None,
            result_size=result_size,
            result_unit=result_unit,
            error=error,
        )
    )


def _tool_result_char_limit(provider: object) -> int:
    """Return the character budget for one tool result.

    A single result may use about 10% of the provider's input-token limit, at
    two characters per token.  Real source text tokenizes at roughly three to
    four characters per token, so the cap stays below the budget the request
    counter measures.
    """
    max_input_tokens = getattr(provider, "max_input_tokens", None)
    if not isinstance(max_input_tokens, int) or max_input_tokens <= 0:
        return DEFAULT_TOOL_RESULT_MAX_CHARS
    budget_tokens = int(max_input_tokens * _TOOL_RESULT_INPUT_FRACTION)
    return max(1_000, budget_tokens * _CHARS_PER_ESTIMATED_TOKEN)


def _truncated_text(text: str, max_chars: int | None) -> str:
    """Cut text to a character budget with a visible truncation marker."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated: {len(text) - max_chars} more chars]"


def _history_without_unanswered_calls(history: list[Any]) -> list[Any]:
    """Drop trailing function calls that have no function responses.

    The Gemini SDK records a model turn whose function calls were never
    executed when automatic function calling reaches its remote-call budget.
    Sending that turn into the fix request would leave a function call without
    a response, so keep only its text parts and drop the call.
    """
    if not history:
        return history
    last = history[-1]
    parts = list(getattr(last, "parts", None) or [])
    if getattr(last, "role", None) != "model" or not any(
        getattr(part, "function_call", None) is not None for part in parts
    ):
        return history
    text_parts = [part for part in parts if getattr(part, "text", None)]
    if not text_parts:
        return list(history[:-1])
    return [*history[:-1], last.model_copy(update={"parts": text_parts})]


class _FixProposal(BaseModel):
    """The fields Gemini must return before local file content is attached."""

    file_path: str
    new_content: str
    reasoning: str


def read_file(
    repo_path: str, file_path: str, *, _max_chars: int | None = None
) -> str:
    """Return a repository file's contents without allowing path traversal.

    ``_max_chars`` caps the returned text and appends a truncation marker.
    """
    path = resolve_repository_path(repo_path, file_path)
    if not path.is_file():
        raise ValueError(f"File does not exist: {file_path}")
    return _truncated_text(path.read_text(encoding="utf-8"), _max_chars)


def list_directory(
    repo_path: str, dir_path: str = ".", *, _max_chars: int | None = None
) -> list[str]:
    """Return the names directly contained in a repository directory.

    ``_max_chars`` caps the returned entries and appends a truncation marker.
    """
    path = resolve_repository_path(repo_path, dir_path)
    if not path.is_dir():
        raise ValueError(f"Directory does not exist: {dir_path}")
    entries = sorted(entry.name for entry in path.iterdir())
    if _max_chars is None:
        return entries
    selected: list[str] = []
    used_chars = 0
    for entry in entries:
        if used_chars + len(entry) > _max_chars:
            break
        selected.append(entry)
        used_chars += len(entry) + 1
    remaining = sum(len(entry) + 1 for entry in entries) - used_chars
    if remaining <= 0:
        return selected
    return [*selected, f"[truncated: {remaining} more chars]"]


def search_code(
    repo_path: str, query: str, *, _max_chars: int | None = None
) -> list[str]:
    """Return up to 50 ``grep -rn`` matches for a query in the repository.

    ``_max_chars`` bounds the total characters, because a single matching line
    can be megabytes long, and appends a truncation marker.
    """
    root = repository_root(repo_path)
    if not query:
        raise ValueError("Search query must not be empty.")

    result = subprocess.run(
        ["grep", "-rn", "--", query, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise ValueError(f"Code search failed: {result.stderr.strip()}")
    lines = result.stdout.splitlines()
    selected: list[str] = []
    used_chars = 0
    truncated = len(lines) > 50
    for line in lines[:50]:
        if _max_chars is not None and used_chars + len(line) > _max_chars:
            available = _max_chars - used_chars
            if available > 0:
                selected.append(line[:available])
                used_chars += available
            truncated = True
            break
        selected.append(line)
        used_chars += len(line) + 1
    if not truncated:
        return selected
    remaining = max(0, len(result.stdout) - used_chars)
    return [*selected, f"[truncated: {remaining} more chars]"]


def _parse_proposal(response: object) -> _FixProposal:
    """Extract Gemini's structured response or report an exhausted tool loop."""
    return parse_structured_response(response, _FixProposal)


def propose_fix(
    repo_path: str,
    issue_title: str,
    issue_body: str,
    client: object,
    *,
    test_command: list[str] | None = None,
    protected_paths: list[str] | None = None,
    exploration_steps: list[ExplorationStep] | None = None,
) -> FixResult:
    """Use llm tool calling to propose, but not apply, a bug fix."""
    root = repository_root(repo_path)
    if not issue_title.strip():
        raise ValueError("Issue title must be provided.")

    provider = coerce_provider(client)
    max_result_chars = _tool_result_char_limit(provider)
    steps = exploration_steps if exploration_steps is not None else []

    def repository_read_file(file_path: str) -> str:
        """Read a file relative to the repository being investigated."""
        try:
            content = read_file(str(root), file_path, _max_chars=max_result_chars)
        except Exception as error:
            _record_exploration(
                steps, "read_file", {"path": file_path}, 0, "characters",
                error=str(error),
            )
            raise
        _record_exploration(
            steps, "read_file", {"path": file_path}, len(content), "characters"
        )
        return content

    def repository_list_directory(dir_path: str = ".") -> list[str]:
        """List a directory relative to the repository being investigated."""
        try:
            entries = list_directory(
                str(root), dir_path, _max_chars=max_result_chars
            )
        except Exception as error:
            _record_exploration(
                steps, "list_directory", {"dir_path": dir_path}, 0, "entries",
                error=str(error),
            )
            raise
        _record_exploration(
            steps, "list_directory", {"dir_path": dir_path}, len(entries), "entries"
        )
        return entries

    def repository_search_code(query: str) -> list[str]:
        """Search source code relative to the repository being investigated."""
        try:
            matches = search_code(str(root), query, _max_chars=max_result_chars)
        except Exception as error:
            _record_exploration(
                steps, "search_code", {"query": query}, 0, "lines",
                error=str(error),
            )
            raise
        _record_exploration(
            steps, "search_code", {"query": query}, len(matches), "lines"
        )
        return matches

    exploration, _, exploration_rounds = provider.explore(
        _SYSTEM_INSTRUCTION,
        f"Issue title: {issue_title}\n\nIssue body:\n{issue_body or '(No issue body was provided.)'}",
        [repository_read_file, repository_list_directory, repository_search_code],
    )
    if hasattr(exploration, "get_history"):
        contents = _history_without_unanswered_calls(exploration.get_history()) + [
            types.Content(role="user", parts=[types.Part.from_text(text=(
                "Now output your proposed fix as structured JSON matching "
                "_FixProposal: file_path, new_content, reasoning."
            ))])
        ]
    else:
        contents = exploration + [{"role": "user", "content": (
            "Now output your proposed fix as JSON with file_path, new_content, reasoning."
        )}]

    response = provider.structured(
        _SYSTEM_INSTRUCTION,
        contents,
        _FixProposal,
        "fix_generation",
    )

    # Local validation: reject hallucinated paths and protected test files
    # before any repository content is read into a proposal.
    proposal = _parse_proposal(response)

    rejection = proposal_rejection(
        str(root), proposal.file_path, test_command, protected_paths
    )
    if rejection is not None:
        kind, reason = rejection
        raise RejectedProposalError(proposal.file_path, kind, reason)

    original_content = read_file(str(root), proposal.file_path)

    return FixResult(
        file_path=proposal.file_path,
        original_content=original_content,
        new_content=proposal.new_content,
        reasoning=proposal.reasoning,
        exploration_rounds=exploration_rounds,
    )
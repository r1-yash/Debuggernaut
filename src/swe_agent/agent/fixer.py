"""Explore a cloned repository and propose a full-file bug fix with Gemini."""

from __future__ import annotations

import subprocess
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from swe_agent.llm import coerce_provider
from swe_agent.agent.paths import repository_root, resolve_repository_path
from swe_agent.agent.structured import parse_structured_response

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
- Add focused tests when necessary to reproduce and validate the bug.
- Run the narrowest relevant tests first, then the broader test suite when
  practical.
- Do not declare success based solely on code inspection.

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


class _FixProposal(BaseModel):
    """The fields Gemini must return before local file content is attached."""

    file_path: str
    new_content: str
    reasoning: str


def read_file(repo_path: str, file_path: str) -> str:
    """Return a repository file's contents without allowing path traversal."""
    path = resolve_repository_path(repo_path, file_path)
    if not path.is_file():
        raise ValueError(f"File does not exist: {file_path}")
    return path.read_text(encoding="utf-8")


def list_directory(repo_path: str, dir_path: str = ".") -> list[str]:
    """Return the names directly contained in a repository directory."""
    path = resolve_repository_path(repo_path, dir_path)
    if not path.is_dir():
        raise ValueError(f"Directory does not exist: {dir_path}")
    return sorted(entry.name for entry in path.iterdir())


def search_code(repo_path: str, query: str) -> list[str]:
    """Return up to 50 ``grep -rn`` matches for a query in the repository."""
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
    return result.stdout.splitlines()[:50]


def _parse_proposal(response: object) -> _FixProposal:
    """Extract Gemini's structured response or report an exhausted tool loop."""
    return parse_structured_response(response, _FixProposal)


def propose_fix(
    repo_path: str,
    issue_title: str,
    issue_body: str,
    client: object,
    *,
    exploration_steps: list[ExplorationStep] | None = None,
) -> FixResult:
    """Use llm tool calling to propose, but not apply, a bug fix."""
    root = repository_root(repo_path)
    if not issue_title.strip():
        raise ValueError("Issue title must be provided.")

    steps = exploration_steps if exploration_steps is not None else []

    def repository_read_file(file_path: str) -> str:
        """Read a file relative to the repository being investigated."""
        try:
            content = read_file(str(root), file_path)
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
            entries = list_directory(str(root), dir_path)
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
            matches = search_code(str(root), query)
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

    provider = coerce_provider(client)
    exploration, _ = provider.explore(
        _SYSTEM_INSTRUCTION,
        f"Issue title: {issue_title}\n\nIssue body:\n{issue_body or '(No issue body was provided.)'}",
        [repository_read_file, repository_list_directory, repository_search_code],
    )
    if hasattr(exploration, "get_history"):
        contents = exploration.get_history() + [
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

## local validation step that checks whether the LLM-proposed file_path actually exists in the repository before reading or modifying it,
# preventing hallucinated/stale paths from crashing the agent.
    proposal = _parse_proposal(response)

    target_path = resolve_repository_path(str(root), proposal.file_path)

    if not target_path.is_file():
        raise ValueError(
            f"LLM proposed a file that does not exist: {proposal.file_path}"
        )

    original_content = read_file(str(root), proposal.file_path)

    return FixResult(
        file_path=proposal.file_path,
        original_content=original_content,
        new_content=proposal.new_content,
        reasoning=proposal.reasoning,
    )
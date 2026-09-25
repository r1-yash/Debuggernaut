"""Explore a cloned repository and propose a full-file bug fix with Gemini."""

from __future__ import annotations

import subprocess

from google import genai
from google.genai import types
from pydantic import BaseModel

from swe_agent.agent.paths import repository_root, resolve_repository_path
from swe_agent.agent.structured import parse_structured_response


_SYSTEM_INSTRUCTION = """You are a careful software-maintenance agent. Given a
GitHub issue, explore the repository with the supplied tools before proposing a
fix. Once you have enough information, respond with plain text confirming that
you are ready to provide your final fix. Do not call any more tools at that
point. The final fix will be requested separately."""


class FixResult(BaseModel):
    """A complete, un-applied file rewrite proposed for an issue."""

    file_path: str
    original_content: str
    new_content: str
    reasoning: str


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
    client: genai.Client,
) -> FixResult:
    """Use Gemini tool calling to propose, but not apply, a bug fix."""
    root = repository_root(repo_path)
    if not issue_title.strip():
        raise ValueError("Issue title must be provided.")

    def repository_read_file(file_path: str) -> str:
        """Read a file relative to the repository being investigated."""
        return read_file(str(root), file_path)

    def repository_list_directory(dir_path: str = ".") -> list[str]:
        """List a directory relative to the repository being investigated."""
        return list_directory(str(root), dir_path)

    def repository_search_code(query: str) -> list[str]:
        """Search source code relative to the repository being investigated."""
        return search_code(str(root), query)

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            tools=[
                repository_read_file,
                repository_list_directory,
                repository_search_code,
            ],
        ),
    )
    chat.send_message(
        f"Issue title: {issue_title}\n\n"
        f"Issue body:\n{issue_body or '(No issue body was provided.)'}"
    )
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=chat.get_history()
        + [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Now output your proposed fix as structured JSON matching "
                            "_FixProposal: file_path, new_content, reasoning."
                        )
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_FixProposal,
        ),
    )
    proposal = _parse_proposal(response)
    original_content = read_file(str(root), proposal.file_path)
    return FixResult(
        file_path=proposal.file_path,
        original_content=original_content,
        new_content=proposal.new_content,
        reasoning=proposal.reasoning,
    )

# Run `uv add streamlit` and `uv add python-dotenv` before launching this app.
"""Provide a Streamlit interface for resolving GitHub issues with SWE Agent."""

from __future__ import annotations

import shlex

import streamlit as st
from dotenv import load_dotenv
from google import genai

from swe_agent.agent.loop import Attempt, LoopResult, resolve_issue
from swe_agent.agent.pr import create_fix_pull_request
from swe_agent.ingestion.clone import clone_repository
from swe_agent.ingestion.issues import Issue, fetch_recent_issues
from swe_agent.ingestion.repo_url import parse_repository_url


load_dotenv()


def _issue_label(issue: Issue) -> str:
    """Return a concise, selectable description of an issue."""
    preview = " ".join((issue.body or "No description provided.").split())
    if len(preview) > 120:
        preview = f"{preview[:117]}..."
    return f"#{issue.number} — {issue.title} — {preview}"


def _display_attempts(loop_result: LoopResult) -> None:
    """Render the proposed change and captured output for every attempt."""
    st.write(f"Attempts: {len(loop_result.attempts)}")
    for number, attempt in enumerate(loop_result.attempts, start=1):
        _display_attempt(number, attempt)


def _display_attempt(number: int, attempt: Attempt) -> None:
    """Render one attempted fix in an expandable section."""
    with st.expander(f"Attempt {number}: {attempt.fix_result.file_path}"):
        st.markdown(f"**Changed file:** `{attempt.fix_result.file_path}`")
        st.markdown("**Reasoning:**")
        st.write(attempt.fix_result.reasoning)
        st.markdown("**Test stdout:**")
        st.code(attempt.outcome.stdout or "(no stdout)")
        st.markdown("**Test stderr:**")
        st.code(attempt.outcome.stderr or "(no stderr)")


def main() -> None:
    """Render the SWE Agent issue-selection and execution workflow."""
    st.set_page_config(page_title="SWE Agent")
    st.title("SWE Agent")
    st.write("Explore a GitHub issue, test a proposed fix, and optionally open a pull request.")

    repository_url = st.text_input(
        "GitHub repository URL",
        placeholder="https://github.com/owner/repository",
    )
    if st.button("Fetch Issues"):
        try:
            owner, repository = parse_repository_url(repository_url).split("/", maxsplit=1)
            st.session_state["repository_owner"] = owner
            st.session_state["repository_name"] = repository
            st.session_state["issues"] = fetch_recent_issues(owner, repository, limit=10)
        except Exception as error:
            st.error(str(error))

    issues: list[Issue] = st.session_state.get("issues", [])
    if not issues:
        return

    selected_issue = st.selectbox("Select an open issue", issues, format_func=_issue_label)
    test_command_text = st.text_input(
        "Test command",
        placeholder="python -m pytest test_calc.py",
    )
    open_pull_request = st.checkbox("Open a pull request if the fix succeeds")

    if not st.button("Run Agent"):
        return

    try:
        test_command = shlex.split(test_command_text)
        if not test_command:
            raise ValueError("Enter a test command before running the agent.")

        owner = st.session_state["repository_owner"]
        repository = st.session_state["repository_name"]
        with st.spinner(
            "Cloning repo, exploring, and attempting a fix — this can take a few minutes..."
        ):
            repo_path = clone_repository(owner, repository)
            loop_result = resolve_issue(
                repo_path,
                selected_issue.title,
                selected_issue.body or "",
                test_command,
                genai.Client(),
            )

        if loop_result.succeeded:
            st.success("The agent found a fix that passed the configured test command.")
        else:
            st.error("The agent could not find a passing fix within its retry limit.")
        _display_attempts(loop_result)

        if loop_result.succeeded and open_pull_request:
            pr_result = create_fix_pull_request(
                repo_path,
                owner,
                repository,
                selected_issue.number,
                loop_result,
            )
            st.markdown(f"[Open pull request #{pr_result.number}]({pr_result.url})")
    except Exception as error:
        st.error(str(error))


if __name__ == "__main__":
    main()

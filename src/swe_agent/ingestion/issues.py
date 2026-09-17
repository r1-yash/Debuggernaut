"""Fetch open GitHub issues and their comments for repository ingestion."""

from __future__ import annotations

import os
from typing import Any

import requests
from pydantic import BaseModel


_GITHUB_API_URL = "https://api.github.com"
_REQUEST_TIMEOUT_SECONDS = 10


class Issue(BaseModel):
    """The GitHub issue fields needed by the SWE Agent."""

    number: int
    title: str
    body: str | None
    url: str
    comments_url: str
    comments_count: int


def _github_headers() -> dict[str, str]:
    """Build GitHub request headers, adding authentication when configured."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str) -> list[dict[str, Any]]:
    """Get a GitHub API collection or raise a readable error."""
    response = requests.get(
        url,
        headers=_github_headers(),
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        message = response.text
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("message"), str):
            message = payload["message"]
        raise ValueError(
            f"GitHub API request failed with status {response.status_code}: {message}"
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("GitHub API returned an unexpected response format.")
    return payload


def fetch_recent_issues(
    owner: str,
    repository: str,
    limit: int = 10,
) -> list[Issue]:
    """Return the most recently created open issues, excluding pull requests."""
    if not owner.strip() or not repository.strip():
        raise ValueError("GitHub owner and repository must be provided.")
    if not 1 <= limit <= 100:
        raise ValueError("Issue limit must be between 1 and 100.")

    url = (
        f"{_GITHUB_API_URL}/repos/{owner}/{repository}/issues"
        f"?state=open&sort=created&direction=desc&per_page={limit}" #from here can do open all and for closed too
    )
    issues = _get_json(url)
    return [
        Issue(
            number=item["number"],
            title=item["title"],
            body=item.get("body"),
            url=item.get("html_url", item["url"]),
            comments_url=item["comments_url"],
            comments_count=item["comments"],
        )
        for item in issues
        if "pull_request" not in item
    ]


def fetch_issue_comments(comments_url: str) -> list[str]:
    """Return the plain-text bodies from an issue's GitHub comment thread."""
    if not comments_url.strip():
        raise ValueError("A GitHub comments URL must be provided.")

    comments = _get_json(comments_url)
    return [comment.get("body") or "" for comment in comments]

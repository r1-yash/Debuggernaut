"""Resolve the GitHub issue closed by a pull request."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests


RESOLUTION_STATUSES = (
    "resolved",
    "no_issue",
    "pr_reference",
    "title_mismatch",
    "unresolved",
)
ResolutionStatus = Literal[
    "resolved",
    "no_issue",
    "pr_reference",
    "title_mismatch",
    "unresolved",
]

_API_ROOT = "https://api.github.com"
_USER_AGENT = "swe-agent-candidate-builder"
_CLOSING_BLOCK = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*"
    r"(?P<refs>(?:(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#\d+"
    r"(?:\s*(?:,|and)\s*)?)+)"
)
_REFERENCE = re.compile(
    r"(?:(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+))?"
    r"#(?P<number>\d+)"
)
_GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 50) {
        nodes { number }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one pull request to an issue."""

    status: ResolutionStatus
    issue_number: int | None = None
    issue_title: str | None = None
    dataset_title: str | None = None
    issue_titles: tuple[str, ...] = ()
    pr_reference_numbers: tuple[int, ...] = ()
    detail: str | None = None


def normalize_title(text: str) -> str:
    """Collapse all whitespace in a title for comparison."""
    return " ".join(text.split())


def first_line_title(problem_statement: str) -> str:
    """Return the first line of a problem statement, if present."""
    lines = problem_statement.splitlines()
    return lines[0] if lines else ""


def parse_closing_issue_references(
    body: str, repo: str
) -> tuple[int, ...]:
    """Parse same-repository closing keywords from a pull request body."""
    references: list[int] = []
    for block in _CLOSING_BLOCK.finditer(body or ""):
        for match in _REFERENCE.finditer(block.group("refs")):
            referenced_repo = match.group("repo")
            if (
                referenced_repo is not None
                and referenced_repo.casefold() != repo.casefold()
            ):
                continue
            number = int(match.group("number"))
            if number > 0 and number not in references:
                references.append(number)
    return tuple(references)


class IssueResolver:
    """Resolve pull requests with GitHub GraphQL/REST and a local cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        offline: bool = False,
        token: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_path = Path(cache_dir) / "issue_cache.json"
        self.offline = offline
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.session = session or requests.Session()
        self._references, self._issues = self._load_cache()
        self._dirty = False

    def resolve(
        self,
        repo: str,
        pr_number: int,
        problem_statement: str,
    ) -> Resolution:
        """Resolve one PR to a matching same-repo issue, using the cache."""
        dataset_title = normalize_title(first_line_title(problem_statement))
        try:
            references = self._get_references(repo, pr_number)
            if references is None:
                return Resolution(
                    "unresolved",
                    dataset_title=dataset_title,
                    detail=(
                        "offline reference cache miss"
                        if self.offline
                        else "GitHub reference lookup failed"
                    ),
                )
            if not references:
                return Resolution(
                    "no_issue",
                    dataset_title=dataset_title,
                )

            issue_titles: list[str] = []
            pr_references: list[int] = []
            fetch_failed = False
            matched_number: int | None = None
            matched_title: str | None = None
            for number in references:
                issue = self._get_issue(repo, number)
                if issue is None:
                    fetch_failed = True
                    continue
                if issue["is_pull_request"]:
                    pr_references.append(number)
                    continue
                issue_title = normalize_title(str(issue["title"]))
                issue_titles.append(issue_title)
                if (
                    matched_number is None
                    and issue_title == dataset_title
                ):
                    matched_number = number
                    matched_title = issue_title

            if matched_number is not None:
                return Resolution(
                    "resolved",
                    issue_number=matched_number,
                    issue_title=matched_title,
                    dataset_title=dataset_title,
                    issue_titles=tuple(issue_titles),
                    pr_reference_numbers=tuple(pr_references),
                )
            if issue_titles:
                return Resolution(
                    "title_mismatch",
                    dataset_title=dataset_title,
                    issue_titles=tuple(issue_titles),
                    pr_reference_numbers=tuple(pr_references),
                )
            if fetch_failed:
                return Resolution(
                    "unresolved",
                    dataset_title=dataset_title,
                    pr_reference_numbers=tuple(pr_references),
                    detail=(
                        "offline issue cache miss"
                        if self.offline
                        else "GitHub issue lookup failed"
                    ),
                )
            if pr_references:
                return Resolution(
                    "pr_reference",
                    dataset_title=dataset_title,
                    pr_reference_numbers=tuple(pr_references),
                    detail="all closing references are pull requests",
                )
            return Resolution(
                "unresolved",
                dataset_title=dataset_title,
                detail="no closing issue data",
            )
        finally:
            if self._dirty:
                self.save()

    def save(self) -> None:
        """Write the issue cache atomically, if it changed."""
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pr_refs": {
                key: list(value)
                for key, value in sorted(self._references.items())
            },
            "issues": {
                key: dict(value)
                for key, value in sorted(self._issues.items())
            },
        }
        temporary = self.cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.cache_path)
        self._dirty = False

    def _get_references(
        self, repo: str, pr_number: int
    ) -> tuple[int, ...] | None:
        key = _cache_key(repo, pr_number)
        if key in self._references:
            return self._references[key]
        if self.offline:
            return None

        graphql_refs = self._graphql_references(repo, pr_number)
        if graphql_refs:
            references = graphql_refs
        else:
            rest_refs = self._rest_references(repo, pr_number)
            if rest_refs is not None:
                references = rest_refs
            elif graphql_refs is not None:
                references = graphql_refs
            else:
                return None

        references = tuple(sorted(set(references)))
        self._references[key] = references
        self._dirty = True
        return references

    def _graphql_references(
        self, repo: str, pr_number: int
    ) -> tuple[int, ...] | None:
        if not self.token:
            return None
        owner, separator, name = repo.partition("/")
        if not separator or not owner or not name:
            return None
        try:
            response = self.session.post(
                f"{_API_ROOT}/graphql",
                headers=self._headers(),
                json={
                    "query": _GRAPHQL_QUERY,
                    "variables": {
                        "owner": owner,
                        "name": name,
                        "number": pr_number,
                    },
                },
                timeout=60,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict) or payload.get("errors"):
            return None
        try:
            nodes = (
                payload["data"]["repository"]["pullRequest"]
                ["closingIssuesReferences"]["nodes"]
            )
        except (KeyError, TypeError):
            return None
        if not isinstance(nodes, list):
            return None
        references: list[int] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            number = node.get("number")
            if isinstance(number, int) and not isinstance(number, bool):
                if number > 0 and number not in references:
                    references.append(number)
        return tuple(references)

    def _rest_references(
        self, repo: str, pr_number: int
    ) -> tuple[int, ...] | None:
        try:
            response = self.session.get(
                f"{_API_ROOT}/repos/{repo}/pulls/{pr_number}",
                headers=self._headers(),
                timeout=60,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        body = payload.get("body")
        if not isinstance(body, str):
            body = ""
        return parse_closing_issue_references(body, repo)

    def _get_issue(
        self, repo: str, issue_number: int
    ) -> dict[str, object] | None:
        key = _cache_key(repo, issue_number)
        if key in self._issues:
            return self._issues[key]
        if self.offline:
            return None

        try:
            response = self.session.get(
                f"{_API_ROOT}/repos/{repo}/issues/{issue_number}",
                headers=self._headers(),
                timeout=60,
            )
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        title = payload.get("title")
        issue = {
            "title": title if isinstance(title, str) else "",
            "is_pull_request": "pull_request" in payload,
        }
        self._issues[key] = issue
        self._dirty = True
        return issue

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _load_cache(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, dict[str, object]]]:
        if not self.cache_path.is_file():
            return {}, {}
        try:
            payload: Any = json.loads(
                self.cache_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Could not read issue cache: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("Issue cache must be a JSON object.")
        if payload.get("version") not in (None, 1):
            raise ValueError("Unsupported issue cache version.")

        references: dict[str, tuple[int, ...]] = {}
        raw_references = payload.get("pr_refs", {})
        if not isinstance(raw_references, dict):
            raise ValueError("Issue cache pr_refs must be an object.")
        for key, values in raw_references.items():
            if not isinstance(key, str) or not isinstance(values, list):
                raise ValueError("Issue cache has malformed pr_refs.")
            if not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in values
            ):
                raise ValueError("Issue cache has malformed pr_refs.")
            references[key] = tuple(sorted(set(values)))

        issues: dict[str, dict[str, object]] = {}
        raw_issues = payload.get("issues", {})
        if not isinstance(raw_issues, dict):
            raise ValueError("Issue cache issues must be an object.")
        for key, value in raw_issues.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, dict)
                or not isinstance(value.get("title"), str)
                or not isinstance(value.get("is_pull_request"), bool)
            ):
                raise ValueError("Issue cache has malformed issues.")
            issues[key] = {
                "title": value["title"],
                "is_pull_request": value["is_pull_request"],
            }
        return references, issues


def _cache_key(repo: str, number: int) -> str:
    """Return the cache key for a repository and issue/PR number."""
    return f"{repo}#{number}"

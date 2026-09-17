import os
import unittest
from unittest.mock import Mock, patch

from swe_agent.ingestion.issues import fetch_issue_comments, fetch_recent_issues


def response(status_code: int, payload: object, text: str = "") -> Mock:
    mocked_response = Mock()
    mocked_response.status_code = status_code
    mocked_response.json.return_value = payload
    mocked_response.text = text
    return mocked_response


class FetchRecentIssuesTests(unittest.TestCase):
    @patch("swe_agent.ingestion.issues.requests.get")
    def test_fetches_open_issues_including_expected_fields(self, get: Mock) -> None:
        get.return_value = response(
            200,
            [
                {
                    "number": 42,
                    "title": "Fix parser",
                    "body": "URLs with fragments fail.",
                    "html_url": "https://github.com/acme/widget/issues/42",
                    "url": "https://api.github.com/repos/acme/widget/issues/42",
                    "comments_url": "https://api.github.com/repos/acme/widget/issues/42/comments",
                    "comments": 3,
                }
            ],
        )

        issues = fetch_recent_issues("acme", "widget", limit=5)

        self.assertEqual(issues[0].number, 42)
        self.assertEqual(issues[0].url, "https://github.com/acme/widget/issues/42")
        get.assert_called_once_with(
            "https://api.github.com/repos/acme/widget/issues"
            "?state=open&sort=created&direction=desc&per_page=5",
            headers={"Accept": "application/vnd.github+json"},
            timeout=10,
        )

    @patch("swe_agent.ingestion.issues.requests.get")
    def test_filters_pull_requests(self, get: Mock) -> None:
        get.return_value = response(
            200,
            [
                {
                    "number": 1,
                    "title": "Issue",
                    "body": None,
                    "url": "https://api.github.com/repos/acme/widget/issues/1",
                    "comments_url": "https://api.github.com/repos/acme/widget/issues/1/comments",
                    "comments": 0,
                },
                {"number": 2, "pull_request": {}},
            ],
        )

        issues = fetch_recent_issues("acme", "widget")

        self.assertEqual([issue.number for issue in issues], [1])

    @patch.dict(os.environ, {}, clear=True)
    @patch("swe_agent.ingestion.issues.requests.get")
    def test_works_without_a_token(self, get: Mock) -> None:
        get.return_value = response(200, [])

        self.assertEqual(fetch_recent_issues("acme", "widget"), [])
        self.assertNotIn("Authorization", get.call_args.kwargs["headers"])

    @patch.dict(os.environ, {"GITHUB_TOKEN": "invalid-token"}, clear=True)
    @patch("swe_agent.ingestion.issues.requests.get")
    def test_reports_invalid_token_response(self, get: Mock) -> None:
        get.return_value = response(401, {"message": "Bad credentials"})

        with self.assertRaisesRegex(ValueError, "status 401: Bad credentials"):
            fetch_recent_issues("acme", "widget")
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"],
            "Bearer invalid-token",
        )

    @patch("swe_agent.ingestion.issues.requests.get")
    def test_reports_non_200_api_errors(self, get: Mock) -> None:
        get.return_value = response(404, {"message": "Not Found"})

        with self.assertRaisesRegex(ValueError, "status 404: Not Found"):
            fetch_recent_issues("missing", "repository")


class FetchIssueCommentsTests(unittest.TestCase):
    @patch("swe_agent.ingestion.issues.requests.get")
    def test_returns_comment_bodies(self, get: Mock) -> None:
        get.return_value = response(200, [{"body": "First"}, {"body": "Second"}])

        self.assertEqual(
            fetch_issue_comments("https://api.github.com/repos/acme/widget/issues/1/comments"),
            ["First", "Second"],
        )

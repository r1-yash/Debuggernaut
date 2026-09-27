import unittest
from unittest.mock import Mock, patch

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import Attempt, LoopResult
from swe_agent.agent.pr import PullRequestResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.runner import run_single_entry
from swe_agent.ingestion.issues import Issue


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = BenchmarkEntry(
            owner="acme", repository="widget", issue_number=42, test_command=["pytest"]
        )
        self.issue = Issue(
            number=42,
            title="Fix parser",
            body="Parser should return a mapping.",
            url="https://github.com/acme/widget/issues/42",
            comments_url="https://api.github.com/repos/acme/widget/issues/42/comments",
            comments_count=0,
        )

    def _loop_result(self, succeeded: bool) -> LoopResult:
        fix_result = FixResult(
            file_path="parser.py",
            original_content="old\n",
            new_content="new\n",
            reasoning="Return a mapping.",
        )
        attempt = Attempt(
            fix_result=fix_result,
            outcome=TestOutcome(
                passed=succeeded,
                return_code=0 if succeeded else 1,
                stdout="",
                stderr="",
            ),
        )
        return LoopResult(succeeded=succeeded, final_attempt=attempt, attempts=[attempt])

    @patch("swe_agent.benchmark.runner.create_fix_pull_request")
    @patch("swe_agent.benchmark.runner.resolve_issue")
    @patch("swe_agent.benchmark.runner.fetch_issue")
    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_successful_entry_opens_pull_request(
        self, clone: Mock, fetch_issue: Mock, resolve: Mock, create_pr: Mock
    ) -> None:
        clone.return_value = "/tmp/widget"
        fetch_issue.return_value = self.issue
        loop_result = self._loop_result(True)
        resolve.return_value = loop_result
        create_pr.return_value = PullRequestResult(
            url="https://github.com/acme/widget/pull/7", number=7, branch_name="fix"
        )

        outcome = run_single_entry(self.entry, Mock(), open_pr=True)

        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.loop_result.succeeded)
        self.assertEqual(outcome.pr_result.number, 7)
        create_pr.assert_called_once_with(
            "/tmp/widget", "acme", "widget", 42, loop_result
        )

    @patch("swe_agent.benchmark.runner.create_fix_pull_request")
    @patch("swe_agent.benchmark.runner.resolve_issue")
    @patch("swe_agent.benchmark.runner.fetch_issue")
    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_successful_entry_skips_pull_request_when_disabled(
        self, clone: Mock, fetch_issue: Mock, resolve: Mock, create_pr: Mock
    ) -> None:
        clone.return_value = "/tmp/widget"
        fetch_issue.return_value = self.issue
        resolve.return_value = self._loop_result(True)

        outcome = run_single_entry(self.entry, Mock(), open_pr=False)

        self.assertTrue(outcome.loop_result.succeeded)
        self.assertIsNone(outcome.pr_result)
        create_pr.assert_not_called()

    @patch("swe_agent.benchmark.runner.create_fix_pull_request")
    @patch("swe_agent.benchmark.runner.resolve_issue")
    @patch("swe_agent.benchmark.runner.fetch_issue")
    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_failed_entry_does_not_open_pull_request(
        self, clone: Mock, fetch_issue: Mock, resolve: Mock, create_pr: Mock
    ) -> None:
        clone.return_value = "/tmp/widget"
        fetch_issue.return_value = self.issue
        resolve.return_value = self._loop_result(False)

        outcome = run_single_entry(self.entry, Mock(), open_pr=True)

        self.assertFalse(outcome.loop_result.succeeded)
        self.assertIsNone(outcome.pr_result)
        create_pr.assert_not_called()

    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_records_entry_exception_without_raising(self, clone: Mock) -> None:
        clone.side_effect = ValueError("Git clone failed")

        outcome = run_single_entry(self.entry, Mock())

        self.assertFalse(outcome.loop_result.succeeded)
        self.assertEqual(outcome.error, "Git clone failed")
        self.assertIsNone(outcome.pr_result)


if __name__ == "__main__":
    unittest.main()

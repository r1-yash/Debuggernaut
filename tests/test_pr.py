import os
import subprocess
import unittest
from unittest.mock import Mock, patch

import responses

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import Attempt, LoopResult
from swe_agent.agent.pr import _ensure_fork, create_fix_pull_request


class PullRequestTests(unittest.TestCase):
    def _loop_result(self, succeeded: bool = True) -> LoopResult:
        fix_result = FixResult(
            file_path="parser.py",
            original_content="old\n",
            new_content="new\n",
            reasoning="Return an empty mapping.",
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
        return LoopResult(
            succeeded=succeeded,
            final_attempt=attempt if succeeded else None,
            attempts=[attempt],
        )

    @responses.activate
    @patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}, clear=True)
    @patch("swe_agent.agent.pr.subprocess.run")
    def test_creates_pull_request(
        self, run: Mock
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 1, stdout="", stderr="No such remote"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=" M parser.py\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
        responses.add(
            responses.POST,
            "https://api.github.com/repos/acme/widget/forks",
            json={"owner": {"login": "fork-owner"}},
            status=202,
        )
        responses.add(
            responses.GET,
            "https://api.github.com/repos/acme/widget",
            json={"default_branch": "main"},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.github.com/repos/acme/widget/pulls",
            json={"html_url": "https://github.com/acme/widget/pull/7", "number": 7},
            status=201,
        )

        result = create_fix_pull_request(
            "repository", "acme", "widget", 42, self._loop_result()
        )

        self.assertEqual(result.url, "https://github.com/acme/widget/pull/7")
        self.assertEqual(result.number, 7)
        self.assertEqual(result.branch_name, "swe-agent/fix-issue-42")
        self.assertEqual(run.call_count, 7)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["git", "remote", "remove", "swe-agent-fork"],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [
                "git",
                "remote",
                "add",
                "swe-agent-fork",
                "https://github.com/fork-owner/widget.git",
            ],
        )
        self.assertEqual(
            run.call_args_list[3].args[0],
            ["git", "checkout", "-b", "swe-agent/fix-issue-42"],
        )
        self.assertEqual(
            run.call_args_list[5].args[0],
            ["git", "commit", "-m", "Fix #42"],
        )
        self.assertEqual(
            run.call_args_list[6].args[0],
            ["git", "push", "-u", "swe-agent-fork", "swe-agent/fix-issue-42"],
        )
        post_request = responses.calls[2].request
        self.assertEqual(post_request.headers["Authorization"], "Bearer test-token")
        post_body = post_request.body.decode()
        self.assertIn('"base": "main"', post_body)
        self.assertIn('"head": "fork-owner:swe-agent/fix-issue-42"', post_body)
        self.assertIn("Fixes #42", post_body)

    @responses.activate
    def test_uses_current_user_when_existing_fork_omits_owner(self) -> None:
        responses.add(
            responses.POST,
            "https://api.github.com/repos/acme/widget/forks",
            json={},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.github.com/user",
            json={"login": "existing-fork-owner"},
            status=200,
        )

        self.assertEqual(_ensure_fork("acme", "widget"), "existing-fork-owner")

    @patch("swe_agent.agent.pr.subprocess.run")
    def test_rejects_unsuccessful_loop_result(self, run: Mock) -> None:
        with self.assertRaisesRegex(ValueError, "successful final attempt"):
            create_fix_pull_request(
                "repository", "acme", "widget", 42, self._loop_result(False)
            )
        run.assert_not_called()

    @responses.activate
    @patch("swe_agent.agent.pr.subprocess.run")
    def test_reports_github_pull_request_error(self, run: Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 1, stdout="", stderr="No such remote"),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=" M parser.py\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
        responses.add(
            responses.POST,
            "https://api.github.com/repos/acme/widget/forks",
            json={"owner": {"login": "fork-owner"}},
            status=202,
        )
        responses.add(
            responses.GET,
            "https://api.github.com/repos/acme/widget",
            json={"default_branch": "main"},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.github.com/repos/acme/widget/pulls",
            json={"message": "Validation Failed"},
            status=422,
        )

        with self.assertRaisesRegex(ValueError, "status 422: Validation Failed"):
            create_fix_pull_request(
                "repository", "acme", "widget", 42, self._loop_result()
            )


if __name__ == "__main__":
    unittest.main()

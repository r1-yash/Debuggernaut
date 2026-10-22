import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import Attempt, GuardEvent, LoopResult
from swe_agent.agent.pr import PullRequestResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.runner import run_benchmark, run_single_entry
from swe_agent.ingestion.issues import Issue


def _git(repo_path: Path, *arguments: str) -> str:
    """Run a git command in a repository and return its stripped stdout."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_repository(path: Path) -> Path:
    """Create a tiny committed repository for patch application tests."""
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "parser.py").write_text("old\n")
    _git(path, "add", "parser.py")
    _git(path, "commit", "-q", "-m", "base")
    return path


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = BenchmarkEntry(
            owner="acme",
            repository="widget",
            issue_number=42,
            commit="a" * 40,
            test_command=["pytest"],
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
            "/tmp/widget", "acme", "widget", 42, loop_result,
            test_patch_applied=False,
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

    @patch("swe_agent.benchmark.runner.create_fix_pull_request")
    @patch("swe_agent.benchmark.runner.resolve_issue")
    @patch("swe_agent.benchmark.runner.fetch_issue")
    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_guard_exhausted_entry_is_failed_not_errored(
        self, clone: Mock, fetch_issue: Mock, resolve: Mock, create_pr: Mock
    ) -> None:
        clone.return_value = "/tmp/widget"
        fetch_issue.return_value = self.issue
        resolve.return_value = LoopResult(
            succeeded=False,
            final_attempt=None,
            attempts=[],
            guard_events=[
                GuardEvent(
                    attempt=1,
                    file_path="tests/test_parser.py",
                    kind="test_file",
                    reason="Test file.",
                )
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            outcome = run_single_entry(self.entry, Mock(), report_dir=directory)
            payload = json.loads(Path(outcome.report_path).read_text())

        self.assertIsNone(outcome.error)
        self.assertEqual(payload["status"], "failed")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["attempts"], 0)
        self.assertEqual(len(payload["guard_events"]), 1)
        self.assertEqual(payload["guard_events"][0]["kind"], "test_file")
        create_pr.assert_not_called()

    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_applies_test_patch_before_resolving(self, clone: Mock) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        repository = _git_repository(Path(root.name) / "widget")
        diff_file = Path(root.name) / "test.diff"
        diff_file.write_text(
            """diff --git a/tests/test_parser.py b/tests/test_parser.py
new file mode 100644
--- /dev/null
+++ b/tests/test_parser.py
@@ -0,0 +1,2 @@
+def test_parser():
+    assert False
"""
        )
        clone.return_value = str(repository)
        entry = self.entry.model_copy(update={"test_patch": str(diff_file)})

        with patch(
            "swe_agent.benchmark.runner.fetch_issue", return_value=self.issue
        ), patch(
            "swe_agent.benchmark.runner.resolve_issue"
        ) as resolve:
            captured: dict = {}

            def resolve_side_effect(*args, **kwargs):
                captured["repo_path"] = args[0]
                captured["protected_paths"] = kwargs["protected_paths"]
                return self._loop_result(True)

            resolve.side_effect = resolve_side_effect
            with tempfile.TemporaryDirectory() as directory:
                outcome = run_single_entry(
                    entry, Mock(), report_dir=directory
                )
                payload = json.loads(Path(outcome.report_path).read_text())

        self.assertTrue(
            (Path(captured["repo_path"]) / "tests" / "test_parser.py").is_file()
        )
        self.assertEqual(captured["protected_paths"], ["tests/test_parser.py"])
        self.assertTrue(outcome.test_patch.applied)
        self.assertEqual(
            payload["test_patch"],
            {"applied": True, "file_paths": ["tests/test_parser.py"]},
        )
        self.assertEqual(payload["status"], "success")

    @patch("swe_agent.benchmark.runner.fetch_issue")
    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_test_patch_failure_marks_entry_errored(
        self, clone: Mock, fetch_issue: Mock
    ) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        repository = _git_repository(Path(root.name) / "widget")
        diff_file = Path(root.name) / "bad.diff"
        diff_file.write_text(
            """diff --git a/parser.py b/parser.py
--- a/parser.py
+++ b/parser.py
@@ -1 +1 @@
-missing
+new
"""
        )
        clone.return_value = str(repository)
        entry = self.entry.model_copy(update={"test_patch": str(diff_file)})

        with tempfile.TemporaryDirectory() as directory:
            outcome = run_single_entry(entry, Mock(), report_dir=directory)
            payload = json.loads(Path(outcome.report_path).read_text())

        self.assertIsNotNone(outcome.error)
        self.assertIn("Test patch failed to apply", outcome.error)
        fetch_issue.assert_not_called()
        self.assertEqual(payload["status"], "errored")
        self.assertIn("Test patch failed to apply", payload["error"])
        self.assertEqual(
            payload["test_patch"], {"applied": False, "file_paths": []}
        )

    def test_open_pr_is_refused_for_patched_entry(self) -> None:
        entry = self.entry.model_copy(update={"test_patch": "/tmp/test.diff"})

        with self.assertRaisesRegex(ValueError, "cannot open a pull request"):
            run_single_entry(entry, Mock(), open_pr=True)

    @patch("swe_agent.benchmark.runner.run_single_entry")
    def test_run_benchmark_refuses_open_pr_for_any_patched_entry(
        self, run_single: Mock
    ) -> None:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        (Path(root.name) / "test.diff").write_text("")
        config_path = Path(root.name) / "repos.json"
        config_path.write_text(
            json.dumps(
                {
                    "acme/widget": {
                        "issue_number": 1,
                        "commit": "a" * 40,
                        "test_command": ["pytest"],
                    },
                    "acme/gadget": {
                        "issue_number": 2,
                        "commit": "b" * 40,
                        "test_command": ["pytest"],
                        "test_patch": "test.diff",
                    },
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "open_pr is not supported"):
            run_benchmark(
                str(config_path),
                Mock(),
                workspace_dir=str(Path(root.name) / "workspace"),
                open_pr=True,
                report_dir=str(Path(root.name) / "runs"),
            )

        run_single.assert_not_called()

    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_records_entry_exception_without_raising(self, clone: Mock) -> None:
        clone.side_effect = ValueError("Git clone failed")

        outcome = run_single_entry(self.entry, Mock())

        self.assertFalse(outcome.loop_result.succeeded)
        self.assertEqual(outcome.error, "Git clone failed")
        self.assertIsNone(outcome.pr_result)

    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_error_report_uses_errored_status(self, clone: Mock) -> None:
        clone.side_effect = ValueError("Git clone failed")

        with tempfile.TemporaryDirectory() as directory:
            outcome = run_single_entry(self.entry, Mock(), report_dir=directory)
            payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(payload["status"], "errored")
        self.assertEqual(payload["error"], "Git clone failed")
        self.assertEqual(payload["commit"], "a" * 40)
        self.assertEqual(
            payload["test_patch"], {"applied": False, "file_paths": []}
        )

    @patch("swe_agent.benchmark.runner.clone_repository")
    def test_run_benchmark_writes_summary(self, clone: Mock) -> None:
        clone.side_effect = ValueError("Git clone failed")

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps({
                "acme/widget": {
                    "issue_number": 42,
                    "commit": "a" * 40,
                    "test_command": ["pytest"],
                }
            }))
            outcomes = run_benchmark(
                str(config_path), Mock(), report_dir=directory
            )
            summaries = list(Path(directory).glob("*_benchmark_summary.json"))
            self.assertEqual(len(summaries), 1)
            payload = json.loads(summaries[0].read_text())

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(
            payload["counts"],
            {"total": 1, "succeeded": 0, "failed": 0, "errored": 1},
        )
        self.assertEqual(payload["reports"], [outcomes[0].report_path])


if __name__ == "__main__":
    unittest.main()

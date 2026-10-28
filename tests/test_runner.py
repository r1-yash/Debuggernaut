import io
import json
import re
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import Attempt, GuardEvent, LoopResult, ResolutionInterrupted
from swe_agent.agent.pr import PullRequestResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.environment import (
    EnvironmentSetupError,
    EnvironmentSummary,
)
from swe_agent.benchmark.runner import _resumable_results, run_benchmark, run_single_entry
from swe_agent.ingestion.issues import Issue
from swe_agent.process import CommandResult
from swe_agent.llm import ProviderAuthError, ProviderUnavailable


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


class RecordingStream(io.StringIO):
    """Captured stdout that also counts how often it was flushed."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class FakeEnvironment:
    """Mocked isolated environment used by runner lifecycle tests."""

    def __init__(
        self,
        repo_path: str,
        *,
        test_timeout_seconds: float = 120.0,
        baseline_result: CommandResult | None = None,
        cleanup_error: str | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.test_timeout_seconds = test_timeout_seconds
        self.baseline_result = baseline_result or CommandResult(
            command=["pytest"],
            return_code=1,
            stdout="1 failed",
        )
        self.cleanup_error = cleanup_error
        self.summary = EnvironmentSummary()
        self.closed = False
        self.commands: list[list[str]] = []

    def run_test_command(
        self,
        command: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        self.commands.append(list(command))
        return self.baseline_result

    def close(self) -> None:
        self.closed = True
        if self.cleanup_error is not None:
            self.summary.cleanup_errors.append(self.cleanup_error)


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

    def _temp_directory(self) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        return Path(root.name)

    def _loop_result(
        self,
        succeeded: bool,
        *,
        timed_out: bool = False,
        timeout_seconds: float | None = None,
    ) -> LoopResult:
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
                return_code=0 if succeeded else (-1 if timed_out else 1),
                stdout="",
                stderr="",
                timed_out=timed_out,
                timeout_seconds=timeout_seconds,
            ),
        )
        return LoopResult(succeeded=succeeded, final_attempt=attempt, attempts=[attempt])

    def _run_harness(
        self,
        *,
        entry: BenchmarkEntry | None = None,
        environment: FakeEnvironment | None = None,
        loop_result: LoopResult | None = None,
        resolve_error: Exception | None = None,
        clone_error: Exception | None = None,
        setup_error: EnvironmentSetupError | None = None,
        report_dir: Path | None = None,
        open_pr: bool = False,
    ) -> tuple[object, SimpleNamespace]:
        entry = entry or self.entry
        report_dir = report_dir or self._temp_directory()
        environment = environment or FakeEnvironment("/tmp/widget")
        with patch(
            "swe_agent.benchmark.runner.clone_repository"
        ) as clone, patch(
            "swe_agent.benchmark.runner.fetch_issue"
        ) as fetch_issue, patch(
            "swe_agent.benchmark.runner.resolve_issue"
        ) as resolve_issue, patch(
            "swe_agent.benchmark.runner.EntryEnvironment.create"
        ) as create_environment, patch(
            "swe_agent.benchmark.runner.create_fix_pull_request"
        ) as create_pr:
            clone.return_value = str(environment.repo_path)
            if clone_error is not None:
                clone.side_effect = clone_error
            fetch_issue.return_value = self.issue
            resolve_issue.return_value = loop_result or self._loop_result(True)
            if resolve_error is not None:
                resolve_issue.side_effect = resolve_error
            create_environment.return_value = environment
            if setup_error is not None:
                create_environment.side_effect = setup_error
            create_pr.return_value = PullRequestResult(
                url="https://github.com/acme/widget/pull/7",
                number=7,
                branch_name="fix",
            )
            outcome = run_single_entry(
                entry,
                Mock(),
                workspace_dir=str(report_dir / "workspace"),
                report_dir=str(report_dir),
                open_pr=open_pr,
            )
        return outcome, SimpleNamespace(
            clone=clone,
            fetch_issue=fetch_issue,
            resolve_issue=resolve_issue,
            create_environment=create_environment,
            create_pr=create_pr,
            environment=environment,
            report_dir=report_dir,
        )

    def test_successful_entry_opens_pull_request(self) -> None:
        outcome, harness = self._run_harness(open_pr=True)

        self.assertEqual(outcome.status, "success")
        self.assertIsNone(outcome.failure_class)
        self.assertIsNone(outcome.error)
        self.assertTrue(outcome.loop_result.succeeded)
        self.assertEqual(outcome.pr_result.number, 7)
        harness.create_pr.assert_called_once_with(
            str(harness.environment.repo_path),
            "acme",
            "widget",
            42,
            outcome.loop_result,
            test_patch_applied=False,
        )
        self.assertTrue(harness.environment.closed)

    def test_successful_entry_skips_pull_request_when_disabled(self) -> None:
        outcome, harness = self._run_harness(open_pr=False)

        self.assertEqual(outcome.status, "success")
        self.assertTrue(outcome.loop_result.succeeded)
        self.assertIsNone(outcome.pr_result)
        harness.create_pr.assert_not_called()

    def test_failed_entry_does_not_open_pull_request(self) -> None:
        outcome, harness = self._run_harness(
            loop_result=self._loop_result(False),
            open_pr=True,
        )

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_class, "agent")
        self.assertFalse(outcome.loop_result.succeeded)
        self.assertIsNone(outcome.pr_result)
        harness.create_pr.assert_not_called()

    def test_guard_exhausted_entry_is_failed_not_errored(self) -> None:
        outcome, _harness = self._run_harness(
            loop_result=LoopResult(
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
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.failure_class, "agent")
        self.assertIsNone(outcome.error)
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["failure_class"], "agent")
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["attempts"], 0)
        self.assertEqual(len(payload["guard_events"]), 1)
        self.assertEqual(payload["guard_events"][0]["kind"], "test_file")

    def test_baseline_passes_unexpectedly_is_errored_and_skips_agent(self) -> None:
        environment = FakeEnvironment(
            "/tmp/widget",
            baseline_result=CommandResult(
                command=["pytest"],
                return_code=0,
                stdout="1 passed",
            ),
        )
        outcome, harness = self._run_harness(environment=environment)
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(outcome.status, "errored")
        self.assertEqual(outcome.failure_class, "infrastructure")
        self.assertEqual(outcome.baseline.status, "passes_unexpectedly")
        self.assertIn("Baseline passed unexpectedly", outcome.status_reason)
        self.assertEqual(payload["status"], "errored")
        self.assertEqual(payload["failure_class"], "infrastructure")
        self.assertEqual(payload["baseline"]["status"], "passes_unexpectedly")
        self.assertIn("passed unexpectedly", payload["status_reason"])
        harness.fetch_issue.assert_not_called()
        harness.resolve_issue.assert_not_called()

    def test_baseline_timeout_is_errored_infrastructure_and_skips_agent(self) -> None:
        environment = FakeEnvironment(
            "/tmp/widget",
            test_timeout_seconds=300.0,
            baseline_result=CommandResult(
                command=["pytest"],
                return_code=-1,
                timed_out=True,
            ),
        )
        outcome, harness = self._run_harness(environment=environment)

        self.assertEqual(outcome.status, "errored")
        self.assertEqual(outcome.failure_class, "infrastructure")
        self.assertEqual(outcome.baseline.status, "test_timeout")
        self.assertIn("300 s", outcome.status_reason)
        harness.fetch_issue.assert_not_called()
        harness.resolve_issue.assert_not_called()

    def test_post_agent_timeout_after_normal_baseline_is_test_timeout(self) -> None:
        outcome, harness = self._run_harness(
            loop_result=self._loop_result(
                False,
                timed_out=True,
                timeout_seconds=120.0,
            )
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(outcome.status, "test_timeout")
        self.assertEqual(outcome.failure_class, "agent")
        self.assertIn("tests timed out after 120 s", outcome.status_reason)
        self.assertEqual(payload["status"], "test_timeout")
        self.assertEqual(payload["failure_class"], "agent")
        self.assertIn("tests timed out after 120 s", payload["status_reason"])
        harness.resolve_issue.assert_called_once()

    def test_provider_interruption_report_retains_completed_attempts(self) -> None:
        partial = self._loop_result(False)
        outcome, _harness = self._run_harness(
            resolve_error=ResolutionInterrupted(
                ProviderUnavailable("rate limited"), partial.attempts, [],
            )
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(outcome.status, "provider_unavailable")
        self.assertEqual(payload["attempts"], 1)
        self.assertEqual(payload["attempt_records"][0]["file_path"], "parser.py")

    def test_cleanup_errors_appear_in_written_run_report(self) -> None:
        environment = FakeEnvironment(
            "/tmp/widget",
            cleanup_error="Could not remove /tmp/widget: PermissionError: denied",
        )
        outcome, _harness = self._run_harness(environment=environment)
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertTrue(environment.closed)
        self.assertEqual(
            payload["environment"]["cleanup_errors"],
            ["Could not remove /tmp/widget: PermissionError: denied"],
        )
        self.assertEqual(
            outcome.environment.cleanup_errors,
            ["Could not remove /tmp/widget: PermissionError: denied"],
        )

    def test_environment_setup_error_is_could_not_install(self) -> None:
        summary = EnvironmentSummary()
        outcome, harness = self._run_harness(
            setup_error=EnvironmentSetupError(
                "No installation attempt succeeded.",
                summary=summary,
                reason="could_not_install",
                status="could_not_install",
                infrastructure=False,
            )
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(outcome.status, "could_not_install")
        self.assertEqual(outcome.failure_class, "infrastructure")
        self.assertEqual(payload["status"], "could_not_install")
        self.assertEqual(payload["failure_class"], "infrastructure")
        self.assertEqual(payload["environment"]["cleanup_errors"], [])
        harness.fetch_issue.assert_not_called()

    def test_environment_setup_infrastructure_error_is_errored(self) -> None:
        summary = EnvironmentSummary()
        outcome, harness = self._run_harness(
            setup_error=EnvironmentSetupError(
                "uv is unavailable.",
                summary=summary,
                reason="uv_unavailable",
            )
        )

        self.assertEqual(outcome.status, "errored")
        self.assertEqual(outcome.failure_class, "infrastructure")
        harness.fetch_issue.assert_not_called()

    def test_applies_test_patch_before_baseline_and_resolving(self) -> None:
        root = self._temp_directory()
        repository = _git_repository(root / "widget")
        diff_file = root / "test.diff"
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
        environment = FakeEnvironment(str(repository))
        entry = self.entry.model_copy(update={"test_patch": str(diff_file)})
        outcome, harness = self._run_harness(
            entry=entry,
            environment=environment,
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertTrue((repository / "tests" / "test_parser.py").is_file())
        harness.resolve_issue.assert_called_once()
        self.assertEqual(
            harness.resolve_issue.call_args.kwargs["protected_paths"],
            ["tests/test_parser.py"],
        )
        self.assertTrue(outcome.test_patch.applied)
        self.assertEqual(
            payload["test_patch"],
            {"applied": True, "file_paths": ["tests/test_parser.py"]},
        )
        self.assertEqual(payload["status"], "success")

    def test_test_patch_failure_marks_entry_errored(self) -> None:
        root = self._temp_directory()
        repository = _git_repository(root / "widget")
        diff_file = root / "bad.diff"
        diff_file.write_text(
            """diff --git a/parser.py b/parser.py
--- a/parser.py
+++ b/parser.py
@@ -1 +1 @@
-missing
+new
"""
        )
        entry = self.entry.model_copy(update={"test_patch": str(diff_file)})
        outcome, harness = self._run_harness(
            entry=entry,
            environment=FakeEnvironment(str(repository)),
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertIsNotNone(outcome.error)
        self.assertIn("Test patch failed to apply", outcome.error)
        self.assertEqual(outcome.status, "errored")
        self.assertEqual(outcome.failure_class, "infrastructure")
        harness.fetch_issue.assert_not_called()
        self.assertEqual(payload["status"], "errored")
        self.assertIn("Test patch failed to apply", payload["error"])
        self.assertEqual(
            payload["test_patch"], {"applied": False, "file_paths": []}
        )

    def test_open_pr_is_refused_for_patched_entry(self) -> None:
        entry = self.entry.model_copy(update={"test_patch": "/tmp/test.diff"})

        with self.assertRaisesRegex(ValueError, "cannot open a pull request"):
            run_single_entry(entry, Mock(), open_pr=True)

    def test_progress_prints_started_and_finished_for_a_success(self) -> None:
        stream = RecordingStream()
        with redirect_stdout(stream):
            outcome, _harness = self._run_harness()

        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(
            lines[0],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z "
            r"acme/widget#42 started$",
        )
        self.assertRegex(
            lines[1],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z "
            r"acme/widget#42 finished status=success failure_class=- "
            r"attempts=1 exploration_rounds=0 total_tokens=- "
            r"duration_seconds=\d+\.\d{2} report="
            + re.escape(outcome.report_path)
            + "$",
        )
        self.assertGreaterEqual(stream.flush_count, 2)
        # Progress never repeats record content.
        self.assertNotIn("Return a mapping.", stream.getvalue())
        self.assertNotIn("@@", stream.getvalue())

    def test_progress_finish_line_for_an_errored_entry(self) -> None:
        stream = RecordingStream()
        with redirect_stdout(stream):
            outcome, _harness = self._run_harness(
                clone_error=ValueError("Git clone failed")
            )

        started, finished = stream.getvalue().splitlines()
        self.assertTrue(started.endswith("acme/widget#42 started"))
        self.assertIn("acme/widget#42 finished", finished)
        self.assertIn("status=errored", finished)
        self.assertIn("failure_class=infrastructure", finished)
        self.assertIn("attempts=0", finished)
        self.assertIn(f"report={outcome.report_path}", finished)
        self.assertNotIn("Git clone failed", finished)

    def test_progress_prints_one_skipped_line_for_a_recorded_entry(self) -> None:
        root = self._temp_directory()
        (root / "entry.json").write_text(json.dumps({
            "repository": "acme/widget", "issue": 42, "commit": "a" * 40,
            "provider": "tokenharbor", "model": "deepseek-test",
            "base_url_host": "tokenharbor.ai", "status": "success",
        }))
        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "acme/widget": {
                "issue_number": 42, "commit": "a" * 40,
                "test_command": ["pytest"],
            },
        }))
        provider = SimpleNamespace(
            provider="tokenharbor", model="deepseek-test",
            base_url_host="tokenharbor.ai",
        )
        stream = RecordingStream()
        with patch(
            "swe_agent.benchmark.runner.coerce_provider", return_value=provider,
        ), patch(
            "swe_agent.benchmark.runner.run_single_entry"
        ) as run_single, redirect_stdout(stream):
            outcomes = run_benchmark(
                str(config_path),
                Mock(),
                workspace_dir=str(root / "workspace"),
                report_dir=str(root),
            )

        lines = stream.getvalue().splitlines()
        skipped = [line for line in lines if "skipped" in line]
        self.assertEqual(len(skipped), 1)
        self.assertRegex(
            skipped[0],
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z acme/widget#42 skipped "
            r"\(recorded status: success\)$",
        )
        self.assertGreaterEqual(stream.flush_count, 1)
        self.assertEqual(outcomes, [])
        run_single.assert_not_called()

    def test_run_benchmark_refuses_open_pr_for_any_patched_entry(
        self,
    ) -> None:
        root = self._temp_directory()
        (root / "test.diff").write_text("")
        config_path = root / "repos.json"
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

        with patch(
            "swe_agent.benchmark.runner.run_single_entry"
        ) as run_single, patch(
            "swe_agent.benchmark.runner.sweep_stale_workspaces"
        ) as sweep:
            with self.assertRaisesRegex(ValueError, "open_pr is not supported"):
                run_benchmark(
                    str(config_path),
                    Mock(),
                    workspace_dir=str(root / "workspace"),
                    open_pr=True,
                    report_dir=str(root / "runs"),
                )

        run_single.assert_not_called()
        sweep.assert_not_called()

    def test_records_entry_exception_without_raising(self) -> None:
        outcome, _harness = self._run_harness(
            clone_error=ValueError("Git clone failed")
        )

        self.assertFalse(outcome.loop_result.succeeded)
        self.assertEqual(outcome.error, "Git clone failed")
        self.assertEqual(outcome.status, "errored")
        self.assertEqual(outcome.failure_class, "infrastructure")
        self.assertIsNone(outcome.pr_result)

    def test_error_report_uses_errored_status(self) -> None:
        outcome, _harness = self._run_harness(
            clone_error=ValueError("Git clone failed")
        )
        payload = json.loads(Path(outcome.report_path).read_text())

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["status"], "errored")
        self.assertEqual(payload["failure_class"], "infrastructure")
        self.assertEqual(payload["error"], "Git clone failed")
        self.assertEqual(payload["commit"], "a" * 40)
        self.assertIsNone(payload["baseline"])
        self.assertIsNone(payload["environment"])
        self.assertEqual(
            payload["test_patch"], {"applied": False, "file_paths": []}
        )

    def test_run_benchmark_writes_summary_and_sweeps_stale_workspace(
        self,
    ) -> None:
        root = self._temp_directory()
        workspace = root / "workspace"
        stale_environment = workspace / ".environments" / "acme__widget"
        stale_environment.mkdir(parents=True)
        stale_clone = workspace / "acme__widget"
        stale_clone.mkdir()
        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "acme/widget": {
                "issue_number": 42,
                "commit": "a" * 40,
                "test_command": ["pytest"],
            }
        }))

        with patch(
            "swe_agent.benchmark.runner.clone_repository",
            side_effect=ValueError("Git clone failed"),
        ):
            outcomes = run_benchmark(
                str(config_path),
                Mock(),
                workspace_dir=str(workspace),
                report_dir=str(root / "runs"),
            )
        summaries = list((root / "runs").glob("*_benchmark_summary.json"))
        payload = json.loads(summaries[0].read_text())

        self.assertFalse(stale_environment.exists())
        self.assertFalse(stale_clone.exists())
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].status, "errored")
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(
            payload["counts"],
            {
                "total": 1,
                "succeeded": 0,
                "failed": 0,
                "test_timeout": 0,
                "context_too_large": 0,
                "could_not_install": 0,
                "provider_unavailable": 0,
                "errored": 1,
            },
        )
        self.assertEqual(payload["reports"], [outcomes[0].report_path])

    def test_resume_skips_only_matching_terminal_records_unless_retried(self) -> None:
        root = self._temp_directory()
        record = {
            "repository": "acme/widget", "issue": 42, "commit": "a" * 40,
            "provider": "tokenharbor", "model": "deepseek-test",
            "base_url_host": "tokenharbor.ai", "status": "provider_unavailable",
        }
        (root / "entry.json").write_text(json.dumps(record))
        provider = SimpleNamespace(
            provider="tokenharbor", model="deepseek-test",
            base_url_host="tokenharbor.ai",
        )

        skipped = _resumable_results(
            str(root), [self.entry], provider, retry_errored=False,
        )
        retried = _resumable_results(
            str(root), [self.entry], provider, retry_errored=True,
        )

        self.assertIn(("acme/widget", 42, "a" * 40), skipped)
        self.assertEqual(retried, {})

    def test_preflight_auth_failure_stops_before_entry_is_scheduled(self) -> None:
        root = self._temp_directory()
        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "acme/widget": {
                "issue_number": 42, "commit": "a" * 40,
                "test_command": ["pytest"],
            },
        }))
        provider = SimpleNamespace(provider="tokenharbor", model="model")
        provider.preflight = Mock(
            side_effect=ProviderAuthError(
                "Token Harbor preflight failed: authentication rejected (HTTP 401)"
            )
        )
        with patch("swe_agent.benchmark.runner.coerce_provider", return_value=provider), patch(
            "swe_agent.benchmark.runner.run_single_entry"
        ) as run_single:
            with self.assertRaises(ProviderAuthError):
                run_benchmark(str(config_path), Mock(), report_dir=str(root / "runs"))

        run_single.assert_not_called()
        # A failed preflight leaves no per-entry record behind.
        self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()

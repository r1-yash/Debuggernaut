import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from swe_agent.agent.executor import apply_and_test, apply_fix, run_tests
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.guardrails import RejectedProposalError


class ExecutorTests(unittest.TestCase):
    def _fix_result(self) -> FixResult:
        return FixResult(
            file_path="parser.py",
            original_content="return None\n",
            new_content="return {}\n",
            reasoning="Return an empty mapping.",
        )

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_successful_fix_stays_applied(self, run: Mock) -> None:
        run.return_value = Mock(returncode=0, stdout="passed", stderr="")
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            target = repository / "parser.py"
            target.write_text("return None\n")

            outcome = apply_and_test(
                str(repository), self._fix_result(), ["pytest"]
            )

            self.assertTrue(outcome.passed)
            self.assertEqual(target.read_text(), "return {}\n")
            run.assert_called_once_with(
                ["pytest"],
                cwd=str(repository),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_failed_fix_is_reverted(self, run: Mock) -> None:
        run.return_value = Mock(returncode=1, stdout="", stderr="failed")
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            target = repository / "parser.py"
            target.write_text("return None\n")

            outcome = apply_and_test(
                str(repository), self._fix_result(), ["pytest"]
            )

            self.assertFalse(outcome.passed)
            self.assertEqual(target.read_text(), "return None\n")

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_timeout_returns_failed_outcome_without_raising(self, run: Mock) -> None:
        run.side_effect = subprocess.TimeoutExpired(["pytest"], 120)
        with tempfile.TemporaryDirectory() as temporary_dir:
            outcome = run_tests(temporary_dir, ["pytest"])

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.return_code, -1)
        self.assertIn("timed out", outcome.stderr)

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_refuses_to_apply_test_file(self, run: Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "tests").mkdir()
            target = repository / "tests" / "test_calc.py"
            target.write_text("def test_add():\n    assert True\n")
            fix = FixResult(
                file_path="tests/test_calc.py",
                original_content="def test_add():\n    assert True\n",
                new_content="def test_add():\n    assert False\n",
                reasoning="Break the test.",
            )

            with self.assertRaises(RejectedProposalError) as context:
                apply_and_test(str(repository), fix, ["pytest"])

            self.assertEqual(context.exception.kind, "test_file")
            self.assertEqual(
                target.read_text(), "def test_add():\n    assert True\n"
            )
            run.assert_not_called()

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_refuses_to_create_missing_file(self, run: Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            fix = FixResult(
                file_path="missing.py",
                original_content="",
                new_content="created = True\n",
                reasoning="Create a new file.",
            )

            with self.assertRaises(RejectedProposalError) as context:
                apply_fix(str(repository), fix)

            self.assertEqual(context.exception.kind, "nonexistent_path")
            self.assertFalse((repository / "missing.py").exists())
            run.assert_not_called()

    @patch("swe_agent.agent.executor.subprocess.run")
    def test_refuses_file_named_by_test_command(self, run: Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            target = repository / "check_math.py"
            target.write_text("assert True\n")
            fix = FixResult(
                file_path="check_math.py",
                original_content="assert True\n",
                new_content="assert False\n",
                reasoning="Break the check.",
            )

            with self.assertRaises(RejectedProposalError) as context:
                apply_fix(
                    str(repository), fix, ["python", "-m", "pytest", "check_math.py"]
                )

            self.assertEqual(context.exception.kind, "test_file")
            self.assertEqual(target.read_text(), "assert True\n")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

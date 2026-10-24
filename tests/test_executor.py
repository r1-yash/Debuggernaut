import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from swe_agent.agent.executor import apply_and_test, apply_fix, run_tests
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.guardrails import RejectedProposalError
from swe_agent.process import CommandResult


class ExecutorTests(unittest.TestCase):
    def _fix_result(self) -> FixResult:
        return FixResult(
            file_path="parser.py",
            original_content="return None\n",
            new_content="return {}\n",
            reasoning="Return an empty mapping.",
        )

    def test_successful_fix_stays_applied(self) -> None:
        test_runner = Mock(
            return_value=CommandResult(
                command=["pytest"],
                return_code=0,
                stdout="passed",
            )
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            target = repository / "parser.py"
            target.write_text("return None\n")

            outcome = apply_and_test(
                str(repository),
                self._fix_result(),
                ["pytest"],
                test_runner=test_runner,
            )

            self.assertTrue(outcome.passed)
            self.assertEqual(target.read_text(), "return {}\n")
            test_runner.assert_called_once_with(["pytest"])

    def test_failed_fix_is_reverted(self) -> None:
        test_runner = Mock(
            return_value=CommandResult(
                command=["pytest"],
                return_code=1,
                stderr="failed",
            )
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            target = repository / "parser.py"
            target.write_text("return None\n")

            outcome = apply_and_test(
                str(repository),
                self._fix_result(),
                ["pytest"],
                test_runner=test_runner,
            )

            self.assertFalse(outcome.passed)
            self.assertEqual(target.read_text(), "return None\n")

    def test_timeout_returns_failed_outcome_without_raising(self) -> None:
        test_runner = Mock(
            return_value=CommandResult(
                command=["pytest"],
                return_code=-1,
                timed_out=True,
            )
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            outcome = run_tests(
                temporary_dir,
                ["pytest"],
                timeout_seconds=300.0,
                test_runner=test_runner,
            )

        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.return_code, -1)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.timeout_seconds, 300.0)

    def test_process_that_cannot_start_is_not_started(self) -> None:
        command_runner = Mock()
        command_runner.run.side_effect = FileNotFoundError("pytest")
        with tempfile.TemporaryDirectory() as temporary_dir:
            outcome = run_tests(
                temporary_dir,
                ["pytest"],
                command_runner=command_runner,
            )

        self.assertFalse(outcome.passed)
        self.assertIsNone(outcome.return_code)
        self.assertFalse(outcome.started)
        self.assertIn("FileNotFoundError", outcome.stderr)

    def test_refuses_to_apply_test_file(self) -> None:
        test_runner = Mock()
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
                apply_and_test(
                    str(repository),
                    fix,
                    ["pytest"],
                    test_runner=test_runner,
                )

            self.assertEqual(context.exception.kind, "test_file")
            self.assertEqual(
                target.read_text(), "def test_add():\n    assert True\n"
            )
            test_runner.assert_not_called()

    def test_refuses_to_create_missing_file(self) -> None:
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

    def test_refuses_file_named_by_test_command(self) -> None:
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
                    str(repository),
                    fix,
                    ["python", "-m", "pytest", "check_math.py"],
                )

            self.assertEqual(context.exception.kind, "test_file")
            self.assertEqual(target.read_text(), "assert True\n")

    def test_refuses_explicit_test_patch_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            fixtures = repository / "fixtures"
            fixtures.mkdir()
            target = fixtures / "helper.py"
            target.write_text("VALUE = 1\n")
            fix = FixResult(
                file_path="fixtures/helper.py",
                original_content="VALUE = 1\n",
                new_content="VALUE = 2\n",
                reasoning="Change the fixture.",
            )

            with self.assertRaises(RejectedProposalError) as context:
                apply_fix(
                    str(repository),
                    fix,
                    ["pytest"],
                    protected_paths=["fixtures/helper.py"],
                )

            self.assertEqual(context.exception.kind, "test_patch_file")
            self.assertEqual(target.read_text(), "VALUE = 1\n")


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock, patch

from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.agent.loop import resolve_issue


class LoopTests(unittest.TestCase):
    def _fix_result(self, number: int) -> FixResult:
        return FixResult(
            file_path=f"module_{number}.py",
            original_content="original\n",
            new_content="fixed\n",
            reasoning=f"Fix approach {number}.",
        )

    def _test_outcome(self, passed: bool, number: int) -> TestOutcome:
        return TestOutcome(
            passed=passed,
            return_code=0 if passed else 1,
            stdout=f"stdout {number}",
            stderr="" if passed else f"stderr {number}",
        )

    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_succeeds_on_first_attempt(
        self,
        propose: Mock,
        apply_and_test: Mock,
    ) -> None:
        fix_result = self._fix_result(1)
        propose.return_value = fix_result
        apply_and_test.return_value = self._test_outcome(True, 1)
        client = Mock()

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], client
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_attempt.fix_result, fix_result)
        self.assertEqual(len(result.attempts), 1)
        propose.assert_called_once_with(
            "repository", "Broken parser", "Details", client
        )
        apply_and_test.assert_called_once_with("repository", fix_result, ["pytest"])

    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_succeeds_after_a_failed_attempt(
        self,
        propose: Mock,
        apply_and_test: Mock,
    ) -> None:
        first_fix = self._fix_result(1)
        second_fix = self._fix_result(2)
        propose.side_effect = [first_fix, second_fix]
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(True, 2),
        ]

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_attempt.fix_result, second_fix)
        self.assertEqual(len(result.attempts), 2)
        retry_body = propose.call_args_list[1].args[2]
        self.assertIn("do not repeat the same approach", retry_body)
        self.assertIn(first_fix.file_path, retry_body)
        self.assertIn(first_fix.reasoning, retry_body)
        self.assertIn("stdout 1", retry_body)
        self.assertIn("stderr 1", retry_body)

    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_exhausts_max_attempts_without_success(
        self,
        propose: Mock,
        apply_and_test: Mock,
    ) -> None:
        fixes = [self._fix_result(1), self._fix_result(2)]
        propose.side_effect = fixes
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(False, 2),
        ]

        result = resolve_issue(
            "repository",
            "Broken parser",
            "Details",
            ["pytest"],
            Mock(),
            max_attempts=2,
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.final_attempt.fix_result, fixes[-1])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(propose.call_count, 2)
        self.assertEqual(apply_and_test.call_count, 2)

    def test_rejects_max_attempts_less_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resolve_issue(
                "repository",
                "Broken parser",
                "Details",
                ["pytest"],
                Mock(),
                max_attempts=0,
            )


if __name__ == "__main__":
    unittest.main()

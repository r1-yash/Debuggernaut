import unittest
from unittest.mock import Mock, patch

from swe_agent.agent.diagnose import Diagnosis
from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import ExplorationStep, FixResult
from swe_agent.agent.guardrails import RejectedProposalError
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

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_succeeds_on_first_attempt(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
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
            "repository", "Broken parser", "Details", client,
            test_command=["pytest"], protected_paths=[], exploration_steps=[],
        )
        apply_and_test.assert_called_once_with(
            "repository", fix_result, ["pytest"], protected_paths=[]
        )
        diagnose.assert_not_called()

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_succeeds_after_wrong_fix_diagnosis(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        first_fix = self._fix_result(1)
        second_fix = self._fix_result(2)
        propose.side_effect = [first_fix, second_fix]
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(True, 2),
        ]
        diagnose.return_value = Diagnosis(reason="wrong_fix", explanation="Bad logic.")

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.final_attempt.fix_result, second_fix)
        self.assertEqual(len(result.attempts), 2)
        diagnose.assert_called_once()
        retry_body = propose.call_args_list[1].args[2]
        self.assertIn("do not repeat the same approach", retry_body)
        self.assertIn(first_fix.file_path, retry_body)
        self.assertNotIn("Do not target these files again", retry_body)

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_wrong_file_diagnosis_avoids_target_on_next_proposal(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        first_fix = self._fix_result(1)
        second_fix = self._fix_result(2)
        propose.side_effect = [first_fix, second_fix]
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(True, 2),
        ]
        diagnose.return_value = Diagnosis(
            reason="wrong_file", explanation="The bug is not in module_1.py."
        )

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertTrue(result.succeeded)
        retry_body = propose.call_args_list[1].args[2]
        self.assertIn("Do not target these files again", retry_body)
        self.assertIn(first_fix.file_path, retry_body)

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_exhausts_max_attempts_without_success(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        fixes = [self._fix_result(1), self._fix_result(2)]
        propose.side_effect = fixes
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(False, 2),
        ]
        diagnose.return_value = Diagnosis(reason="wrong_fix", explanation="Still wrong.")

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock(), max_attempts=2
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.final_attempt.fix_result, fixes[-1])
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(propose.call_count, 2)
        self.assertEqual(apply_and_test.call_count, 2)
        diagnose.assert_called_once()

    def test_rejects_max_attempts_less_than_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            resolve_issue(
                "repository", "Broken parser", "Details", ["pytest"], Mock(), max_attempts=0
            )

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_records_diagnosis_and_avoid_list_on_attempt(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        propose.side_effect = [self._fix_result(1), self._fix_result(2)]
        apply_and_test.side_effect = [
            self._test_outcome(False, 1),
            self._test_outcome(True, 2),
        ]
        diagnose.return_value = Diagnosis(
            reason="wrong_file", explanation="The bug is not in module_1.py."
        )

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        first_attempt = result.attempts[0]
        self.assertEqual(first_attempt.diagnosis.reason, "wrong_file")
        self.assertEqual(
            first_attempt.avoid_file_paths_after, ["module_1.py"]
        )
        self.assertIsNone(result.attempts[1].diagnosis)
        self.assertEqual(result.attempts[1].avoid_file_paths_after, ["module_1.py"])

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_records_exploration_steps_on_attempt(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        fix_result = self._fix_result(1)

        def propose_with_steps(
            repo_path: str,
            issue_title: str,
            issue_body: str,
            client: object,
            *,
            test_command: list[str] | None = None,
            protected_paths: list[str] | None = None,
            exploration_steps: list[ExplorationStep],
        ) -> FixResult:
            exploration_steps.append(
                ExplorationStep(
                    tool="read_file",
                    arguments={"path": "module_1.py"},
                    ok=True,
                    result_size=9,
                    result_unit="characters",
                )
            )
            return fix_result

        propose.side_effect = propose_with_steps
        apply_and_test.return_value = self._test_outcome(True, 1)

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertEqual(len(result.attempts[0].exploration_steps), 1)
        self.assertEqual(
            result.attempts[0].exploration_steps[0].tool, "read_file"
        )
        self.assertEqual(
            result.attempts[0].exploration_steps[0].arguments,
            {"path": "module_1.py"},
        )

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_rejected_test_file_does_not_consume_attempt(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        fix_result = self._fix_result(1)
        propose.side_effect = [
            RejectedProposalError(
                "tests/test_parser.py",
                "test_file",
                "tests/test_parser.py is a test file.",
            ),
            fix_result,
        ]
        apply_and_test.return_value = self._test_outcome(True, 1)

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock(),
            max_attempts=1,
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(len(result.guard_events), 1)
        event = result.guard_events[0]
        self.assertEqual(event.attempt, 1)
        self.assertEqual(event.file_path, "tests/test_parser.py")
        self.assertEqual(event.kind, "test_file")
        self.assertIn("is a test file", event.reason)
        self.assertEqual(apply_and_test.call_count, 1)
        retry_body = propose.call_args_list[1].args[2]
        self.assertIn("rejected", retry_body)
        self.assertIn("tests/test_parser.py", retry_body)

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_rejected_nonexistent_path_is_retried(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        fix_result = self._fix_result(1)
        propose.side_effect = [
            RejectedProposalError(
                "missing.py", "nonexistent_path", "File does not exist."
            ),
            fix_result,
        ]
        apply_and_test.return_value = self._test_outcome(True, 1)

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.guard_events[0].kind, "nonexistent_path")
        self.assertEqual(result.guard_events[0].file_path, "missing.py")
        retry_body = propose.call_args_list[1].args[2]
        self.assertIn("does not exist", retry_body)

    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_guard_rejection_cap_ends_run_failed(
        self, propose: Mock, apply_and_test: Mock
    ) -> None:
        propose.side_effect = [
            RejectedProposalError(
                f"tests/test_{number}.py", "test_file", "Test file."
            )
            for number in range(3)
        ]

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock()
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.attempts, [])
        self.assertEqual(len(result.guard_events), 3)
        self.assertEqual(propose.call_count, 3)
        apply_and_test.assert_not_called()

    @patch("swe_agent.agent.loop.diagnose_failure")
    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_forwards_protected_paths_to_proposal_and_application(
        self, propose: Mock, apply_and_test: Mock, diagnose: Mock
    ) -> None:
        fix_result = self._fix_result(1)
        propose.return_value = fix_result
        apply_and_test.return_value = self._test_outcome(True, 1)

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock(),
            protected_paths=["fixtures/helper.py"],
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(
            propose.call_args.kwargs["protected_paths"],
            ["fixtures/helper.py"],
        )
        self.assertEqual(
            apply_and_test.call_args.kwargs["protected_paths"],
            ["fixtures/helper.py"],
        )

    @patch("swe_agent.agent.loop.apply_and_test")
    @patch("swe_agent.agent.loop.propose_fix")
    def test_records_test_patch_guard_event(
        self, propose: Mock, apply_and_test: Mock
    ) -> None:
        propose.side_effect = RejectedProposalError(
            "fixtures/helper.py",
            "test_patch_file",
            "fixtures/helper.py is part of the benchmark test patch.",
        )

        result = resolve_issue(
            "repository", "Broken parser", "Details", ["pytest"], Mock(),
            protected_paths=["fixtures/helper.py"],
        )

        self.assertFalse(result.succeeded)
        self.assertEqual(result.guard_events[0].kind, "test_patch_file")
        self.assertEqual(result.guard_events[0].file_path, "fixtures/helper.py")
        apply_and_test.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_agent.agent.diagnose import (
    Diagnosis,
    DiagnosisUnavailable,
    diagnose_failure,
)
from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult


class DiagnoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fix_result = FixResult(
            file_path="parser.py",
            original_content="old\n",
            new_content="new\n",
            reasoning="Correct parser behavior.",
        )
        self.outcome = TestOutcome(
            passed=False, return_code=1, stdout="failed", stderr="traceback"
        )

    def test_classifies_wrong_file(self) -> None:
        client = Mock()
        client.models.generate_content.return_value.parsed = {
            "reason": "wrong_file",
            "explanation": "The failing behavior is implemented elsewhere.",
        }

        diagnosis = diagnose_failure(self.fix_result, self.outcome, client)

        self.assertEqual(diagnosis.reason, "wrong_file")
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_schema, Diagnosis)

    def test_classifies_wrong_fix(self) -> None:
        client = Mock()
        client.models.generate_content.return_value.parsed = {
            "reason": "wrong_fix",
            "explanation": "The selected file is relevant but the logic is incomplete.",
        }

        diagnosis = diagnose_failure(self.fix_result, self.outcome, client)

        self.assertEqual(diagnosis.reason, "wrong_fix")

    def test_repairs_one_invalid_reply(self) -> None:
        provider = Mock()
        provider.structured.side_effect = [
            SimpleNamespace(parsed={"reason": "maybe"}),
            SimpleNamespace(
                parsed={
                    "reason": "wrong_fix",
                    "explanation": "The logic is incomplete.",
                }
            ),
        ]

        with patch(
            "swe_agent.agent.diagnose.coerce_provider", return_value=provider
        ):
            diagnosis = diagnose_failure(
                self.fix_result, self.outcome, object()
            )

        self.assertEqual(diagnosis.reason, "wrong_fix")
        self.assertEqual(provider.structured.call_count, 2)
        repair_prompt = provider.structured.call_args_list[1].args[1]
        self.assertIn("json", repair_prompt.lower())
        self.assertIn("could not be validated", repair_prompt)

    def test_degrades_to_unknown_after_one_failed_repair(self) -> None:
        provider = Mock()
        provider.structured.side_effect = [
            SimpleNamespace(parsed={"reason": "maybe"}),
            SimpleNamespace(parsed={"reason": "still-not-valid"}),
        ]

        with patch(
            "swe_agent.agent.diagnose.coerce_provider", return_value=provider
        ):
            with self.assertRaises(DiagnosisUnavailable) as raised:
                diagnose_failure(self.fix_result, self.outcome, object())

        self.assertEqual(provider.structured.call_count, 2)
        self.assertEqual(raised.exception.diagnosis.reason, "unknown")
        self.assertIn("one repair attempt", str(raised.exception))

    def test_raises_for_unparseable_response(self) -> None:
        client = Mock()
        client.models.generate_content.return_value.parsed = None
        client.models.generate_content.return_value.text = None

        with self.assertRaisesRegex(ValueError, "parseable structured response"):
            diagnose_failure(self.fix_result, self.outcome, client)


if __name__ == "__main__":
    unittest.main()

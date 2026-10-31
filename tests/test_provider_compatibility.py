"""Deterministic tests for provider feature rejection and JSON fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import BaseModel, ValidationError

from swe_agent.agent.diagnose import diagnose_failure
from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult, propose_fix
from swe_agent.agent.structured import parse_structured_response
from swe_agent.llm import (
    ProviderCompatibilityError,
    ProviderFeatureUnavailable,
    ProviderInvalidRequest,
    ProviderUnavailable,
    TokenHarborProvider,
    UsageTracker,
    usage_run,
)


class _Sample(BaseModel):
    value: int


def _count(tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=tokens, method="compatibility-test", exact=True, reason=None
    )


def _reply(content: str | None = None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, tool_calls=tool_calls
                )
            )
        ]
    )


def _provider_error(
    message: str, *, status: int = 400, body: dict | None = None
) -> Exception:
    class ProviderError(Exception):
        pass

    error = ProviderError(message)
    error.status_code = status
    error.body = body if body is not None else {
        "error": {"message": message}
    }
    return error


class ProviderCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.openai_patch = patch("openai.OpenAI")
        openai_mock = self.openai_patch.start()
        self.addCleanup(self.openai_patch.stop)
        self.count_patch = patch(
            "swe_agent.llm.count_input_tokens", return_value=_count()
        )
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)
        self.client = openai_mock.return_value

    def _provider(self, **overrides) -> TokenHarborProvider:
        options = {
            "max_input_tokens": 1000,
            "max_retries": 0,
            "sleep": lambda _: self.fail("compatibility tests must not sleep"),
            "random_value": lambda: 0.0,
        }
        options.update(overrides)
        return TokenHarborProvider(
            "deepseek-test",
            "configured-secret",
            "https://tokenharbor.ai/v1",
            **options,
        )

    def _calls(self):
        return self.client.chat.completions.create.call_args_list

    def test_normal_json_mode_succeeds_without_fallback(self) -> None:
        self.client.chat.completions.create.return_value = _reply(
            '{"value": 3}'
        )
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker):
            response = provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        self.assertEqual(parse_structured_response(response, _Sample).value, 3)
        self.assertEqual(len(self._calls()), 1)
        request = self._calls()[0].kwargs
        self.assertEqual(
            request["response_format"], {"type": "json_object"}
        )
        self.assertNotIn("tools", request)
        call = tracker.calls[0]
        self.assertTrue(call.success)
        self.assertIsNone(call.compatibility)
        self.assertEqual(
            call.request_config["response_format"], {"type": "json_object"}
        )
        self.assertTrue(provider.capabilities.json_mode)
        self.assertTrue(provider.capabilities.structured_output)
        self.assertTrue(provider.capabilities.text)

    def test_explicit_json_mode_rejection_uses_text_fallback(self) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format json_object is not supported for this model"
            ),
            _reply('{"value": 7}'),
        ]
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker):
            response = provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        self.assertEqual(parse_structured_response(response, _Sample).value, 7)
        first, fallback = self._calls()
        self.assertEqual(
            first.kwargs["response_format"], {"type": "json_object"}
        )
        self.assertNotIn("response_format", fallback.kwargs)
        fallback_messages = fallback.kwargs["messages"]
        self.assertEqual(
            [message["role"] for message in fallback_messages],
            ["system", "user", "user"],
        )
        self.assertIn("JSON", fallback_messages[-1]["content"])
        self.assertIn('"value"', fallback_messages[-1]["content"])

        original_call, fallback_call = tracker.calls
        self.assertFalse(original_call.success)
        self.assertEqual(original_call.compatibility["decision"], "fallback")
        self.assertEqual(
            original_call.compatibility["fallback"],
            "text_mode_with_json_instruction",
        )
        self.assertEqual(original_call.compatibility["fallback_result"], "succeeded")
        self.assertEqual(
            fallback_call.compatibility["original_call_number"],
            original_call.call_number,
        )
        self.assertEqual(
            fallback_call.compatibility["fallback_result"], "succeeded"
        )
        self.assertEqual(fallback_call.validation["outcome"], "valid")
        self.assertEqual(fallback_call.validation["schema"], "_Sample")
        self.assertNotIn(
            "response_format", fallback_call.request_config
        )
        self.assertFalse(provider.capabilities.json_mode)
        self.assertFalse(provider.capabilities.structured_output)
        self.assertTrue(provider.capabilities.text)

    def test_generic_http_400_does_not_trigger_fallback(self) -> None:
        self.client.chat.completions.create.side_effect = _provider_error(
            "Bad request: the payload contains an invalid field setting"
        )
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(
            ProviderInvalidRequest
        ) as raised:
            provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        self.assertNotIsInstance(raised.exception, ProviderUnavailable)
        self.assertEqual(len(self._calls()), 1)
        self.assertIsNone(tracker.calls[0].compatibility)
        self.assertIsNone(provider.capabilities.json_mode)

    def test_unsupported_tool_calling_is_not_silently_removed(self) -> None:
        def read_file(path: str) -> str:
            """Read a repository file."""
            return path

        self.client.chat.completions.create.side_effect = _provider_error(
            "tools are not supported for this model"
        )
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(
            ProviderFeatureUnavailable
        ) as raised:
            provider.explore("system", "explore", [read_file])

        self.assertEqual(raised.exception.feature, "tool_calling")
        self.assertEqual(len(self._calls()), 1)
        request = self._calls()[0].kwargs
        self.assertEqual(
            request["tools"][0]["function"]["name"], "read_file"
        )
        self.assertNotIn("response_format", request)
        self.assertFalse(provider.capabilities.tool_calling)
        self.assertIsNone(provider.capabilities.json_mode)

    def test_fallback_invalid_json_is_rejected_by_validation(self) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format is unsupported for this model"
            ),
            _reply("this is not valid JSON"),
        ]
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker):
            response = provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        with self.assertRaises(ValueError):
            parse_structured_response(response, _Sample)
        fallback_call = tracker.calls[-1]
        self.assertEqual(fallback_call.validation["outcome"], "invalid")
        self.assertEqual(fallback_call.validation["schema"], "_Sample")
        self.assertIn("JSON", fallback_call.validation["error"])

    def test_fallback_valid_json_that_fails_schema_validation_is_rejected(
        self,
    ) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "json mode is not supported for this model"
            ),
            _reply('{"unexpected": "shape"}'),
        ]
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker):
            response = provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        with self.assertRaises(ValidationError):
            parse_structured_response(response, _Sample)
        fallback_call = tracker.calls[-1]
        self.assertEqual(fallback_call.validation["outcome"], "invalid")
        self.assertIn("value", fallback_call.validation["error"])

    def test_fallback_request_rejection_preserves_both_errors(self) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format json_object is not supported for this model",
                body={
                    "error": {
                        "message": (
                            "response_format json_object is not supported "
                            "for this model"
                        )
                    }
                },
            ),
            _provider_error(
                "fallback text request was rejected for a separate reason",
                body={
                    "error": {
                        "message": (
                            "fallback text request was rejected for a "
                            "separate reason"
                        )
                    }
                },
            ),
        ]
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(
            ProviderCompatibilityError
        ) as raised:
            provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        error = raised.exception
        self.assertEqual(error.status_code, 400)
        self.assertIsInstance(error.original_error, ProviderFeatureUnavailable)
        self.assertIsInstance(error.fallback_error, ProviderInvalidRequest)
        self.assertIn("fallback also failed", str(error))

        original_call, fallback_call = tracker.calls
        self.assertEqual(
            original_call.compatibility["fallback_result"], "failed"
        )
        self.assertEqual(
            fallback_call.compatibility["original_call_number"],
            original_call.call_number,
        )
        self.assertEqual(
            fallback_call.compatibility["fallback_result"], "failed"
        )
        self.assertIn(
            "not supported",
            json.dumps(original_call.error["provider_body"]),
        )
        self.assertIn(
            "separate reason",
            json.dumps(fallback_call.error["provider_body"]),
        )
        self.assertNotEqual(
            original_call.error["provider_body"],
            fallback_call.error["provider_body"],
        )

    def test_credentials_remain_redacted_in_compatibility_diagnostics(
        self,
    ) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format json_object is not supported; "
                "Authorization: Bearer original-token; api_key=original-key",
                body={
                    "error": {
                        "message": (
                            "response_format json_object is not supported; "
                            "Authorization: Bearer body-token"
                        )
                    },
                    "authorization": "Bearer nested-token",
                    "api_key": "nested-key",
                },
            ),
            _provider_error(
                "fallback rejected; Cookie: session=fallback-cookie; "
                "access_token=fallback-token"
            ),
        ]
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(
            ProviderCompatibilityError
        ) as raised:
            provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        serialized = json.dumps(
            {
                "error": str(raised.exception),
                "calls": [
                    call.model_dump(mode="json") for call in tracker.calls
                ],
            }
        )
        for secret in (
            "configured-secret",
            "original-token",
            "original-key",
            "body-token",
            "nested-token",
            "nested-key",
            "fallback-cookie",
            "fallback-token",
        ):
            self.assertNotIn(secret, serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_known_json_rejection_is_not_repeated_and_retries_stay_bounded(
        self,
    ) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format json_object is not supported for this model"
            ),
            _provider_error("fallback rejected once"),
            _reply('{"value": 11}'),
        ]
        provider = self._provider(max_retries=2)
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(
            ProviderCompatibilityError
        ):
            provider.structured(
                "Return JSON.", "Return a value.", _Sample, "diagnosis"
            )

        with usage_run(tracker):
            response = provider.structured(
                "Return JSON.", "Return another value.", _Sample, "diagnosis"
            )

        self.assertEqual(parse_structured_response(response, _Sample).value, 11)
        self.assertEqual(len(self._calls()), 3)
        self.assertEqual(
            self._calls()[0].kwargs["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("response_format", self._calls()[1].kwargs)
        self.assertNotIn("response_format", self._calls()[2].kwargs)
        self.assertEqual(
            tracker.calls[2].compatibility["decision"],
            "skip_known_unsupported",
        )
        self.assertIn(
            "was not sent again", tracker.calls[2].compatibility["reason"]
        )

    def test_diagnosis_uses_validated_text_fallback(self) -> None:
        self.client.chat.completions.create.side_effect = [
            _provider_error(
                "response_format json_object is not supported for this model"
            ),
            _reply(
                '{"reason": "wrong_fix", "explanation": '
                '"The condition is inverted."}'
            ),
        ]
        provider = self._provider()
        fix_result = FixResult(
            file_path="parser.py",
            original_content="old\n",
            new_content="new\n",
            reasoning="Correct parser behavior.",
        )
        outcome = TestOutcome(
            passed=False,
            return_code=1,
            stdout="failed",
            stderr="traceback",
        )
        tracker = UsageTracker()

        with usage_run(tracker):
            diagnosis = diagnose_failure(fix_result, outcome, provider)

        self.assertEqual(diagnosis.reason, "wrong_fix")
        self.assertEqual(len(self._calls()), 2)
        self.assertEqual(tracker.calls[-1].validation["outcome"], "valid")

    def test_fix_generation_uses_validated_text_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            repository = Path(temporary_dir)
            (repository / "parser.py").write_text(
                "def parse():\n    return None\n"
            )
            provider = self._provider()
            provider.explore = Mock(
                return_value=(
                    [{"role": "user", "content": "Investigate the issue."}],
                    SimpleNamespace(),
                    1,
                )
            )
            self.client.chat.completions.create.side_effect = [
                _provider_error(
                    "response_format json_object is not supported for this model"
                ),
                _reply(
                    json.dumps(
                        {
                            "file_path": "parser.py",
                            "new_content": "def parse():\n    return {}\n",
                            "reasoning": "Return an empty mapping instead.",
                        }
                    )
                ),
            ]
            tracker = UsageTracker()

            with usage_run(tracker):
                result = propose_fix(
                    str(repository),
                    "Parser returns None",
                    "Return an empty mapping.",
                    provider,
                )

        self.assertEqual(result.file_path, "parser.py")
        self.assertEqual(result.new_content, "def parse():\n    return {}\n")
        self.assertEqual(len(result.proposal_attempts), 1)
        self.assertTrue(result.proposal_attempts[0].valid)
        self.assertEqual(tracker.calls[-1].validation["outcome"], "valid")


if __name__ == "__main__":
    unittest.main()

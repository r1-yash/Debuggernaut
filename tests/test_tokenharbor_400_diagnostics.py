"""Focused diagnostics for the original TokenHarbor diagnosis HTTP 400.

The live provider is never contacted.  One test uses the official DeepSeek
V4.1 converter that already ships as a runtime dependency, and the others mock
the OpenAI-compatible client to verify the exact outgoing request shapes and
the safe error details recorded by the production adapter.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swe_agent.llm import (
    DEFAULT_TOKENHARBOR_MODEL,
    ProviderCompatibilityError,
    TokenHarborProvider,
    UsageTracker,
    usage_run,
)

try:
    from deepseek_recipe import (
        ChatCompletionRequest,
        ConversionError,
        ConversionOptions,
        ModelVersion,
    )
except ImportError:  # pragma: no cover - dependency is declared in pyproject.
    ChatCompletionRequest = None
    ConversionError = None
    ConversionOptions = None
    ModelVersion = None


_ORIGINAL_DIAGNOSIS_SYSTEM = (
    "You diagnose failed software-maintenance fixes. Explain concisely."
)
_ORIGINAL_DIAGNOSIS_PROMPT = (
    "Proposed file path: parser.py\n\n"
    "Test stdout:\n1 failed\n\n"
    "Classify the failure as wrong_file or wrong_fix."
)
_CURRENT_DIAGNOSIS_SYSTEM = (
    "You diagnose failed software-maintenance fixes. "
    "Return only a JSON object with reason and explanation keys."
)
_CURRENT_DIAGNOSIS_PROMPT = (
    "Proposed file path: parser.py\n\n"
    "Test stdout:\n1 failed\n\n"
    "Return the diagnosis as a JSON object."
)
_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Report that the provider is reachable.",
        "parameters": {"type": "object", "properties": {}},
    },
}
_DEEPSEEK_JSON_ERROR = (
    "'messages' must contain the word 'json' in some form, to use "
    "'response_format' of type 'json_object'."
)


def _count(tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=tokens, method="diagnostic-test", exact=True, reason=None
    )


class OfficialDeepSeekConverterTests(unittest.TestCase):
    """The official converter reproduces the provider-side json rule locally."""

    @unittest.skipIf(ChatCompletionRequest is None, "deepseek-recipe is missing")
    def _convert(self, body: dict) -> None:
        options = ConversionOptions().with_model_alias(
            DEFAULT_TOKENHARBOR_MODEL, ModelVersion.DeepseekV41Flash
        )
        request = ChatCompletionRequest(body)
        request.convert(options)

    @unittest.skipIf(ChatCompletionRequest is None, "deepseek-recipe is missing")
    def test_json_mode_is_rejected_only_when_prompt_lacks_json(self) -> None:
        body = {
            "model": DEFAULT_TOKENHARBOR_MODEL,
            "messages": [
                {"role": "system", "content": _ORIGINAL_DIAGNOSIS_SYSTEM},
                {"role": "user", "content": _ORIGINAL_DIAGNOSIS_PROMPT},
            ],
            "response_format": {"type": "json_object"},
        }

        with self.assertRaises(ConversionError) as raised:
            self._convert(body)
        self.assertIn("must contain the word 'json'", str(raised.exception))

        body["messages"] = [
            {"role": "system", "content": _CURRENT_DIAGNOSIS_SYSTEM},
            {"role": "user", "content": _CURRENT_DIAGNOSIS_PROMPT},
        ]
        self._convert(body)

    @unittest.skipIf(ChatCompletionRequest is None, "deepseek-recipe is missing")
    def test_tools_alone_and_json_plus_tools_with_json_are_accepted(self) -> None:
        messages = [
            {"role": "system", "content": "You are a connectivity check."},
            {"role": "user", "content": "Call the ping tool."},
        ]
        self._convert(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": messages,
                "tools": [_PING_TOOL],
                "tool_choice": "auto",
            }
        )
        self._convert(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {"role": "system", "content": _CURRENT_DIAGNOSIS_SYSTEM},
                    {"role": "user", "content": _CURRENT_DIAGNOSIS_PROMPT},
                ],
                "tools": [_PING_TOOL],
                "tool_choice": "auto",
                "response_format": {"type": "json_object"},
            }
        )

    @unittest.skipIf(ChatCompletionRequest is None, "deepseek-recipe is missing")
    def test_json_plus_tools_without_json_is_rejected_for_the_same_reason(self) -> None:
        body = {
            "model": DEFAULT_TOKENHARBOR_MODEL,
            "messages": [
                {"role": "system", "content": _ORIGINAL_DIAGNOSIS_SYSTEM},
                {"role": "user", "content": _ORIGINAL_DIAGNOSIS_PROMPT},
            ],
            "tools": [_PING_TOOL],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
        }

        with self.assertRaises(ConversionError) as raised:
            self._convert(body)
        self.assertIn("must contain the word 'json'", str(raised.exception))


class ProviderRequestShapeTests(unittest.TestCase):
    """Mocked provider calls verify request shapes and safe 400 diagnostics."""

    def _provider(self, **overrides) -> TokenHarborProvider:
        return TokenHarborProvider(
            DEFAULT_TOKENHARBOR_MODEL,
            "configured-secret",
            "https://tokenharbor.ai/v1",
            max_input_tokens=1000,
            max_retries=0,
            sleep=lambda _: self.fail("diagnostic requests must not sleep"),
            random_value=lambda: 0.0,
            **overrides,
        )

    @patch("swe_agent.llm.count_input_tokens", return_value=_count())
    @patch("openai.OpenAI")
    def test_original_diagnosis_request_falls_back_to_text_mode(
        self, openai_client, _count_input
    ) -> None:
        class BadRequest(Exception):
            status_code = 400
            body = {
                "error": {
                    "message": _DEEPSEEK_JSON_ERROR,
                    "type": "invalid_request_error",
                }
            }

        openai_client.return_value.chat.completions.create.side_effect = (
            BadRequest("Bad Request")
        )
        provider = self._provider()
        tracker = UsageTracker()

        with usage_run(tracker), self.assertRaises(ProviderCompatibilityError) as raised:
            provider.structured(
                _ORIGINAL_DIAGNOSIS_SYSTEM,
                _ORIGINAL_DIAGNOSIS_PROMPT,
                dict,
                "diagnosis",
            )

        requests = (
            openai_client.return_value.chat.completions.create.call_args_list
        )
        self.assertEqual(len(requests), 2)
        original, fallback = requests
        self.assertEqual(
            original.kwargs["response_format"], {"type": "json_object"}
        )
        self.assertNotIn("tools", original.kwargs)
        self.assertEqual(
            [message["role"] for message in original.kwargs["messages"]],
            ["system", "user"],
        )
        self.assertNotIn(
            "json",
            f"{original.kwargs['messages'][0]['content']} "
            f"{original.kwargs['messages'][1]['content']}".lower(),
        )
        self.assertNotIn("response_format", fallback.kwargs)
        self.assertNotIn("tools", fallback.kwargs)
        self.assertIn(
            "JSON", fallback.kwargs["messages"][-1]["content"]
        )

        error = raised.exception
        self.assertEqual(error.status_code, 400)
        self.assertIn("must contain the word 'json'", error.provider_body)
        original_call, fallback_call = tracker.calls
        self.assertEqual(original_call.http_status, 400)
        self.assertFalse(original_call.success)
        self.assertEqual(
            original_call.request_config["response_format"],
            {"type": "json_object"},
        )
        self.assertNotIn("tool_names", original_call.request_config)
        self.assertIn(
            "must contain the word 'json'",
            json.dumps(original_call.error["provider_body"]),
        )
        self.assertEqual(
            original_call.compatibility["fallback_result"], "failed"
        )
        self.assertEqual(
            fallback_call.compatibility["original_call_number"],
            original_call.call_number,
        )
        serialized = json.dumps(
            {
                "error": str(error),
                "calls": [
                    call.model_dump(mode="json") for call in tracker.calls
                ],
            }
        )
        self.assertNotIn("configured-secret", serialized)

    @patch("swe_agent.llm.count_input_tokens", return_value=_count())
    @patch("openai.OpenAI")
    def test_text_tools_and_combined_shapes_stay_independent(
        self, openai_client, _count_input
    ) -> None:
        reply = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ready", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        openai_client.return_value.chat.completions.create.return_value = reply
        provider = self._provider()
        tracker = UsageTracker()
        messages = [
            {"role": "system", "content": "You are a connectivity check."},
            {"role": "user", "content": "Reply with the single word: ready."},
        ]

        with usage_run(tracker):
            provider._call("diagnostic_text", messages=messages)
            provider._call(
                "diagnostic_tools",
                messages=messages,
                tools=[_PING_TOOL],
                tool_choice="auto",
            )
            provider._call(
                "diagnostic_combined",
                messages=[
                    {"role": "system", "content": _CURRENT_DIAGNOSIS_SYSTEM},
                    {"role": "user", "content": _CURRENT_DIAGNOSIS_PROMPT},
                ],
                tools=[_PING_TOOL],
                tool_choice="auto",
                response_format={"type": "json_object"},
            )

        calls = openai_client.return_value.chat.completions.create.call_args_list
        self.assertNotIn("response_format", calls[0].kwargs)
        self.assertNotIn("tools", calls[0].kwargs)
        self.assertNotIn("response_format", calls[1].kwargs)
        self.assertEqual(calls[1].kwargs["tool_choice"], "auto")
        self.assertEqual(
            calls[1].kwargs["tools"][0]["function"]["name"], "ping"
        )
        self.assertEqual(
            calls[2].kwargs["response_format"], {"type": "json_object"}
        )
        self.assertEqual(calls[2].kwargs["tool_choice"], "auto")
        self.assertEqual(
            calls[2].kwargs["tools"][0]["function"]["name"], "ping"
        )

        text, tools, combined = tracker.calls
        self.assertNotIn("tool_names", text.request_config)
        self.assertNotIn("response_format", text.request_config)
        self.assertEqual(tools.request_config["tool_names"], ["ping"])
        self.assertNotIn("response_format", tools.request_config)
        self.assertEqual(
            combined.request_config["response_format"], {"type": "json_object"}
        )
        self.assertEqual(combined.request_config["tool_names"], ["ping"])
        self.assertEqual(combined.request_config["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()

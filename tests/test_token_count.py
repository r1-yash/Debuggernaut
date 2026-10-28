"""Tests for preflight input-token counting and the request limit.

Counting tests use a fake encoder, so they never download a tokenizer and never
touch the network.  ``OfficialEncoderTests`` exercises the real DeepSeek V4.1
prompt encoding and is skipped unless ``DEEPSEEK_TOKENIZER_PATH`` points at a
local ``tokenizer.json`` (see ``src/swe_agent/token_count.py``).
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_agent.agent.diagnose import build_diagnosis_prompt

# Imported under a non-test name so pytest does not try to collect it.
from swe_agent.agent.executor import TestOutcome as _TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.llm import (
    DEFAULT_TOKENHARBOR_MODEL,
    ContextTooLarge,
    OpenAICompatibleProvider,
    UsageTracker,
    usage_run,
)
from swe_agent.token_count import (
    EXACT_METHOD,
    MISSING_RESOURCE_METHOD,
    OTHER_MODEL_METHOD,
    TOKENIZER_PATH_ENV,
    UNSUPPORTED_REQUEST_METHOD,
    TokenCountUnavailable,
    conservative_byte_estimate,
    count_input_tokens,
    is_deepseek_v41_flash,
    reset_cached_encoder,
)


class _FakeEncoder:
    """Stand-in for the official encoder that records what it was given."""

    def __init__(self, tokens: int = 7) -> None:
        self.tokens = tokens
        self.bodies: list[dict] = []

    def count(self, body) -> int:
        self.bodies.append(dict(body))
        return self.tokens


def _completion_reply(total_tokens: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None))],
        usage={"input_tokens": 11, "output_tokens": total_tokens, "total_tokens": total_tokens},
    )


class CountingMethodTests(unittest.TestCase):
    """Every count says which method produced it, and fallbacks are explicit."""

    def setUp(self) -> None:
        reset_cached_encoder()

    def tearDown(self) -> None:
        reset_cached_encoder()

    def test_complete_request_is_counted_with_the_official_encoding(self) -> None:
        encoder = _FakeEncoder(tokens=4321)
        body = {
            "model": DEFAULT_TOKENHARBOR_MODEL,
            "messages": [
                {"role": "system", "content": "You are a senior software engineer."},
                {"role": "user", "content": "Issue title: split dollar quoted strings"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "repository_read_file",
                                "arguments": '{"file_path": "sqlparse/engine/statement_splitter.py"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '["line one"]'},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "repository_read_file",
                        "description": "Read a file relative to the repository.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "none",
        }

        with patch("swe_agent.token_count._request_encoder", return_value=encoder):
            count = count_input_tokens(
                body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
            )

        self.assertEqual(count.method, EXACT_METHOD)
        self.assertTrue(count.exact)
        self.assertEqual(count.tokens, 4321)
        # The counter sees the whole outgoing request, not just message text.
        self.assertEqual(encoder.bodies, [body])

    def test_unavailable_tokenizer_falls_back_and_is_reported(self) -> None:
        body = {"model": DEFAULT_TOKENHARBOR_MODEL, "messages": [{"role": "user", "content": "hi"}]}
        failure = TokenCountUnavailable(
            "the deepseek-recipe package is not installed (No module named 'deepseek_recipe')"
        )

        with patch("swe_agent.token_count._request_encoder", side_effect=failure):
            with self.assertLogs("swe_agent.token_count", level="WARNING") as logged:
                first = count_input_tokens(
                    body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
                )
                second = count_input_tokens(
                    body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
                )

        self.assertFalse(first.exact)
        self.assertEqual(first.method, MISSING_RESOURCE_METHOD)
        self.assertEqual(first.tokens, conservative_byte_estimate(body))
        self.assertEqual(second.tokens, first.tokens)
        # A fallback is announced once, and never described as an exact count.
        self.assertEqual(len(logged.records), 1)
        message = logged.records[0].getMessage()
        self.assertIn("approximation, not an exact count", message)
        self.assertIn("not installed", message)

    def test_unsupported_request_format_falls_back_and_is_recorded(self) -> None:
        body = {
            "model": DEFAULT_TOKENHARBOR_MODEL,
            "messages": [{"role": "user", "content": "no json keyword here"}],
            "response_format": {"type": "json_object"},
        }
        failure = TokenCountUnavailable(
            "the official V4.1 prompt encoding rejected the request "
            "(ConversionError: Prompt must contain the word 'json')",
            unsupported=True,
        )

        with patch("swe_agent.token_count._request_encoder", side_effect=failure):
            with self.assertLogs("swe_agent.token_count", level="WARNING"):
                count = count_input_tokens(
                    body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
                )

        self.assertFalse(count.exact)
        self.assertEqual(count.method, UNSUPPORTED_REQUEST_METHOD)
        self.assertEqual(count.tokens, conservative_byte_estimate(body))

    def test_other_provider_keeps_the_conservative_estimate(self) -> None:
        body = {"model": "qwen/qwen3-coder:free", "messages": [{"role": "user", "content": "x"}]}
        with patch("swe_agent.token_count._request_encoder") as encoder:
            with self.assertLogs("swe_agent.token_count", level="INFO") as logged:
                count = count_input_tokens(
                    body, provider="qwen", model="qwen/qwen3-coder:free"
                )
            self.assertEqual(logged.records[0].levelname, "INFO")

        encoder.assert_not_called()
        self.assertFalse(count.exact)
        self.assertEqual(count.method, OTHER_MODEL_METHOD)
        self.assertEqual(count.tokens, conservative_byte_estimate(body))

    def test_failed_loading_is_remembered_instead_of_retried(self) -> None:
        loader = Mock(side_effect=TokenCountUnavailable("the tokenizer could not be loaded"))
        body = {"model": DEFAULT_TOKENHARBOR_MODEL, "messages": [{"role": "user", "content": "hi"}]}

        with patch("swe_agent.token_count._load_encoder", loader):
            with self.assertLogs("swe_agent.token_count", level="WARNING"):
                for _ in range(3):
                    count_input_tokens(
                        body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
                    )

        self.assertEqual(loader.call_count, 1)

    def test_reset_cached_encoder_retries_after_resources_become_available(self) -> None:
        encoder = _FakeEncoder(tokens=99)
        body = {"model": DEFAULT_TOKENHARBOR_MODEL, "messages": [{"role": "user", "content": "hi"}]}
        loader = Mock(
            side_effect=[
                TokenCountUnavailable("the tokenizer could not be loaded"),
                encoder,
            ]
        )

        with patch("swe_agent.token_count._load_encoder", loader):
            with self.assertLogs("swe_agent.token_count", level="WARNING"):
                degraded = count_input_tokens(
                    body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
                )
            reset_cached_encoder()
            exact = count_input_tokens(
                body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
            )

        self.assertFalse(degraded.exact)
        self.assertTrue(exact.exact)
        self.assertEqual(exact.tokens, 99)

    def test_model_family_detection(self) -> None:
        self.assertTrue(is_deepseek_v41_flash(DEFAULT_TOKENHARBOR_MODEL))
        self.assertTrue(is_deepseek_v41_flash("deepseek-v4.1-flash"))
        self.assertFalse(is_deepseek_v41_flash("deepseek-v4-pro"))
        self.assertFalse(is_deepseek_v41_flash("qwen/qwen3-coder:free"))


class ProviderPreflightLimitTests(unittest.TestCase):
    """The limit is enforced with the preflight count, before the network call."""

    def _provider(self, **overrides) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            "tokenharbor", DEFAULT_TOKENHARBOR_MODEL, "secret",
            "https://tokenharbor.ai/v1", max_input_tokens=120_000,
            max_retries=0, sleep=lambda _: self.fail("must not sleep"),
            **overrides,
        )

    @patch("openai.OpenAI")
    def test_oversized_request_is_blocked_before_the_provider_call(self, openai_client) -> None:
        provider = self._provider()
        tracker = UsageTracker()

        with patch(
            "swe_agent.llm.count_input_tokens",
            return_value=SimpleNamespace(tokens=120_001, method=EXACT_METHOD, exact=True),
        ):
            with usage_run(tracker), self.assertRaises(ContextTooLarge) as raised:
                provider.structured("system", "hello", dict, "diagnosis")

        openai_client.return_value.chat.completions.create.assert_not_called()
        self.assertIn("120001", str(raised.exception))
        self.assertIn(EXACT_METHOD, str(raised.exception))
        self.assertEqual(tracker.calls[0].estimated_input_tokens, 120_001)
        self.assertEqual(tracker.calls[0].input_token_limit, 120_000)
        self.assertEqual(tracker.calls[0].input_token_count_method, EXACT_METHOD)
        self.assertFalse(tracker.calls[0].success)

    @patch("openai.OpenAI")
    def test_request_within_the_limit_is_sent(self, openai_client) -> None:
        openai_client.return_value.chat.completions.create.return_value = _completion_reply()
        provider = self._provider()
        tracker = UsageTracker()

        with patch(
            "swe_agent.llm.count_input_tokens",
            return_value=SimpleNamespace(tokens=119_999, method=EXACT_METHOD, exact=True),
        ):
            with usage_run(tracker):
                provider.structured("system", "hello", dict, "diagnosis")

        openai_client.return_value.chat.completions.create.assert_called_once()
        self.assertEqual(tracker.calls[0].estimated_input_tokens, 119_999)
        self.assertEqual(tracker.calls[0].input_token_count_method, EXACT_METHOD)

    @patch("openai.OpenAI")
    def test_the_counter_receives_the_exact_request_that_is_sent(self, openai_client) -> None:
        openai_client.return_value.chat.completions.create.return_value = _completion_reply()
        provider = self._provider()
        seen: list[dict] = []

        def record(payload, *, provider, model):
            seen.append(dict(payload))
            return SimpleNamespace(tokens=10, method=EXACT_METHOD, exact=True)

        with patch("swe_agent.llm.count_input_tokens", side_effect=record):
            provider.structured("You classify failures.", "the test output", dict, "diagnosis")

        sent = openai_client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(seen, [{"model": DEFAULT_TOKENHARBOR_MODEL, **sent}])

    @patch("openai.OpenAI")
    def test_provider_reported_usage_is_still_tracked(self, openai_client) -> None:
        openai_client.return_value.chat.completions.create.return_value = _completion_reply()
        provider = self._provider()
        tracker = UsageTracker()

        with patch(
            "swe_agent.llm.count_input_tokens",
            return_value=SimpleNamespace(tokens=42, method=EXACT_METHOD, exact=True),
        ):
            with usage_run(tracker):
                provider.structured("system", "hello", dict, "diagnosis")

        call = tracker.calls[0]
        self.assertEqual(call.input_tokens, 11)
        self.assertEqual(call.total_tokens, 5)
        self.assertEqual(call.estimated_input_tokens, 42)
        self.assertEqual(call.input_token_count_method, EXACT_METHOD)


class SqlparseOversizedEstimateRegressionTests(unittest.TestCase):
    """A bounded fixture for the recorded sqlparse diagnosis over-estimate.

    The recorded run estimated one ``sqlparse`` diagnosis request at 147,690,175
    tokens, roughly 295 MB of payload, while the request only carries the
    diagnosis prompt.  These tests pin the two properties that matter with a
    bounded fixture built from the real prompt construction: the count is the
    single prompt's token count (no duplication or expansion), and it comes from
    the exact method rather than a byte-length guess.
    """

    _TEST_OUTPUT_LINES = 4_000

    @classmethod
    def setUpClass(cls) -> None:
        cls.fix_result = FixResult(
            file_path="sqlparse/engine/statement_splitter.py",
            original_content="",
            new_content="",
            reasoning="Widen the dollar-quote check in the splitter.",
            exploration_rounds=3,
        )
        cls.outcome = _TestOutcome(
            passed=False,
            return_code=1,
            stdout="\n".join(
                "FAILED tests/test_parse.py::test_psql_quotation_marks - "
                "AssertionError: assert 2 == 1"
                for _ in range(cls._TEST_OUTPUT_LINES)
            ),
            stderr="",
        )

    def _request_body(self) -> dict:
        return {
            "model": DEFAULT_TOKENHARBOR_MODEL,
            "messages": [
                {"role": "system", "content": "You diagnose failed software-maintenance fixes."},
                {"role": "user", "content": build_diagnosis_prompt(self.fix_result, self.outcome)},
            ],
            "response_format": {"type": "json_object"},
        }

    def setUp(self) -> None:
        reset_cached_encoder()

    def tearDown(self) -> None:
        reset_cached_encoder()

    def test_diagnosis_request_is_counted_once_without_expansion(self) -> None:
        body = self._request_body()
        prompt = body["messages"][1]["content"]
        # The official encoder counts the rendered prompt, so a fake encoder that
        # budgets three characters per token has to see the prompt exactly once.
        encoder = _FakeEncoder(tokens=len(prompt) // 3)

        with patch("swe_agent.token_count._request_encoder", return_value=encoder):
            count = count_input_tokens(
                body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
            )

        self.assertTrue(count.exact)
        self.assertEqual(encoder.bodies, [body])
        # Only the system message and the one diagnosis prompt are counted.
        self.assertEqual(len(encoder.bodies[0]["messages"]), 2)
        self.assertLessEqual(count.tokens, (len(json.dumps(body)) // 3) + 1)

    def test_bytes_heuristic_over_estimates_this_shape(self) -> None:
        body = self._request_body()
        encoder = _FakeEncoder(tokens=len(body["messages"][1]["content"]) // 4)

        with patch("swe_agent.token_count._request_encoder", return_value=encoder):
            count = count_input_tokens(
                body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
            )

        # This is the recorded failure mode: the byte-length estimate reports
        # well above the real prompt length, refusing requests that fit.
        self.assertGreater(conservative_byte_estimate(body), count.tokens * 1.25)


class OfficialEncoderTests(unittest.TestCase):
    """Accuracy checks against the real DeepSeek V4.1 prompt encoding."""

    @classmethod
    def setUpClass(cls) -> None:
        path = os.getenv(TOKENIZER_PATH_ENV, "").strip()
        if not path or not os.path.isfile(path):
            raise unittest.SkipTest(
                f"set {TOKENIZER_PATH_ENV} to a local tokenizer.json to run these tests"
            )
        cls.path = path

    def setUp(self) -> None:
        reset_cached_encoder()
        self._previous = os.environ.get(TOKENIZER_PATH_ENV)
        os.environ[TOKENIZER_PATH_ENV] = self.path

    def tearDown(self) -> None:
        reset_cached_encoder()
        if self._previous is None:
            os.environ.pop(TOKENIZER_PATH_ENV, None)
        else:
            os.environ[TOKENIZER_PATH_ENV] = self._previous

    def _count(self, body: dict) -> int:
        count = count_input_tokens(
            body, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
        )
        self.assertTrue(count.exact, count.method)
        self.assertEqual(count.method, EXACT_METHOD)
        return count.tokens

    @staticmethod
    def _tool() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "repository_read_file",
                "description": "Read a file relative to the repository being investigated.",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }

    def _conversation(self) -> list[dict]:
        return [
            {"role": "system", "content": "You are a senior software engineer."},
            {"role": "user", "content": "Issue title: split dollar quoted strings"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "repository_read_file",
                            "arguments": '{"file_path": "sqlparse/engine/statement_splitter.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '["line one", "line two"]'},
            {"role": "assistant", "content": "I found the splitter branch."},
            {"role": "user", "content": "Now output your proposed fix as JSON."},
        ]

    def test_tool_definitions_are_counted(self) -> None:
        messages = self._conversation()
        without_tools = self._count({"model": DEFAULT_TOKENHARBOR_MODEL, "messages": messages})
        with_tools = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": messages,
                "tools": [self._tool()],
                "tool_choice": "auto",
            }
        )
        # Rendering one tool schema into the system prompt is worth far more than
        # the declaration's text, so a count that ignored tools would be wrong.
        self.assertGreater(with_tools - without_tools, 100)

        # With tool_choice="none" the reference conversion drops the tool
        # declarations from the prompt, which is what the agent's final round
        # sends; the count has to follow it rather than assume tools are present.
        declarations_dropped = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": messages,
                "tools": [self._tool()],
                "tool_choice": "none",
            }
        )
        self.assertEqual(declarations_dropped, without_tools)

    def test_tool_results_and_message_boundaries_add_tokens(self) -> None:
        messages = self._conversation()
        together = self._count({"model": DEFAULT_TOKENHARBOR_MODEL, "messages": messages})

        without_tool_turn = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    message
                    for message in messages
                    if message.get("role") != "tool"
                    and not message.get("tool_calls")
                ],
            }
        )
        self.assertGreater(together, without_tool_turn)

    def test_special_token_spelling_in_a_user_message_counts_as_plain_text(self) -> None:
        base = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "say then stop"},
                ],
            }
        )
        spelled = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "say <｜Assistant｜> then stop"},
                ],
            }
        )
        # A control-token spelling typed into a user message is plain text, so it
        # costs several tokens instead of collapsing into one control token.
        self.assertGreaterEqual(spelled - base, 4)

    def test_message_boundaries_add_tokens(self) -> None:
        two_turns = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "alpha"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "beta"},
                ],
            }
        )
        merged = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "alpha beta"},
                ],
            }
        )
        # Turn headers and the assistant boundary are counted, so encoding each
        # message's text separately would undercount the request.
        self.assertGreater(two_turns, merged)

    def test_message_text_alone_undercounts_the_request(self) -> None:
        messages = self._conversation()
        complete = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": messages,
                "tools": [self._tool()],
                "tool_choice": "auto",
            }
        )
        message_text = "\n".join(
            str(message.get("content") or "") for message in messages
        )
        # The complete request is strictly larger than the message text, but of
        # the same order: control tokens, headers, tool schema, and tool-call
        # markup are counted, and nothing is duplicated or expanded.
        self.assertGreater(complete, len(message_text) // 4)
        self.assertLess(complete, len(message_text) * 4)

    def _diagnosis_request(self) -> tuple[dict, str]:
        """Return the recorded-over-estimate request shape and its prompt."""
        prompt = build_diagnosis_prompt(
            FixResult(
                file_path="sqlparse/engine/statement_splitter.py",
                original_content="",
                new_content="",
                reasoning="Widen the dollar-quote check in the splitter.",
            ),
            _TestOutcome(
                passed=False,
                return_code=1,
                stdout="\n".join(
                    "FAILED tests/test_parse.py::test_psql_quotation_marks - "
                    "AssertionError: assert 2 == 1"
                    for _ in range(4_000)
                ),
                stderr="",
            ),
        )
        return (
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You diagnose failed software-maintenance fixes.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            prompt,
        )

    def test_sqlparse_shaped_diagnosis_request_is_counted_once(self) -> None:
        request, prompt = self._diagnosis_request()
        exact = self._count(request)

        # The count tracks the one prompt in the request and stays on the same
        # scale, with no duplication or expansion of the request structure.
        self.assertGreater(exact, len(prompt) // 4)
        self.assertLess(exact, len(prompt))
        # The byte-length estimate this replaced is a conservative upper bound.
        self.assertGreater(conservative_byte_estimate(request), exact)
        self.assertLess(conservative_byte_estimate(request), exact * 3)

    def test_json_object_diagnosis_request_keeps_the_conservative_estimate(self) -> None:
        request, _ = self._diagnosis_request()
        request = {**request, "response_format": {"type": "json_object"}}

        with self.assertLogs("swe_agent.token_count", level="WARNING") as logged:
            count = count_input_tokens(
                request, provider="tokenharbor", model=DEFAULT_TOKENHARBOR_MODEL
            )

        # The reference conversion rejects a json_object request whose prompt
        # never says "json", so the count stays conservative and says so.
        self.assertFalse(count.exact)
        self.assertEqual(count.method, UNSUPPORTED_REQUEST_METHOD)
        self.assertEqual(count.tokens, conservative_byte_estimate(request))
        self.assertIn("approximation, not an exact count", logged.records[0].getMessage())

    def test_the_sqlparse_pinned_test_output_is_tiny(self) -> None:
        # Evidence for the recorded 147,690,175-token estimate: the pinned
        # sqlparse test output is far too small to produce it, so the anomaly
        # did not come from this request's prompt in a normal run.
        outcome = _TestOutcome(
            passed=False,
            return_code=1,
            stdout=(
                "FAILED tests/test_parse.py::test_psql_quotation_marks - "
                "AssertionError: assert 2 == 1\n1 failed, 74 passed in 0.12s\n"
            ),
            stderr="",
        )
        prompt = build_diagnosis_prompt(
            FixResult(
                file_path="sqlparse/engine/statement_splitter.py",
                original_content="",
                new_content="",
                reasoning="Widen the dollar-quote check in the splitter.",
            ),
            outcome,
        )
        self.assertLess(len(prompt), 10_000)
        count = self._count(
            {
                "model": DEFAULT_TOKENHARBOR_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        self.assertLess(count, 3_000)


if __name__ == "__main__":
    unittest.main()

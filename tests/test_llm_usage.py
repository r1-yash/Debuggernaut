import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_agent.agent.loop import LoopResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.reporting import RunTiming, write_run_report
from swe_agent.llm import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_TOKENHARBOR_MODEL,
    QWEN_OPENROUTER_BASE_URL,
    ContextTooLarge,
    LLMSettings,
    OpenAICompatibleProvider,
    ProviderUnavailable,
    UsageTracker,
    usage_run,
    normalize_usage,
    provider_from_environment,
)


class UsageTrackingTests(unittest.TestCase):
    def test_successful_call_normalizes_gemini_usage(self) -> None:
        usage = normalize_usage(SimpleNamespace(usage_metadata=SimpleNamespace(
            prompt_token_count=11, candidates_token_count=7, total_token_count=18
        )))
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.total_tokens), (11, 7, 18))

    def test_multiple_calls_and_attempts_aggregate(self) -> None:
        tracker = UsageTracker()
        tracker.set_attempt(1)
        tracker.record("gemini", "model", "exploration", SimpleNamespace(usage=SimpleNamespace(input_tokens=2, output_tokens=3, total_tokens=5)), success=True)
        tracker.set_attempt(2)
        tracker.record("qwen", "model", "diagnosis", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11, total_tokens=18)), success=True)
        self.assertEqual([call.attempt for call in tracker.calls], [1, 2])
        self.assertEqual(tracker.totals().total_tokens, 23)

    def test_missing_usage_and_failed_call_are_recorded(self) -> None:
        tracker = UsageTracker()
        tracker.record("qwen", "model", "fix_generation", success=False)
        self.assertFalse(tracker.calls[0].success)
        self.assertIsNone(tracker.calls[0].total_tokens)
        self.assertIsNone(tracker.totals().total_tokens)

    def test_qwen_and_gemini_configuration_from_environment(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "qwen", "LLM_MODEL": "qwen3-coder-next"}, clear=False):
            self.assertEqual(LLMSettings.from_environment().provider, "qwen")
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "LLM_MODEL": "custom-gemini"}, clear=False):
            self.assertEqual(LLMSettings.from_environment().model, "custom-gemini")

    @patch("openai.OpenAI")
    def test_qwen_openrouter_configuration_is_read_from_environment(self, openai_client) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "qwen", "LLM_MODEL": "qwen/qwen3-coder:free",
            "QWEN_API_KEY": "test-key-not-a-real-secret",
            "QWEN_BASE_URL": "https://example.test/openrouter/v1",
        }, clear=False):
            provider = provider_from_environment()
        self.assertEqual(provider.provider, "qwen")
        self.assertEqual(provider.model, "qwen/qwen3-coder:free")
        self.assertEqual(openai_client.call_args.kwargs["api_key"], "test-key-not-a-real-secret")
        self.assertEqual(
            openai_client.call_args.kwargs["base_url"],
            "https://example.test/openrouter/v1",
        )

    @patch("openai.OpenAI")
    def test_qwen_defaults_to_openrouter_model_and_url(self, openai_client) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "qwen",
            "QWEN_API_KEY": "test-key-not-a-real-secret",
        }, clear=True):
            provider = provider_from_environment()
        self.assertEqual(provider.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(DEFAULT_QWEN_MODEL, "qwen/qwen3-coder:free")
        self.assertEqual(
            openai_client.call_args.kwargs["base_url"], QWEN_OPENROUTER_BASE_URL
        )

    @patch("openai.OpenAI")
    def test_tokenharbor_requires_limit_and_uses_configured_values(self, openai_client) -> None:
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "tokenharbor",
            "TOKENHARBOR_API_KEY": "tokenharbor-secret",
            "TOKENHARBOR_MAX_INPUT_TOKENS": "321",
        }, clear=True):
            provider = provider_from_environment()
        self.assertEqual(provider.provider, "tokenharbor")
        self.assertEqual(provider.model, DEFAULT_TOKENHARBOR_MODEL)
        self.assertEqual(provider.base_url_host, "tokenharbor.ai")
        self.assertEqual(provider.max_input_tokens, 321)
        with patch.dict(os.environ, {"LLM_PROVIDER": "tokenharbor"}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires --max-input-tokens"):
                LLMSettings.from_environment()

    @patch("openai.OpenAI")
    def test_retry_honors_retry_after_then_records_usage(self, openai_client) -> None:
        class RateLimited(Exception):
            status_code = 429
            headers = {"retry-after": "2"}
        client = openai_client.return_value
        client.chat.completions.create.side_effect = [RateLimited(), SimpleNamespace(usage={"total_tokens": 5})]
        slept: list[float] = []
        provider = OpenAICompatibleProvider(
            "tokenharbor", "model", "secret", "https://tokenharbor.ai/v1",
            max_input_tokens=1000, sleep=slept.append,
        )
        tracker = UsageTracker()
        with usage_run(tracker):
            provider.structured("system", "hello", dict, "diagnosis")
        self.assertEqual(slept, [2.0])
        self.assertEqual(len(tracker.calls), 1)
        self.assertEqual(tracker.calls[0].total_tokens, 5)

    @patch("openai.OpenAI")
    def test_exhausted_rate_limits_raise_provider_unavailable_without_key(self, openai_client) -> None:
        class RateLimited(Exception):
            status_code = 429
        openai_client.return_value.chat.completions.create.side_effect = [
            RateLimited("key=tokenharbor-secret") for _ in range(3)
        ]
        provider = OpenAICompatibleProvider(
            "tokenharbor", "model", "tokenharbor-secret",
            "https://tokenharbor.ai/v1", max_input_tokens=1000,
            max_retries=2, sleep=lambda _: None,
        )

        with self.assertRaises(ProviderUnavailable) as raised:
            provider.structured("system", "hello", dict, "diagnosis")

        self.assertNotIn("tokenharbor-secret", str(raised.exception))
        self.assertEqual(
            openai_client.return_value.chat.completions.create.call_count, 3,
        )

    @patch("openai.OpenAI")
    def test_retry_after_cap_and_transport_errors_use_configured_policy(self, openai_client) -> None:
        class RateLimited(Exception):
            status_code = 429
            headers = {"retry-after": "61"}
        client = openai_client.return_value
        provider = OpenAICompatibleProvider(
            "tokenharbor", "model", "secret", "https://tokenharbor.ai/v1",
            max_input_tokens=1000, max_retry_after_seconds=60,
            sleep=lambda _: self.fail("oversized Retry-After must not sleep"),
        )
        client.chat.completions.create.side_effect = RateLimited()
        with self.assertRaisesRegex(ProviderUnavailable, "Retry-After"):
            provider.structured("system", "hello", dict, "diagnosis")

        sleeps: list[float] = []
        client.chat.completions.create.side_effect = [
            ConnectionError("offline"), SimpleNamespace(usage={"total_tokens": 1}),
        ]
        provider = OpenAICompatibleProvider(
            "tokenharbor", "model", "secret", "https://tokenharbor.ai/v1",
            max_input_tokens=1000, sleep=sleeps.append, random_value=lambda: 0,
        )
        provider.structured("system", "hello", dict, "diagnosis")
        self.assertEqual(sleeps, [1.0])

    @patch("openai.OpenAI")
    def test_feature_rejection_and_context_limit_are_recorded_safely(self, openai_client) -> None:
        class Unsupported(Exception):
            status_code = 400
        client = openai_client.return_value
        client.chat.completions.create.side_effect = Unsupported("tools unsupported")
        provider = OpenAICompatibleProvider("tokenharbor", "model", "secret", "https://tokenharbor.ai/v1", max_input_tokens=1000)
        with self.assertRaises(ProviderUnavailable):
            provider.preflight()
        limited = OpenAICompatibleProvider("tokenharbor", "model", "secret", "https://tokenharbor.ai/v1", max_input_tokens=1)
        tracker = UsageTracker()
        with usage_run(tracker), self.assertRaises(ContextTooLarge):
            limited.structured("system", "hello", dict, "diagnosis")
        self.assertGreater(tracker.calls[0].estimated_input_tokens, 1)
        self.assertEqual(tracker.calls[0].input_token_limit, 1)

    def test_report_contains_usage_but_not_secret(self) -> None:
        tracker = UsageTracker()
        tracker.set_attempt(1)
        tracker.record("qwen", "qwen3-coder-next", "exploration", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)), success=True)
        loop = LoopResult(succeeded=False, final_attempt=None, attempts=[], llm_calls=tracker.calls)
        entry = BenchmarkEntry(
            owner="acme", repository="widget", issue_number=42,
            commit="a" * 40, test_command=["pytest"],
        )
        timing = RunTiming(
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            finished_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            duration_seconds=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            report = write_run_report(
                directory, entry, "qwen", "qwen3-coder-next", loop,
                "failed", None, timing,
            )
            payload = json.loads(Path(report).read_text())
        self.assertEqual(payload["total_tokens"], 3)
        self.assertEqual(payload["calls"][0]["attempt"], 1)
        self.assertNotIn("QWEN_API_KEY", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()

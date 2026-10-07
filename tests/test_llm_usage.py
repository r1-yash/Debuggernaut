import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swe_agent.agent.loop import LoopResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.reporting import write_run_report
from swe_agent.llm import (
    DEFAULT_QWEN_MODEL,
    QWEN_OPENROUTER_BASE_URL,
    LLMSettings,
    UsageTracker,
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

    def test_report_contains_usage_but_not_secret(self) -> None:
        tracker = UsageTracker()
        tracker.set_attempt(1)
        tracker.record("qwen", "qwen3-coder-next", "exploration", SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)), success=True)
        loop = LoopResult(succeeded=False, final_attempt=None, attempts=[], llm_calls=tracker.calls)
        with tempfile.TemporaryDirectory() as directory:
            report = write_run_report(directory, "acme", "widget", 42, "qwen", "qwen3-coder-next", loop, "failed", None)
            payload = json.loads(Path(report).read_text())
        self.assertEqual(payload["total_tokens"], 3)
        self.assertEqual(payload["calls"][0]["attempt"], 1)
        self.assertNotIn("QWEN_API_KEY", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()

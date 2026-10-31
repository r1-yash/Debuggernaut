#!/usr/bin/env python3
"""Isolate which request feature TokenHarbor rejects for DeepSeek V4.1 Flash.

The original failure was a diagnosis request: two chat messages, no tools, and
``response_format={"type": "json_object"}``.  This script sends minimal prompt
text only, never repository source, and prints only request shapes, statuses,
and redacted provider error details.

Two checks are available:

* The official DeepSeek V4.1 converter (``deepseek-recipe``, already used by
  Debuggernaut's token counter) validates the same request bodies locally.
* When ``TOKENHARBOR_API_KEY`` is set, the live check sends the same bodies to
  TokenHarbor with hidden client retries disabled.

The API key is read from the environment (or ``.env``) and is never printed.
Without a key the script still runs the local converter check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from swe_agent.agent.diagnose import (
    _SYSTEM_INSTRUCTION as CURRENT_DIAGNOSIS_SYSTEM,
    build_diagnosis_prompt,
)
from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import FixResult
from swe_agent.llm import (
    DEFAULT_TOKENHARBOR_MODEL,
    TOKENHARBOR_BASE_URL,
    TokenHarborProvider,
    UsageTracker,
    usage_run,
)

_ORIGINAL_DIAGNOSIS_SYSTEM = """You diagnose failed software-maintenance fixes.
Classify the failure as `wrong_file` only when the proposed file was not the
right target at all. Classify it as `wrong_fix` when the file was relevant but
the proposed change's logic was incorrect or incomplete. Explain concisely."""

_PING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Report that the provider is reachable.",
        "parameters": {"type": "object", "properties": {}},
    },
}


@dataclass(frozen=True)
class Case:
    """One independent request configuration to test."""

    name: str
    description: str
    kwargs: dict[str, Any]


def _minimal_diagnosis_prompt(*, include_json_instruction: bool) -> str:
    """Build a diagnosis prompt without repository content or sensitive data."""
    prompt = build_diagnosis_prompt(
        FixResult(
            file_path="parser.py",
            original_content="",
            new_content="",
            reasoning="The proposed change returns None.",
        ),
        TestOutcome(
            passed=False,
            return_code=1,
            stdout="1 failed",
            stderr="AssertionError: expected a mapping",
        ),
    )
    if include_json_instruction:
        return prompt
    marker = "\n\nReturn the diagnosis as a JSON object"
    return prompt.split(marker, 1)[0]


def _text_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a connectivity check."},
        {"role": "user", "content": "Reply with the single word: ready."},
    ]


def _diagnosis_messages(*, json_aware: bool) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                CURRENT_DIAGNOSIS_SYSTEM
                if json_aware
                else _ORIGINAL_DIAGNOSIS_SYSTEM
            ),
        },
        {
            "role": "user",
            "content": _minimal_diagnosis_prompt(
                include_json_instruction=json_aware
            ),
        },
    ]


def build_cases() -> list[Case]:
    """Return the independent request shapes used by the diagnostic."""
    original_diagnosis = _diagnosis_messages(json_aware=False)
    current_diagnosis = _diagnosis_messages(json_aware=True)
    json_mode = {"type": "json_object"}
    return [
        Case(
            name="text_only",
            description="Ordinary text request; no JSON mode, no tools.",
            kwargs={"messages": _text_messages()},
        ),
        Case(
            name="json_mode_original_diagnosis",
            description=(
                "Original diagnosis shape: JSON mode, no tools, and a prompt "
                "that never contains the word 'json'."
            ),
            kwargs={
                "messages": original_diagnosis,
                "response_format": json_mode,
            },
        ),
        Case(
            name="tools_only",
            description="Tool calling only; no JSON mode.",
            kwargs={
                "messages": _text_messages(),
                "tools": [_PING_TOOL],
                "tool_choice": "auto",
            },
        ),
        Case(
            name="json_and_tools_original_diagnosis",
            description=(
                "Combined original diagnosis shape: JSON mode plus tools, "
                "with a prompt that never contains the word 'json'."
            ),
            kwargs={
                "messages": original_diagnosis,
                "tools": [_PING_TOOL],
                "tool_choice": "auto",
                "response_format": json_mode,
            },
        ),
        Case(
            name="json_mode_current_diagnosis",
            description=(
                "Current diagnosis shape: JSON mode, no tools, and a prompt "
                "that explicitly contains 'JSON'."
            ),
            kwargs={
                "messages": current_diagnosis,
                "response_format": json_mode,
            },
        ),
        Case(
            name="json_and_tools_current_diagnosis",
            description=(
                "Combined current diagnosis shape: JSON mode plus tools, with "
                "an explicit 'JSON' instruction in the prompt."
            ),
            kwargs={
                "messages": current_diagnosis,
                "tools": [_PING_TOOL],
                "tool_choice": "auto",
                "response_format": json_mode,
            },
        ),
    ]


def _converter_check(case: Case, model: str) -> dict[str, Any]:
    """Validate one body with the official DeepSeek V4.1 converter."""
    try:
        from deepseek_recipe import (
            ChatCompletionRequest,
            ConversionOptions,
            ModelVersion,
        )
    except ImportError as error:
        return {
            "status": "unavailable",
            "error": f"deepseek-recipe is not installed ({error})",
        }

    options = ConversionOptions().with_model_alias(
        model, ModelVersion.DeepseekV41Flash
    )
    try:
        request = ChatCompletionRequest({"model": model, **case.kwargs})
        request.convert(options)
    except Exception as error:  # noqa: BLE001 - report every converter failure.
        return {
            "status": "rejected",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return {"status": "accepted"}


def _response_summary(response: Any) -> dict[str, Any]:
    """Summarize a successful answer without dumping generated content."""
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return {"message": None}
    calls = list(getattr(message, "tool_calls", None) or [])
    content = getattr(message, "content", None)
    return {
        "content_chars": len(content) if isinstance(content, str) else 0,
        "tool_call_names": [
            getattr(getattr(call, "function", None), "name", None)
            for call in calls
        ],
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
    }


def _live_check(
    provider: TokenHarborProvider, case: Case
) -> dict[str, Any]:
    """Send one request through the production adapter and record safe details."""
    tracker = UsageTracker()
    error: Exception | None = None
    response: Any = None
    try:
        with usage_run(tracker):
            response = provider._call(f"diagnostic_{case.name}", **case.kwargs)
    except Exception as caught:  # noqa: BLE001 - the failure is the observation.
        error = caught

    record = tracker.calls[-1] if tracker.calls else None
    result: dict[str, Any] = {
        "status": "failed" if error is not None else "succeeded",
        "exception": type(error).__name__ if error is not None else None,
        "http_status": getattr(error, "status_code", None),
        "request_config": getattr(record, "request_config", None),
        "error_details": getattr(record, "error", None),
        "response": _response_summary(response) if response is not None else None,
    }
    return result


def _print_result(mode: str, case: Case, result: dict[str, Any]) -> None:
    label = result.get("status", "unknown")
    print(f"[{label.upper()}] {mode}: {case.name}")
    print(f"  {case.description}")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("TOKENHARBOR_MODEL", DEFAULT_TOKENHARBOR_MODEL),
        help=f"TokenHarbor model (default: {DEFAULT_TOKENHARBOR_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("TOKENHARBOR_BASE_URL", TOKENHARBOR_BASE_URL),
        help="TokenHarbor OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=120_000,
        help="Local input-token ceiling for live requests (default: 120000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per live request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only run the official-converter check, never contact TokenHarbor.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the offline converter check and, when possible, the live comparison."""
    load_dotenv()
    args = _parse_args(argv)
    cases = build_cases()

    print(f"TokenHarbor diagnostic model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print()
    print("== Official DeepSeek V4.1 converter check (local, no network) ==")
    for case in cases:
        _print_result("converter", case, _converter_check(case, args.model))

    api_key = os.getenv("TOKENHARBOR_API_KEY", "").strip()
    if args.offline:
        print("\nLive check skipped: --offline was requested.")
        return 0
    if not api_key:
        print(
            "\nLive check skipped: TOKENHARBOR_API_KEY is not set. "
            "Add it to the environment or .env and rerun without --offline."
        )
        return 0

    print("\n== Live TokenHarbor requests (API key configured, value hidden) ==")
    provider = TokenHarborProvider(
        args.model,
        api_key,
        args.base_url,
        max_input_tokens=args.max_input_tokens,
        max_retries=0,
    )
    # Disable the OpenAI SDK's own hidden retries so each result is one request.
    provider.client = provider.client.with_options(
        timeout=args.timeout, max_retries=0
    )
    for case in cases:
        _print_result("live", case, _live_check(provider, case))
    return 0


if __name__ == "__main__":
    sys.exit(main())

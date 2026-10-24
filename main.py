"""Run SWE Agent benchmark configurations from the command line."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys

import json
from pathlib import Path
from swe_agent.benchmark.runner import run_benchmark, summarize_outcomes
from swe_agent.benchmark.reporting import format_run_summary
from swe_agent.benchmark.validate import (
    ValidationInterrupted,
    summarize_baselines,
    validate_benchmark,
)
from swe_agent.llm import provider_from_environment


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    """Parse a positive command-line number of seconds."""
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def main() -> None:
    """Parse benchmark options, run entries, and print their summary."""
    parser = argparse.ArgumentParser(description="Run an SWE Agent benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--open-pr", action="store_true", help="Create PRs for fixes.")
    parser.add_argument(
        "--provider", choices=("gemini", "qwen", "tokenharbor"),
        help="LLM provider (defaults to LLM_PROVIDER or gemini).",
    )
    parser.add_argument("--model", help="Exact provider model ID.")
    parser.add_argument("--base-url", help="OpenAI-compatible provider base URL.")
    parser.add_argument(
        "--max-input-tokens", type=_positive_int,
        help="Conservative request-input limit; required for Token Harbor.",
    )
    parser.add_argument(
        "--max-retries", type=int,
        help="Retries after temporary provider failures (default: provider configuration or 6).",
    )
    parser.add_argument(
        "--max-retry-after", type=_positive_float,
        help="Maximum Retry-After wait accepted from the provider in seconds.",
    )
    parser.add_argument(
        "--retry-errored", action="store_true",
        help="On resume, rerun infrastructure-error entries.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check each entry's pinned tests fail, without any LLM calls.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="With --validate, check only the first N entries in config order.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help=(
            "With --validate, append one JSON line per completed entry "
            "to this file."
        ),
    )
    parser.add_argument(
        "--install-timeout",
        type=_positive_float,
        default=None,
        help="Per installation command timeout in seconds.",
    )
    parser.add_argument(
        "--test-timeout",
        type=_positive_float,
        default=None,
        help="Per baseline and agent test command timeout in seconds.",
    )
    parser.add_argument(
        "--setup-deadline",
        type=_positive_float,
        default=None,
        help="Per-entry environment setup deadline in seconds.",
    )
    arguments = parser.parse_args()
    if arguments.open_pr:
        parser.error("--open-pr is not supported in benchmark mode.")
    if not arguments.validate and (
        arguments.limit is not None or arguments.results_file is not None
    ):
        parser.error("--limit and --results-file require --validate.")
    if arguments.max_retries is not None and arguments.max_retries < 0:
        parser.error("--max-retries must not be negative.")

    timeout_options = {}
    if arguments.install_timeout is not None:
        timeout_options["install_timeout_seconds"] = arguments.install_timeout
    if arguments.test_timeout is not None:
        timeout_options["test_timeout_seconds"] = arguments.test_timeout
    if arguments.setup_deadline is not None:
        timeout_options["setup_deadline_seconds"] = arguments.setup_deadline

    if arguments.validate:
        # Deliberately return before any provider is constructed: validation
        # must run with zero LLM calls and no API key.
        try:
            results = validate_benchmark(
                arguments.config,
                limit=arguments.limit,
                results_file=arguments.results_file,
                **timeout_options,
            )
        except ValidationInterrupted as interrupted:
            print(summarize_baselines(interrupted.results))
            sys.exit(130)
        except KeyboardInterrupt:
            print(summarize_baselines([]))
            sys.exit(130)
        print(summarize_baselines(results))
        if any(result.status != "fails_as_expected" for result in results):
            sys.exit(1)
        return
    try:
        provider_options = {
            key: value for key, value in {
                "provider": arguments.provider,
                "model": arguments.model,
                "base_url": arguments.base_url,
                "max_input_tokens": arguments.max_input_tokens,
                "max_retries": arguments.max_retries,
                "max_retry_after_seconds": arguments.max_retry_after,
            }.items() if value is not None
        }
        outcomes = run_benchmark(
            arguments.config,
            provider_from_environment(**provider_options),
            open_pr=arguments.open_pr,
            retry_errored=arguments.retry_errored,
            **timeout_options,
        )
    except ValueError as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        sys.exit(2)
    print(summarize_outcomes(outcomes))
    for outcome in outcomes:
        if outcome.report_path:
            print(format_run_summary(json.loads(Path(outcome.report_path).read_text())))


if __name__ == "__main__":
    main()

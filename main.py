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


def main() -> None:
    """Parse benchmark options, run entries, and print their summary."""
    parser = argparse.ArgumentParser(description="Run an SWE Agent benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--open-pr", action="store_true", help="Create PRs for fixes.")
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
    arguments = parser.parse_args()
    if not arguments.validate and (
        arguments.limit is not None or arguments.results_file is not None
    ):
        parser.error("--limit and --results-file require --validate.")
    if arguments.validate:
        # Deliberately return before any provider is constructed: validation
        # must run with zero LLM calls and no API key.
        try:
            results = validate_benchmark(
                arguments.config,
                limit=arguments.limit,
                results_file=arguments.results_file,
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
        outcomes = run_benchmark(
            arguments.config,
            provider_from_environment(),
            open_pr=arguments.open_pr,
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

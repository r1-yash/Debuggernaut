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
from swe_agent.benchmark.validate import summarize_baselines, validate_benchmark
from swe_agent.llm import provider_from_environment


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
    arguments = parser.parse_args()
    if arguments.validate:
        # Deliberately return before any provider is constructed: validation
        # must run with zero LLM calls and no API key.
        results = validate_benchmark(arguments.config)
        print(summarize_baselines(results))
        if any(result.status != "fails_as_expected" for result in results):
            sys.exit(1)
        return
    outcomes = run_benchmark(arguments.config, provider_from_environment(), open_pr=arguments.open_pr)
    print(summarize_outcomes(outcomes))
    for outcome in outcomes:
        if outcome.report_path:
            print(format_run_summary(json.loads(Path(outcome.report_path).read_text())))


if __name__ == "__main__":
    main()

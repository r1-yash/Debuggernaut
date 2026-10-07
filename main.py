"""Run SWE Agent benchmark configurations from the command line."""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse

import json
from pathlib import Path
from swe_agent.benchmark.runner import run_benchmark, summarize_outcomes
from swe_agent.benchmark.reporting import format_run_summary
from swe_agent.llm import provider_from_environment


def main() -> None:
    """Parse benchmark options, run entries, and print their summary."""
    parser = argparse.ArgumentParser(description="Run an SWE Agent benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON.")
    parser.add_argument("--open-pr", action="store_true", help="Create PRs for fixes.")
    arguments = parser.parse_args()
    outcomes = run_benchmark(arguments.config, provider_from_environment(), open_pr=arguments.open_pr)
    print(summarize_outcomes(outcomes))
    for outcome in outcomes:
        if outcome.report_path:
            print(format_run_summary(json.loads(Path(outcome.report_path).read_text())))


if __name__ == "__main__":
    main()

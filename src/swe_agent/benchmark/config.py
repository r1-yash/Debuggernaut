"""Load SWE Agent benchmark entries from JSON configuration files."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


class BenchmarkEntry(BaseModel):
    """One repository issue and test command in a benchmark run."""

    owner: str
    repository: str
    issue_number: int
    test_command: list[str]


def _parse_entry(key: object, value: object) -> BenchmarkEntry:
    """Validate and convert one benchmark configuration mapping entry."""
    if not isinstance(key, str):
        raise ValueError("Benchmark entry keys must be 'owner/repository' strings.")
    owner, separator, repository = key.partition("/")
    if not separator or not owner or not repository or "/" in repository:
        raise ValueError(f"Malformed benchmark repository key: {key!r}.")
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark entry {key!r} must be an object.")

    issue_number = value.get("issue_number")
    test_command = value.get("test_command")
    if not isinstance(issue_number, int) or isinstance(issue_number, bool):
        raise ValueError(f"Benchmark entry {key!r} must include an integer issue_number.")
    if not isinstance(test_command, list) or not all(
        isinstance(argument, str) for argument in test_command
    ):
        raise ValueError(f"Benchmark entry {key!r} must include a list[str] test_command.")
    return BenchmarkEntry(
        owner=owner,
        repository=repository,
        issue_number=issue_number,
        test_command=test_command,
    )


def load_benchmark_config(config_path: str) -> list[BenchmarkEntry]:
    """Return validated benchmark entries from an owner/repository JSON mapping."""
    if not config_path.strip():
        raise ValueError("Benchmark config path must be provided.")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            payload: Any = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read benchmark config: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must be a JSON object.")
    return [_parse_entry(key, value) for key, value in payload.items()]

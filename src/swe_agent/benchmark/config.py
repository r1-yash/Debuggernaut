"""Load SWE Agent benchmark entries from JSON configuration files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class BenchmarkEntry(BaseModel):
    """One repository issue and test command in a benchmark run."""

    owner: str
    repository: str
    issue_number: int
    commit: str
    test_command: list[str]
    test_patch: str | None = None


def _parse_entry(
    key: object, value: object, config_dir: Path
) -> BenchmarkEntry:
    """Validate and convert one benchmark configuration mapping entry."""
    if not isinstance(key, str):
        raise ValueError("Benchmark entry keys must be 'owner/repository' strings.")
    owner, separator, repository = key.partition("/")
    if not separator or not owner or not repository or "/" in repository:
        raise ValueError(f"Malformed benchmark repository key: {key!r}.")
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark entry {key!r} must be an object.")

    issue_number = value.get("issue_number")
    commit = value.get("commit")
    test_command = value.get("test_command")
    if not isinstance(issue_number, int) or isinstance(issue_number, bool):
        raise ValueError(f"Benchmark entry {key!r} must include an integer issue_number.")
    if not isinstance(test_command, list) or not all(
        isinstance(argument, str) for argument in test_command
    ):
        raise ValueError(f"Benchmark entry {key!r} must include a list[str] test_command.")
    if not isinstance(commit, str) or not _COMMIT_SHA.fullmatch(commit):
        raise ValueError(
            f"Benchmark entry {key!r} must include a pinned commit "
            "(the full 40-character hexadecimal SHA of the commit before the fix)."
        )

    test_patch = value.get("test_patch")
    resolved_test_patch = None
    if test_patch is not None:
        if not isinstance(test_patch, str) or not test_patch.strip():
            raise ValueError(
                f"Benchmark entry {key!r} test_patch must be a non-empty string "
                "path relative to the config file."
            )
        relative_path = Path(test_patch)
        if relative_path.is_absolute():
            raise ValueError(
                f"Benchmark entry {key!r} test_patch must be relative to the "
                "config file."
            )
        resolved_path = (config_dir / relative_path).resolve()
        try:
            resolved_path.relative_to(config_dir)
        except ValueError as error:
            raise ValueError(
                f"Benchmark entry {key!r} test_patch must stay within the "
                "config file's directory."
            ) from error
        if not resolved_path.is_file():
            raise ValueError(
                f"Benchmark entry {key!r} test_patch file does not exist: "
                f"{test_patch}"
            )
        resolved_test_patch = str(resolved_path)

    return BenchmarkEntry(
        owner=owner,
        repository=repository,
        issue_number=issue_number,
        commit=commit,
        test_command=test_command,
        test_patch=resolved_test_patch,
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
    config_dir = Path(config_path).resolve().parent
    return [
        _parse_entry(key, value, config_dir) for key, value in payload.items()
    ]

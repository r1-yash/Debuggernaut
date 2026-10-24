"""Tests for the committed final benchmark set."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from swe_agent.benchmark.config import load_benchmark_config


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"

sys.path.insert(0, str(BENCHMARK_DIR / "tools"))

import finalize_benchmark  # noqa: E402


EXPECTED_IDS = {
    ("andialbrecht", "sqlparse", 763),
    ("asottile", "pyupgrade", 936),
    ("childsish", "dynamic-yaml", 28),
    ("facelessuser", "pymdown-extensions", 2243),
    ("Fatal1ty", "mashumaro", 184),
    ("HBehrens", "fixup_shaper_svg", 2),
    ("iterative", "shtab", 165),
    ("pre-commit", "pre-commit-hooks", 1038),
    ("scitokens", "scitokens", 192),
    ("tobgu", "pyrsistent", 280),
    ("tobymao", "sqlglot", 3384),
    ("twisted", "towncrier", 584),
    ("yukinarit", "pyserde", 435),
}


def _git_checkout_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def test_final_benchmark_config_and_test_patches() -> None:
    entries = load_benchmark_config(str(BENCHMARK_DIR / "repos.json"))

    assert len(entries) == 13
    identifiers = {
        (entry.owner, entry.repository, entry.issue_number) for entry in entries
    }
    assert len(identifiers) == len(entries)
    assert identifiers == EXPECTED_IDS

    patch_paths: list[Path] = []
    for entry in entries:
        assert entry.test_patch is not None
        patch_path = Path(entry.test_patch)
        assert patch_path.is_file()
        assert patch_path.is_relative_to(BENCHMARK_DIR / "patches")
        assert patch_path.suffix == ".diff"
        patch_paths.append(patch_path)

        finalize_benchmark.validate_test_patch_paths(
            patch_path.read_text(encoding="utf-8"),
            f"{entry.owner}/{entry.repository}#{entry.issue_number}",
        )

    on_disk = {path.name for path in (BENCHMARK_DIR / "patches").iterdir()}
    assert on_disk == {path.name for path in patch_paths}


def test_test_patch_validator_rejects_source_changes() -> None:
    source_patch = "diff --git a/src/widget.py b/src/widget.py\n"

    with pytest.raises(ValueError, match="not recognizable test files"):
        finalize_benchmark.validate_test_patch_paths(source_patch, "fixture")


def test_test_patch_validator_requires_review_for_conftest() -> None:
    conftest_patch = "diff --git a/tests/conftest.py b/tests/conftest.py\n"

    with pytest.raises(ValueError, match="requiring explicit review"):
        finalize_benchmark.validate_test_patch_paths(conftest_patch, "fixture")


@pytest.mark.skipif(
    not _git_checkout_available(),
    reason="git is unavailable or this directory is not a checkout",
)
def test_tracked_benchmark_paths_exclude_gold_and_candidates() -> None:
    tracked_result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "--", "benchmark"],
        capture_output=True,
        text=False,
        check=True,
    )
    tracked = [
        path.decode("utf-8")
        for path in tracked_result.stdout.split(b"\0")
        if path
    ]

    assert not [path for path in tracked if "gold" in path.lower()]
    assert not [
        path for path in tracked if path.startswith("benchmark/candidates/")
    ]

    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "check-ignore",
            "-q",
            "benchmark/candidates/repos.json",
        ],
        check=False,
    )
    assert ignored.returncode == 0

#!/usr/bin/env python3
"""Create the committed benchmark set from gitignored candidate output."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_IDS_PATH = BENCHMARK_DIR / "final_ids.txt"
DEFAULT_CANDIDATES_DIR = BENCHMARK_DIR / "candidates"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR

_ID_LINE = re.compile(
    r"^(?P<repo>[^/\s#]+/[^/\s#]+)#(?P<issue>[1-9][0-9]*)$"
)
_DIFF_HEADER = re.compile(r"^diff --git (?P<old>\S+) (?P<new>\S+)$")
_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_TEST_DIRECTORIES = frozenset({"test", "tests", "testing"})
_REVIEW_DIRECTORIES = frozenset({"fixture", "fixtures"})


@dataclass(frozen=True)
class FinalId:
    """One requested repository issue in the final benchmark."""

    repo: str
    issue_number: int

    @property
    def identifier(self) -> str:
        return f"{self.repo}#{self.issue_number}"


def load_final_ids(path: str | Path) -> list[FinalId]:
    """Read and validate the checked-in final identifier list."""
    ids_path = Path(path)
    try:
        lines = ids_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read final ids from {ids_path}: {error}") from error

    ids: list[FinalId] = []
    seen_ids: dict[FinalId, int] = {}
    seen_repos: dict[str, int] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ID_LINE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{ids_path}:{line_number}: malformed final id {line!r}; "
                "expected owner/repository#issue"
            )
        final_id = FinalId(
            repo=match.group("repo"),
            issue_number=int(match.group("issue")),
        )
        if final_id in seen_ids:
            raise ValueError(
                f"{ids_path}:{line_number}: duplicate final id {line!r} "
                f"(first seen on line {seen_ids[final_id]})"
            )
        if final_id.repo in seen_repos:
            raise ValueError(
                f"{ids_path}:{line_number}: duplicate repository {final_id.repo!r} "
                f"(first seen on line {seen_repos[final_id.repo]}); the config "
                "is keyed by repository, so one issue per repository is required"
            )
        seen_ids[final_id] = line_number
        seen_repos[final_id.repo] = line_number
        ids.append(final_id)

    if not ids:
        raise ValueError(f"No final benchmark ids found in {ids_path}")
    return ids


def parse_test_patch_paths(patch_text: str, label: str) -> tuple[str, ...]:
    """Return the old and new repository paths named by ``diff --git`` headers."""
    paths: list[str] = []
    for line_number, line in enumerate(patch_text.splitlines(), start=1):
        if not line.startswith("diff --git "):
            continue
        match = _DIFF_HEADER.fullmatch(line)
        if match is None:
            raise ValueError(
                f"{label}:{line_number}: malformed diff --git header: {line!r}"
            )
        for prefix, raw_path in (("a/", match.group("old")), ("b/", match.group("new"))):
            if not raw_path.startswith(prefix):
                raise ValueError(
                    f"{label}:{line_number}: expected an {prefix!r} path, "
                    f"got {raw_path!r}"
                )
            repository_path = raw_path[len(prefix) :]
            parts = repository_path.split("/")
            if (
                not repository_path
                or repository_path.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
            ):
                raise ValueError(
                    f"{label}:{line_number}: unsafe patch path "
                    f"{repository_path!r}"
                )
            paths.append(repository_path)

    if not paths:
        raise ValueError(f"{label}: patch contains no diff --git headers")
    return tuple(dict.fromkeys(paths))


def _is_test_path(path: str) -> bool:
    """Return whether a path is a test module, helper, or lives under a test dir."""
    parts = path.split("/")
    if any(part.lower() in _TEST_DIRECTORIES for part in parts[:-1]):
        return True
    basename = parts[-1].lower()
    return (
        basename in {"test.py", "tests.py"}
        or basename.startswith("test_")
        or basename.endswith(("_test.py", "_tests.py"))
    )


def _requires_review(path: str) -> bool:
    """Return whether a test-tree path still needs explicit human review."""
    parts = path.split("/")
    basename = parts[-1].lower()
    if basename in {"conftest.py", "fixture.py", "fixtures.py"}:
        return True
    return any(part.lower() in _REVIEW_DIRECTORIES for part in parts[:-1])


def validate_test_patch_paths(
    patch_text: str, label: str
) -> tuple[str, ...]:
    """Reject patches that touch anything other than test files."""
    paths = parse_test_patch_paths(patch_text, label)
    unexpected = [path for path in paths if not _is_test_path(path)]
    review = [
        path for path in paths if _is_test_path(path) and _requires_review(path)
    ]
    if not unexpected and not review:
        return paths

    sections: list[str] = []
    if unexpected:
        sections.append(
            "not recognizable test files:\n  - " + "\n  - ".join(unexpected)
        )
    if review:
        sections.append(
            "fixture/conftest paths requiring explicit review:\n  - "
            + "\n  - ".join(review)
        )
    raise ValueError(
        f"{label} touches paths that require attention:\n"
        + "\n".join(sections)
        + "\nReview these paths explicitly before finalizing the benchmark."
    )


def _load_candidates(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read candidate config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate config {path} must be a JSON object")
    return payload


def _build_outputs(
    ids: list[FinalId],
    candidates_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes]]:
    candidates = _load_candidates(candidates_path)
    candidates_dir = candidates_path.parent
    missing: list[str] = []
    for final_id in ids:
        candidate = candidates.get(final_id.repo)
        if not isinstance(candidate, dict):
            missing.append(final_id.identifier)
            continue
        if candidate.get("issue_number") != final_id.issue_number:
            missing.append(
                f"{final_id.identifier} "
                f"(candidate issue is {candidate.get('issue_number')!r})"
            )
    if missing:
        joined = "\n  - ".join(missing)
        raise ValueError(
            "Missing final benchmark ids in candidate config:\n"
            f"  - {joined}"
        )

    entries: dict[str, dict[str, Any]] = {}
    patch_files: dict[str, bytes] = {}
    for final_id in sorted(ids, key=lambda item: (item.repo.casefold(), item.repo)):
        candidate = candidates[final_id.repo]
        issue_number = candidate.get("issue_number")
        if not isinstance(issue_number, int) or isinstance(issue_number, bool):
            raise ValueError(
                f"{final_id.identifier}: candidate issue_number must be an integer"
            )
        commit = candidate.get("commit")
        if not isinstance(commit, str) or _COMMIT_SHA.fullmatch(commit) is None:
            raise ValueError(
                f"{final_id.identifier}: candidate commit must be a full 40-character SHA"
            )
        test_command = candidate.get("test_command")
        if not isinstance(test_command, list) or not all(
            isinstance(argument, str) for argument in test_command
        ):
            raise ValueError(
                f"{final_id.identifier}: candidate test_command must be a list[str]"
            )
        test_patch = candidate.get("test_patch")
        if not isinstance(test_patch, str) or not test_patch.strip():
            raise ValueError(
                f"{final_id.identifier}: candidate test_patch must be a non-empty string"
            )

        relative_patch = Path(test_patch)
        if (
            relative_patch.is_absolute()
            or len(relative_patch.parts) != 2
            or relative_patch.parts[0] != "patches"
        ):
            raise ValueError(
                f"{final_id.identifier}: candidate test_patch must be "
                f"'patches/<name>.diff', got {test_patch!r}"
            )
        if "gold" in test_patch.lower():
            raise ValueError(
                f"{final_id.identifier}: candidate test_patch must not reference gold: "
                f"{test_patch!r}"
            )

        source_patch = (candidates_dir / relative_patch).resolve()
        try:
            source_patch.relative_to((candidates_dir / "patches").resolve())
        except ValueError as error:
            raise ValueError(
                f"{final_id.identifier}: candidate test_patch escapes patches/: "
                f"{test_patch!r}"
            ) from error
        if source_patch.suffix != ".diff" or not source_patch.is_file():
            raise ValueError(
                f"{final_id.identifier}: candidate test patch does not exist: "
                f"{test_patch!r}"
            )

        try:
            patch_bytes = source_patch.read_bytes()
            patch_text = patch_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(
                f"{final_id.identifier}: could not read candidate test patch "
                f"{test_patch!r}: {error}"
            ) from error
        if not patch_text.strip():
            raise ValueError(
                f"{final_id.identifier}: candidate test patch is empty: "
                f"{test_patch!r}"
            )
        validate_test_patch_paths(patch_text, final_id.identifier)

        patch_name = source_patch.name
        if patch_name in patch_files:
            raise ValueError(
                f"{final_id.identifier}: duplicate output patch filename {patch_name!r}"
            )
        patch_files[patch_name] = patch_bytes
        entries[final_id.repo] = {
            "issue_number": issue_number,
            "commit": commit,
            "test_command": list(test_command),
            "test_patch": f"patches/{patch_name}",
        }

    requested_repos = {final_id.repo for final_id in ids}
    if len(entries) != len(ids) or set(entries) != requested_repos:
        raise ValueError(
            "Internal error: generated entries do not exactly match final ids"
        )
    if len(patch_files) != len(ids):
        raise ValueError(
            "Internal error: generated patch files do not exactly match final ids"
        )
    return entries, patch_files


def _write_outputs(
    payload: dict[str, dict[str, Any]],
    patch_files: dict[str, bytes],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = output_dir / "patches"
    if patch_dir.exists():
        if not patch_dir.is_dir():
            raise ValueError(f"Output patch path is not a directory: {patch_dir}")
        existing = {entry.name for entry in patch_dir.iterdir()}
        unexpected = sorted(existing - set(patch_files))
        if unexpected:
            joined = "\n  - ".join(unexpected)
            raise ValueError(
                f"Refusing to write unexpected files in {patch_dir}:\n"
                f"  - {joined}"
            )
    else:
        patch_dir.mkdir()

    for patch_name in sorted(patch_files):
        target = patch_dir / patch_name
        if target.is_dir():
            raise ValueError(f"Output patch path is a directory: {target}")
        target.write_bytes(patch_files[patch_name])

    config_path = output_dir / "repos.json"
    temporary_config = config_path.with_name(f".{config_path.name}.tmp")
    temporary_config.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_config.replace(config_path)


def finalize_benchmark(
    ids_path: str | Path = DEFAULT_IDS_PATH,
    candidates_dir: str | Path = DEFAULT_CANDIDATES_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[str]:
    """Write the committed config and test patches for the requested ids."""
    ids = load_final_ids(ids_path)
    candidates_path = Path(candidates_dir) / "repos.json"
    payload, patch_files = _build_outputs(ids, candidates_path)
    _write_outputs(payload, patch_files, Path(output_dir))
    return list(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build benchmark/repos.json and benchmark/patches/ from the "
            "gitignored candidate output."
        )
    )
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS_PATH)
    parser.add_argument(
        "--candidates", type=Path, default=DEFAULT_CANDIDATES_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)

    try:
        repositories = finalize_benchmark(
            arguments.ids,
            arguments.candidates,
            arguments.output,
        )
    except ValueError as error:
        parser.exit(1, f"error: {error}\n")

    print(
        f"Wrote {len(repositories)} entries to "
        f"{arguments.output / 'repos.json'} and "
        f"{len(repositories)} test patches to {arguments.output / 'patches'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

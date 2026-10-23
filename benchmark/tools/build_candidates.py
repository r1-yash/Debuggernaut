"""Build benchmark candidates from the public SWE-bench-extra dataset.

This module is intentionally standalone. Nothing under ``src/swe_agent`` imports
it, and it never needs an LLM provider or a cloned target repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests


DATASET_ID = "nebius/SWE-bench-extra"
DATASET_REVISION = "11dcbfb30e19552df2a2f8030bd764adc95c92a5"
DATASET_PARQUET = "data/train-00000-of-00001.parquet"
DATASET_SHA256 = (
    "c8633563d41c1a2fde0361650f96b5103b11f74d7feac5c9abf33f58e486d946"
)
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
    f"{DATASET_REVISION}/{DATASET_PARQUET}"
)

REQUIRED_COLUMNS = (
    "instance_id",
    "patch",
    "repo",
    "base_commit",
    "test_patch",
    "problem_statement",
    "FAIL_TO_PASS",
    "created_at",
)
PYTEST_COMMAND = ("python", "-m", "pytest")
TEST_COMMAND_MODE = Literal["files", "node-ids"]

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_PR_NUMBER = re.compile(r"-(\d+)$")
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
_TEST_DIRECTORIES = frozenset({"tests", "test", "testing"})
_DOCUMENTATION_DIRECTORIES = frozenset(
    {"docs", "doc", "documentation", "ci", ".github", ".circleci", ".gitlab"}
)
_CHANGELOG_NAMES = frozenset({"changelog", "changes", "history", "news"})


@dataclass(frozen=True)
class FileChange:
    """One file touched by a unified diff."""

    old_path: str
    path: str
    status: Literal["modified", "added", "deleted", "renamed"]
    additions: int
    deletions: int
    hunks: int


@dataclass(frozen=True)
class FilterResult:
    """A dataset row after candidate-shape checks."""

    row: Mapping[str, Any]
    accepted: bool
    reason: str | None
    pr_number: int | None
    fix_file: str | None
    test_files: tuple[str, ...]
    test_command: tuple[str, ...] | None
    fix_change: FileChange | None


def parse_diff(text: str) -> list[FileChange]:
    """Parse a unified diff into per-file changes without applying it."""
    changes: list[FileChange] = []
    old_path = ""
    path = ""
    status: Literal["modified", "added", "deleted", "renamed"] = "modified"
    additions = 0
    deletions = 0
    hunks = 0
    in_file = False

    def flush() -> None:
        nonlocal in_file
        if not in_file:
            return
        changes.append(
            FileChange(
                old_path=old_path or path,
                path=path,
                status=status,
                additions=additions,
                deletions=deletions,
                hunks=hunks,
            )
        )
        in_file = False

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush()
            match = _DIFF_HEADER.fullmatch(line)
            if match is None:
                raise ValueError(f"Malformed diff header: {line!r}")
            old_path, path = match.groups()
            status = "modified"
            additions = 0
            deletions = 0
            hunks = 0
            in_file = True
            continue
        if not in_file:
            continue
        if line.startswith("new file mode"):
            status = "added"
        elif line.startswith("deleted file mode"):
            status = "deleted"
        elif line.startswith("rename from "):
            old_path = line[len("rename from ") :]
            status = "renamed"
        elif line.startswith("rename to "):
            path = line[len("rename to ") :]
        elif line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    flush()
    return changes


def is_python_test_path(path: str) -> bool:
    """Return whether a path is a Python test file, including test helpers."""
    if not path.endswith(".py"):
        return False
    parts = path.split("/")
    basename = parts[-1].lower()
    if basename in {"conftest.py", "test.py", "tests.py"}:
        return True
    if basename.startswith("test_") or basename.endswith(("_test.py", "_tests.py")):
        return True
    return any(part.lower() in _TEST_DIRECTORIES for part in parts[:-1])


def is_test_module_path(path: str) -> bool:
    """Return whether a path is named like a pytest module, not a test helper."""
    if not path.endswith(".py"):
        return False
    basename = path.rsplit("/", 1)[-1].lower()
    return (
        basename in {"test.py", "tests.py"}
        or basename.startswith("test_")
        or basename.endswith(("_test.py", "_tests.py"))
    )


def is_doc_or_ci_path(path: str) -> bool:
    """Return whether a path is documentation, changelog, or CI configuration."""
    parts = [part.lower() for part in path.split("/")]
    basename = parts[-1]
    if any(part in _DOCUMENTATION_DIRECTORIES for part in parts[:-1]):
        return True
    return basename in _CHANGELOG_NAMES or basename.startswith(
        ("changelog.", "changes.", "history.")
    )


def parse_pr_number(instance_id: str) -> int:
    """Extract the PR number from an SWE-bench instance identifier."""
    match = _PR_NUMBER.search(instance_id)
    if match is None:
        raise ValueError(f"Instance ID has no PR number suffix: {instance_id!r}")
    return int(match.group(1))


def fail_to_pass_files(fail_to_pass: Sequence[object]) -> tuple[str, ...]:
    """Extract valid repo-relative test files from pytest node IDs."""
    files: set[str] = set()
    for node_id in fail_to_pass:
        if not isinstance(node_id, str):
            continue
        file_path = node_id.split("::", 1)[0].strip()
        if file_path.endswith(".py") and _is_safe_repo_path(file_path):
            files.add(file_path)
    return tuple(sorted(files))


def _is_safe_repo_path(path: str) -> bool:
    """Return whether a path is relative and stays inside the repository."""
    if not path or path.startswith(("/", "\\")):
        return False
    parts = path.replace("\\", "/").split("/")
    return ".." not in parts and "." not in parts


def _require_text(row: Mapping[str, Any], key: str) -> str:
    """Return a non-empty string field or raise a filter-oriented error."""
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_field:{key}")
    return value


def _source_changes(
    changes: Sequence[FileChange],
) -> list[FileChange]:
    """Return non-test Python source changes, excluding docs and CI."""
    return [
        change
        for change in changes
        if change.path.endswith(".py")
        and not is_python_test_path(change.path)
        and not is_doc_or_ci_path(change.path)
    ]


def derive_test_files(
    row: Mapping[str, Any],
) -> tuple[str, ...]:
    """Choose test files from the test patch, preferring failing-test paths."""
    test_changes = parse_diff(_require_text(row, "test_patch"))
    patched_modules = {
        change.path
        for change in test_changes
        if change.status != "deleted" and is_test_module_path(change.path)
    }
    failing_files = set(
        fail_to_pass_files(row.get("FAIL_TO_PASS", ()))
    )
    intersection = sorted(patched_modules & failing_files)
    if intersection:
        return tuple(intersection)
    if patched_modules:
        return tuple(sorted(patched_modules))
    if failing_files:
        return tuple(sorted(failing_files))
    raise ValueError("no_test_command")


def derive_test_command(
    row: Mapping[str, Any],
    mode: TEST_COMMAND_MODE = "files",
) -> tuple[str, ...]:
    """Build a pytest command from test-patch files or failing node IDs."""
    if mode == "files":
        return (*PYTEST_COMMAND, *derive_test_files(row))
    if mode == "node-ids":
        node_ids = sorted(
            {
                node_id.strip()
                for node_id in row.get("FAIL_TO_PASS", ())
                if isinstance(node_id, str) and node_id.strip()
            }
        )
        if not node_ids:
            raise ValueError("no_test_command")
        return (*PYTEST_COMMAND, *node_ids)
    raise ValueError(f"Unknown test command mode: {mode!r}")


def filter_row(
    row: Mapping[str, Any],
    *,
    test_command_mode: TEST_COMMAND_MODE = "files",
) -> FilterResult:
    """Check one dataset row against the candidate-shape filters."""
    try:
        instance_id = _require_text(row, "instance_id")
        repo = _require_text(row, "repo")
        commit = _require_text(row, "base_commit")
        _require_text(row, "created_at")
        patch_text = _require_text(row, "patch")
        _require_text(row, "test_patch")
    except ValueError as error:
        return FilterResult(
            row=row,
            accepted=False,
            reason=str(error),
            pr_number=None,
            fix_file=None,
            test_files=(),
            test_command=None,
            fix_change=None,
        )

    owner, separator, repository = repo.partition("/")
    if not separator or not owner or not repository or "/" in repository:
        return _rejected(row, "invalid_repo")
    if not _COMMIT_SHA.fullmatch(commit):
        return _rejected(row, "invalid_base_commit")

    try:
        pr_number = parse_pr_number(instance_id)
        patch_changes = parse_diff(patch_text)
        test_changes = parse_diff(_require_text(row, "test_patch"))
    except ValueError as error:
        return _rejected(row, str(error))

    patch_test_changes = [
        change for change in patch_changes if is_python_test_path(change.path)
    ]
    if patch_test_changes:
        return _rejected(row, "fix_patch_touches_test_file", pr_number)

    non_deleted_test_changes = [
        change
        for change in test_changes
        if change.status != "deleted" and is_python_test_path(change.path)
    ]
    if not non_deleted_test_changes:
        return _rejected(row, "test_patch_has_no_test_file", pr_number)

    source_changes = _source_changes(patch_changes)
    if len(source_changes) != 1:
        return _rejected(
            row, f"fix_patch_source_file_count:{len(source_changes)}", pr_number
        )
    fix_change = source_changes[0]
    if fix_change.status != "modified" or not _is_safe_repo_path(fix_change.path):
        return _rejected(
            row, f"fix_file_not_existing:{fix_change.status}", pr_number
        )

    try:
        test_files = derive_test_files(row)
        test_command = derive_test_command(row, test_command_mode)
    except ValueError as error:
        return _rejected(row, str(error), pr_number)

    if not all(_is_safe_repo_path(path) for path in test_files):
        return _rejected(row, "unsafe_test_path", pr_number)

    return FilterResult(
        row=row,
        accepted=True,
        reason=None,
        pr_number=pr_number,
        fix_file=fix_change.path,
        test_files=test_files,
        test_command=test_command,
        fix_change=fix_change,
    )


def _rejected(
    row: Mapping[str, Any],
    reason: str,
    pr_number: int | None = None,
) -> FilterResult:
    """Return a rejected filter result with empty derived fields."""
    return FilterResult(
        row=row,
        accepted=False,
        reason=reason,
        pr_number=pr_number,
        fix_file=None,
        test_files=(),
        test_command=None,
        fix_change=None,
    )


def load_dataset_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load normalized dataset rows from a JSONL cache."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as rows_file:
        for line_number, line in enumerate(rows_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid dataset row at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"Dataset row at line {line_number} is not an object."
                )
            missing_columns = [
                column for column in REQUIRED_COLUMNS if column not in row
            ]
            if missing_columns:
                raise ValueError(
                    f"Dataset row at line {line_number} is missing columns: "
                    f"{', '.join(missing_columns)}"
                )
            rows.append(row)
    if not rows:
        raise ValueError("Dataset cache contains no rows.")
    return rows


def default_cache_dir() -> Path:
    """Return the default cache location for the pinned dataset revision."""
    override = os.environ.get("DEBUGGERNAUT_CACHE_DIR")
    root = (
        Path(override)
        if override
        else Path.home() / ".cache" / "debuggernaut" / "swebench-extra"
    )
    return root / DATASET_REVISION


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_cache(
    cache_dir: str | Path,
    *,
    offline: bool = False,
    refresh: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Return the normalized JSONL cache, downloading it when allowed."""
    cache_path = Path(cache_dir)
    rows_path = cache_path / "rows.jsonl"
    parquet_path = cache_path / "train.parquet"
    metadata_path = cache_path / "dataset.json"

    if rows_path.is_file() and not refresh:
        return rows_path
    if offline:
        raise ValueError(
            "Offline mode requires an existing dataset cache: "
            f"{rows_path}"
        )

    cache_path.mkdir(parents=True, exist_ok=True)
    if refresh or not parquet_path.is_file():
        _download_dataset(parquet_path, session=session)
    if sha256_file(parquet_path) != DATASET_SHA256:
        raise ValueError(
            "Dataset checksum mismatch for cached parquet: "
            f"{parquet_path}"
        )

    _convert_parquet(parquet_path, rows_path)
    with rows_path.open(encoding="utf-8") as rows_file:
        row_count = sum(1 for line in rows_file if line.strip())
    metadata = {
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "parquet": DATASET_PARQUET,
        "sha256": DATASET_SHA256,
        "rows": row_count,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows_path


def _download_dataset(
    destination: Path,
    *,
    session: requests.Session | None = None,
) -> None:
    """Download the pinned parquet to a temporary file, then replace."""
    client = session or requests.Session()
    temporary = destination.with_suffix(".parquet.tmp")
    try:
        with client.get(
            DATASET_URL, stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            with open(temporary, "wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        target.write(chunk)
        if sha256_file(temporary) != DATASET_SHA256:
            raise ValueError("Downloaded dataset checksum did not match.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _convert_parquet(parquet_path: Path, rows_path: Path) -> None:
    """Convert the parquet table to deterministic JSONL, then replace."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to convert the dataset; install "
            "benchmark/tools/requirements.txt."
        ) from error

    rows = parquet.read_table(parquet_path).to_pylist()
    temporary = rows_path.with_suffix(".jsonl.tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, rows_path)
    finally:
        temporary.unlink(missing_ok=True)

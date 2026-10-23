"""Build benchmark candidates from the public SWE-bench-extra dataset.

This module is intentionally standalone. Nothing under ``src/swe_agent`` imports
it, and it never needs an LLM provider or a cloned target repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
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
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "candidates"

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


@dataclass(frozen=True)
class Candidate:
    """One resolved, selected benchmark candidate."""

    row: Mapping[str, Any]
    issue_number: int
    pr_number: int
    repo: str
    commit: str
    created_at: str
    fix_file: str
    test_files: tuple[str, ...]
    test_command: tuple[str, ...]
    fix_change: FileChange
    test_patch_path: str
    gold_patch_path: str


@dataclass(frozen=True)
class UnresolvedCandidate:
    """A filtered row that could not be selected without an issue number."""

    repo: str
    pr_number: int
    instance_id: str


@dataclass(frozen=True)
class SelectionResult:
    """Deterministic candidate selection and the counts used to explain it."""

    candidates: tuple[Candidate, ...]
    available: tuple[Candidate, ...]
    total_rows: int
    filtered_rows: int
    filtered_repositories: int
    unresolved_rows: tuple[UnresolvedCandidate, ...]
    unresolved_repositories: int
    rejected_reasons: Mapping[str, int]
    limit: int


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


def derive_test_command(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Build a pytest command from test-patch files."""
    return (*PYTEST_COMMAND, *derive_test_files(row))


def filter_row(row: Mapping[str, Any]) -> FilterResult:
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
        test_command = derive_test_command(row)
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


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    issue_numbers: Mapping[tuple[str, int], int],
    limit: int,
) -> SelectionResult:
    """Filter rows, resolve one issue per repo, and rank newest first."""
    if limit < 1:
        raise ValueError("Selection limit must be positive.")
    row_list = list(rows)
    rejected: Counter[str] = Counter()
    filtered: list[FilterResult] = []
    for row in row_list:
        result = filter_row(row)
        if result.accepted:
            filtered.append(result)
        else:
            rejected[result.reason or "unknown"] += 1

    grouped: dict[str, list[FilterResult]] = defaultdict(list)
    for result in filtered:
        grouped[str(result.row["repo"])].append(result)

    unresolved: list[UnresolvedCandidate] = []
    resolved: list[Candidate] = []
    resolved_repositories: set[str] = set()
    for repo in sorted(grouped):
        repo_results = sorted(
            grouped[repo],
            key=lambda item: (
                str(item.row["created_at"]),
                str(item.row["instance_id"]),
            ),
            reverse=True,
        )
        for result in repo_results:
            if result.pr_number is None:
                raise ValueError("Accepted row has no pull request number.")
            issue_number = issue_numbers.get((repo, result.pr_number))
            if issue_number is None:
                unresolved.append(
                    UnresolvedCandidate(
                        repo=repo,
                        pr_number=result.pr_number,
                        instance_id=str(result.row["instance_id"]),
                    )
                )
                continue
            _validate_issue_number(issue_number)
            resolved.append(_build_candidate(result, issue_number))
            resolved_repositories.add(repo)
            break

    resolved.sort(key=lambda candidate: candidate.repo)
    resolved.sort(key=lambda candidate: candidate.created_at, reverse=True)
    return SelectionResult(
        candidates=tuple(resolved[:limit]),
        available=tuple(resolved),
        total_rows=len(row_list),
        filtered_rows=len(filtered),
        filtered_repositories=len(grouped),
        unresolved_rows=tuple(
            sorted(unresolved, key=lambda item: (item.repo, item.pr_number))
        ),
        unresolved_repositories=(
            len(grouped) - len(resolved_repositories)
        ),
        rejected_reasons=dict(sorted(rejected.items())),
        limit=limit,
    )


def _build_candidate(
    result: FilterResult, issue_number: int
) -> Candidate:
    """Convert one accepted filter result into a resolved candidate."""
    if (
        not result.accepted
        or result.pr_number is None
        or result.fix_file is None
        or result.test_command is None
        or result.fix_change is None
    ):
        raise ValueError("Cannot build a candidate from a rejected row.")
    row = result.row
    repo = str(row["repo"])
    filename = candidate_filename(repo, issue_number)
    return Candidate(
        row=row,
        issue_number=issue_number,
        pr_number=result.pr_number,
        repo=repo,
        commit=str(row["base_commit"]),
        created_at=str(row["created_at"]),
        fix_file=result.fix_file,
        test_files=result.test_files,
        test_command=result.test_command,
        fix_change=result.fix_change,
        test_patch_path=f"patches/{filename}",
        gold_patch_path=f"gold/{filename}",
    )


def _validate_issue_number(value: object) -> None:
    """Reject missing or non-positive issue numbers."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("Issue numbers must be positive integers.")


def candidate_filename(repo: str, issue_number: int) -> str:
    """Return the deterministic filename stem for a candidate's patches."""
    owner, separator, repository = repo.partition("/")
    if not separator or not owner or not repository or "/" in repository:
        raise ValueError(f"Malformed repository name: {repo!r}")
    _validate_issue_number(issue_number)
    return f"{owner}__{repository}__{issue_number}.diff"


def load_issue_map(path: str | Path) -> dict[tuple[str, int], int]:
    """Load a repo/PR to issue-number mapping for selection."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read issue map: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Issue map must be a JSON object.")

    issue_numbers: dict[tuple[str, int], int] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("Issue map keys must be 'owner/repo#pr' strings.")
        repo, separator, pr_text = key.rpartition("#")
        owner, slash, repository = repo.partition("/")
        if (
            not separator
            or not slash
            or not owner
            or not repository
            or "/" in repository
            or not pr_text.isdigit()
            or int(pr_text) <= 0
        ):
            raise ValueError(f"Malformed issue map key: {key!r}")
        _validate_issue_number(value)
        issue_numbers[(repo, int(pr_text))] = value
    return issue_numbers


def write_candidate_outputs(
    selection: SelectionResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write config, test/gold patches, and reports atomically."""
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path is not a directory: {output_path}")
    if (
        output_path.exists()
        and any(output_path.iterdir())
        and not overwrite
    ):
        raise ValueError(
            f"Output directory is not empty: {output_path}; use --force."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / f".{output_path.name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        (staging / "patches").mkdir()
        (staging / "gold").mkdir()
        repos: dict[str, dict[str, Any]] = {}
        for candidate in selection.candidates:
            test_patch = _require_text(candidate.row, "test_patch")
            gold_patch = _require_text(candidate.row, "patch")
            (staging / candidate.test_patch_path).write_bytes(
                test_patch.encode("utf-8")
            )
            (staging / candidate.gold_patch_path).write_bytes(
                gold_patch.encode("utf-8")
            )
            repos[candidate.repo] = {
                "issue_number": candidate.issue_number,
                "commit": candidate.commit,
                "test_command": list(candidate.test_command),
                "test_patch": candidate.test_patch_path,
            }

        _write_json(staging / "repos.json", repos)
        _write_json(
            staging / "candidates_report.json",
            _report_payload(selection),
        )
        (staging / "candidates_report.md").write_text(
            _render_markdown(selection),
            encoding="utf-8",
        )
        if output_path.exists():
            shutil.rmtree(output_path)
        os.replace(staging, output_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_json(path: Path, payload: object) -> None:
    """Write deterministic UTF-8 JSON."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report_payload(selection: SelectionResult) -> dict[str, Any]:
    """Build the machine-readable candidate report."""
    return {
        "dataset": {
            "name": DATASET_ID,
            "revision": DATASET_REVISION,
        },
        "selection": {
            "limit": selection.limit,
            "total_rows": selection.total_rows,
            "filtered_rows": selection.filtered_rows,
            "filtered_repositories": selection.filtered_repositories,
            "available_candidates": len(selection.available),
            "selected_candidates": len(selection.candidates),
            "unresolved_rows": len(selection.unresolved_rows),
            "unresolved_repositories": selection.unresolved_repositories,
            "ordering": (
                "created_at descending, then repo ascending; "
                "one resolved candidate per repo"
            ),
            "cutoff_created_at": (
                selection.candidates[-1].created_at
                if selection.candidates
                else None
            ),
        },
        "filter_rejections": dict(selection.rejected_reasons),
        "unresolved": [
            {
                "repo": item.repo,
                "pr_number": item.pr_number,
                "instance_id": item.instance_id,
            }
            for item in selection.unresolved_rows
        ],
        "candidates": [
            _report_candidate(candidate) for candidate in selection.candidates
        ],
    }


def _report_candidate(candidate: Candidate) -> dict[str, Any]:
    """Build one candidate report entry."""
    meta = candidate.row.get("meta")
    is_lite = meta.get("is_lite") if isinstance(meta, dict) else None
    license_value = candidate.row.get("license")
    return {
        "repo": candidate.repo,
        "instance_id": str(candidate.row["instance_id"]),
        "pr_number": candidate.pr_number,
        "issue_number": candidate.issue_number,
        "commit": candidate.commit,
        "created_at": candidate.created_at,
        "fix_file": candidate.fix_file,
        "test_files": list(candidate.test_files),
        "test_command": list(candidate.test_command),
        "test_patch": candidate.test_patch_path,
        "gold_patch": candidate.gold_patch_path,
        "filters_passed": _filters_passed(),
        "fix_patch": {
            "additions": candidate.fix_change.additions,
            "deletions": candidate.fix_change.deletions,
            "hunks": candidate.fix_change.hunks,
        },
        "is_lite": is_lite if isinstance(is_lite, bool) else None,
        "license": license_value if isinstance(license_value, str) else None,
    }


def _filters_passed() -> list[str]:
    """Return the stable list of checks applied to every selected row."""
    return [
        "test_patch_nonempty",
        "test_patch_touches_python_test",
        "exactly_one_non_test_python_file",
        "fix_file_exists_at_base",
        "pytest_file_command",
        "issue_resolved",
        "newest_resolved_for_repo",
        "one_candidate_per_repo",
    ]


def _render_markdown(selection: SelectionResult) -> str:
    """Render a human-readable candidate report."""
    lines = [
        "# SWE-bench-extra candidates",
        "",
        f"- Dataset: `{DATASET_ID}` at `{DATASET_REVISION}`",
        f"- Rows read: {selection.total_rows}",
        f"- Rows passing shape filters: {selection.filtered_rows}",
        f"- Repositories passing shape filters: {selection.filtered_repositories}",
        f"- Resolved candidates available: {len(selection.available)}",
        f"- Candidates selected: {len(selection.candidates)} (limit {selection.limit})",
        f"- Unresolved rows: {len(selection.unresolved_rows)}",
        f"- Repositories dropped without an issue: {selection.unresolved_repositories}",
    ]
    if selection.rejected_reasons:
        lines.extend(
            [
                "",
                "## Filter rejections",
                "",
                "| Reason | Rows |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            f"| {_markdown_cell(reason)} | {count} |"
            for reason, count in selection.rejected_reasons.items()
        )

    lines.extend(["", "## Candidates", ""])
    if not selection.candidates:
        lines.append("No resolved candidates.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Repo | Issue | PR | Fix file | Test files | Gold patch | "
            "Created | Checks |",
            "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    checks = ", ".join(_filters_passed())
    for candidate in selection.candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(candidate.repo),
                    str(candidate.issue_number),
                    str(candidate.pr_number),
                    _markdown_cell(candidate.fix_file),
                    _markdown_cell("<br>".join(candidate.test_files)),
                    _markdown_cell(candidate.gold_patch_path),
                    _markdown_cell(candidate.created_at),
                    _markdown_cell(checks),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    """Escape a value for a single Markdown table cell."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


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


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def print_dry_run(
    selection: SelectionResult,
    *,
    sample_size: int = 20,
) -> None:
    """Print stage counts and a candidate sample without writing files."""
    print(f"rows: {selection.total_rows}")
    print(f"filtered rows: {selection.filtered_rows}")
    print(f"filtered repositories: {selection.filtered_repositories}")
    print(f"resolved candidates: {len(selection.available)}")
    print(f"selected candidates: {len(selection.candidates)}")
    print(f"limit: {selection.limit}")
    print(f"unresolved rows: {len(selection.unresolved_rows)}")
    print(f"unresolved repositories: {selection.unresolved_repositories}")
    print("filter rejections:")
    if selection.rejected_reasons:
        for reason, count in selection.rejected_reasons.items():
            print(f"  {reason}: {count}")
    else:
        print("  (none)")
    print("candidate sample:")
    if not selection.candidates:
        print("  (none; provide --issue-map for resolved candidates)")
        return
    for candidate in selection.candidates[:sample_size]:
        print(
            f"  {candidate.repo}#{candidate.issue_number} "
            f"(PR {candidate.pr_number}) {candidate.fix_file} "
            f"[{candidate.created_at}]"
        )
        print(f"    tests: {', '.join(candidate.test_files)}")
    if selection.unresolved_rows:
        print("unresolved sample:")
        for item in selection.unresolved_rows[:5]:
            print(f"  {item.repo} PR {item.pr_number} ({item.instance_id})")


def main(argv: Sequence[str] | None = None) -> int:
    """Run candidate selection and optional output generation."""
    parser = argparse.ArgumentParser(
        description="Build SWE-bench-extra benchmark candidates."
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        help="Read normalized rows from JSONL instead of the pinned cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Dataset cache directory (defaults to the user cache).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never download the dataset; require an existing cache.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload and reconvert the pinned dataset.",
    )
    parser.add_argument(
        "--issue-map",
        type=Path,
        help="JSON mapping with keys like 'owner/repo#123' and issue numbers.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=150,
        help="Maximum number of candidates to select (default: 150).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for config, patches, gold patches, and reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection counts and a sample without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a non-empty output directory.",
    )
    args = parser.parse_args(argv)

    if args.dataset_jsonl:
        rows = load_dataset_rows(args.dataset_jsonl)
    else:
        rows_path = ensure_dataset_cache(
            args.cache_dir or default_cache_dir(),
            offline=args.offline,
            refresh=args.refresh,
        )
        rows = load_dataset_rows(rows_path)

    issue_numbers = (
        load_issue_map(args.issue_map) if args.issue_map else {}
    )
    if not args.dry_run and not issue_numbers:
        parser.error(
            "--issue-map is required for writing until issue resolution "
            "is added."
        )

    selection = select_candidates(
        rows,
        issue_numbers=issue_numbers,
        limit=args.limit,
    )
    if args.dry_run:
        print_dry_run(selection)
        return 0

    write_candidate_outputs(
        selection,
        args.output,
        overwrite=args.force,
    )
    print(
        f"Selected {len(selection.candidates)} candidates; "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

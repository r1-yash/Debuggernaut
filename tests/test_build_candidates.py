import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmark" / "tools")
)

import build_candidates  # noqa: E402

ensure_dataset_cache = build_candidates.ensure_dataset_cache
filter_row = build_candidates.filter_row
is_doc_or_ci_path = build_candidates.is_doc_or_ci_path
is_python_test_path = build_candidates.is_python_test_path
load_dataset_rows = build_candidates.load_dataset_rows
parse_diff = build_candidates.parse_diff
select_candidates = build_candidates.select_candidates
write_candidate_outputs = build_candidates.write_candidate_outputs


FIX_PATCH = """diff --git a/widget/core.py b/widget/core.py
--- a/widget/core.py
+++ b/widget/core.py
@@ -1 +1 @@
-old value
+new value
"""

TWO_FILE_FIX_PATCH = FIX_PATCH + """diff --git a/widget/util.py b/widget/util.py
--- a/widget/util.py
+++ b/widget/util.py
@@ -1 +1 @@
-old helper
+new helper
"""

ADDED_SOURCE_PATCH = """diff --git a/widget/core.py b/widget/core.py
new file mode 100644
--- /dev/null
+++ b/widget/core.py
@@ -0,0 +1 @@
+new value
"""

TEST_PATCH = """diff --git a/tests/common.py b/tests/common.py
--- a/tests/common.py
+++ b/tests/common.py
@@ -1 +1 @@
-old helper
+new helper
diff --git a/tests/test_widget.py b/tests/test_widget.py
--- a/tests/test_widget.py
+++ b/tests/test_widget.py
@@ -1 +1 @@
-old test
+new test
"""

NON_TEST_PATCH = """diff --git a/tests/fixtures/data.json b/tests/fixtures/data.json
--- a/tests/fixtures/data.json
+++ b/tests/fixtures/data.json
@@ -1 +1 @@
-old
+new
"""

DELETED_TEST_PATCH = """diff --git a/tests/test_widget.py b/tests/test_widget.py
deleted file mode 100644
--- a/tests/test_widget.py
+++ /dev/null
@@ -1 +0,0 @@
-old test
"""


def _row(
    *,
    instance_id: str = "acme__widget-7",
    repo: str = "acme/widget",
    patch: str = FIX_PATCH,
    test_patch: str = TEST_PATCH,
    created_at: str = "2024-01-01 00:00:00+00:00",
) -> dict[str, object]:
    """Return one minimal dataset row."""
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "a" * 40,
        "created_at": created_at,
        "patch": patch,
        "test_patch": test_patch,
        "problem_statement": "Widget is broken.",
        "FAIL_TO_PASS": ["tests/test_widget.py::test_fix"],
        "PASS_TO_PASS": [],
        "meta": {"has_test_patch": True, "is_lite": False},
        "license": "MIT",
    }


def test_parse_diff_tracks_modified_added_deleted_and_renamed_files() -> None:
    changes = parse_diff(
        """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-old
+new
diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,2 @@
+one
+two
diff --git a/pkg/old.py b/pkg/old.py
deleted file mode 100644
--- a/pkg/old.py
+++ /dev/null
@@ -1 +0,0 @@
-old
diff --git a/pkg/renamed.py b/pkg/renamed.py
similarity index 90%
rename from pkg/original.py
rename to pkg/renamed.py
--- a/pkg/original.py
+++ b/pkg/renamed.py
@@ -1 +1 @@
-old
+new
"""
    )

    assert [(change.path, change.status) for change in changes] == [
        ("pkg/mod.py", "modified"),
        ("pkg/new.py", "added"),
        ("pkg/old.py", "deleted"),
        ("pkg/renamed.py", "renamed"),
    ]
    assert changes[0].additions == 1 and changes[0].deletions == 1
    assert changes[1].additions == 2 and changes[1].hunks == 1
    assert changes[2].deletions == 1
    assert changes[3].old_path == "pkg/original.py"


def test_path_classification_separates_tests_and_docs() -> None:
    assert is_python_test_path("tests/common.py")
    assert is_python_test_path("pkg/test_core.py")
    assert is_python_test_path("pkg/core_test.py")
    assert not is_python_test_path("pkg/core.py")
    assert not is_python_test_path("tests/fixtures/data.json")

    assert is_doc_or_ci_path("docs/conf.py")
    assert is_doc_or_ci_path(".github/scripts/check.py")
    assert is_doc_or_ci_path("CHANGELOG.rst")
    assert not is_doc_or_ci_path("pkg/core.py")


def test_filter_row_accepts_one_source_file_and_prefers_failing_test_files() -> None:
    result = filter_row(_row())

    assert result.accepted
    assert result.reason is None
    assert result.pr_number == 7
    assert result.fix_file == "widget/core.py"
    assert result.test_files == ("tests/test_widget.py",)
    assert result.test_command == (
        "python",
        "-m",
        "pytest",
        "tests/test_widget.py",
    )
    assert result.fix_change is not None
    assert result.fix_change.path == "widget/core.py"

    row_without_node_ids = _row()
    row_without_node_ids["FAIL_TO_PASS"] = []
    fallback = filter_row(row_without_node_ids)
    assert fallback.accepted
    assert fallback.test_command == (
        "python",
        "-m",
        "pytest",
        "tests/test_widget.py",
    )


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (_row(patch=TWO_FILE_FIX_PATCH), "fix_patch_source_file_count:2"),
        (_row(patch=ADDED_SOURCE_PATCH), "fix_file_not_existing:added"),
        (_row(test_patch=NON_TEST_PATCH), "test_patch_has_no_test_file"),
        (_row(test_patch=DELETED_TEST_PATCH), "test_patch_has_no_test_file"),
    ],
)
def test_filter_row_rejects_non_candidate_shapes(
    row: dict[str, object], reason: str
) -> None:
    result = filter_row(row)

    assert not result.accepted
    assert result.reason == reason
    assert result.test_command is None


def test_offline_dataset_cache_round_trip_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is forbidden in offline mode")

    monkeypatch.setattr(
        build_candidates.requests, "get", fail_network
    )

    with pytest.raises(ValueError, match="Offline mode"):
        ensure_dataset_cache(tmp_path, offline=True)

    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8"
    )

    assert ensure_dataset_cache(tmp_path, offline=True) == rows_path
    assert load_dataset_rows(rows_path)[0]["instance_id"] == "acme__widget-7"


def test_select_candidates_uses_newest_resolved_row_per_repo_and_limit() -> None:
    rows = [
        _row(
            instance_id="acme__widget-1",
            repo="acme/widget",
            created_at="2024-01-01 00:00:00+00:00",
        ),
        _row(
            instance_id="acme__widget-2",
            repo="acme/widget",
            created_at="2024-03-01 00:00:00+00:00",
        ),
        _row(
            instance_id="acme__gadget-5",
            repo="acme/gadget",
            created_at="2024-01-01 00:00:00+00:00",
        ),
        _row(
            instance_id="acme__gadget-6",
            repo="acme/gadget",
            created_at="2024-02-01 00:00:00+00:00",
        ),
        _row(
            instance_id="acme__third-9",
            repo="acme/third",
            created_at="2024-02-15 00:00:00+00:00",
        ),
        _row(
            instance_id="acme__other-7",
            repo="acme/other",
            created_at="2024-04-01 00:00:00+00:00",
        ),
    ]
    selection = select_candidates(
        rows,
        issue_numbers={
            ("acme/widget", 1): 11,
            ("acme/widget", 2): 12,
            ("acme/gadget", 5): 15,
            ("acme/third", 9): 19,
        },
        limit=2,
    )

    assert selection.total_rows == 6
    assert selection.filtered_rows == 6
    assert selection.filtered_repositories == 4
    assert len(selection.available) == 3
    assert [(item.repo, item.issue_number) for item in selection.candidates] == [
        ("acme/widget", 12),
        ("acme/third", 19),
    ]
    assert len(selection.unresolved_rows) == 2
    assert selection.unresolved_repositories == 1
    assert {
        (item.repo, item.pr_number) for item in selection.unresolved_rows
    } == {("acme/gadget", 6), ("acme/other", 7)}


def test_write_candidate_outputs_writes_config_gold_and_reports(
    tmp_path: Path,
) -> None:
    selection = select_candidates(
        [_row()],
        issue_numbers={("acme/widget", 7): 42},
        limit=150,
    )
    output = tmp_path / "candidates"

    write_candidate_outputs(selection, output)

    config = json.loads((output / "repos.json").read_text(encoding="utf-8"))
    assert config == {
        "acme/widget": {
            "issue_number": 42,
            "commit": "a" * 40,
            "test_command": [
                "python",
                "-m",
                "pytest",
                "tests/test_widget.py",
            ],
            "test_patch": "patches/acme__widget__42.diff",
        }
    }
    assert "gold" not in json.dumps(config)
    assert (output / "patches/acme__widget__42.diff").read_bytes() == (
        TEST_PATCH.encode("utf-8")
    )
    assert (output / "gold/acme__widget__42.diff").read_bytes() == (
        FIX_PATCH.encode("utf-8")
    )

    report = json.loads(
        (output / "candidates_report.json").read_text(encoding="utf-8")
    )
    candidate = report["candidates"][0]
    assert report["selection"]["selected_candidates"] == 1
    assert candidate["issue_number"] == 42
    assert candidate["gold_patch"] == "gold/acme__widget__42.diff"
    assert "one_candidate_per_repo" in candidate["filters_passed"]
    markdown = (output / "candidates_report.md").read_text(
        encoding="utf-8"
    )
    assert "acme/widget" in markdown
    assert "gold/acme__widget__42.diff" in markdown

    with pytest.raises(ValueError, match="not empty"):
        write_candidate_outputs(selection, output)


def test_cli_dry_run_prints_counts_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(
        json.dumps(_row(), sort_keys=True) + "\n", encoding="utf-8"
    )
    issue_map_path = tmp_path / "issues.json"
    issue_map_path.write_text(
        json.dumps({"acme/widget#7": 42}), encoding="utf-8"
    )
    output = tmp_path / "candidates"

    exit_code = build_candidates.main(
        [
            "--dataset-jsonl",
            str(rows_path),
            "--issue-map",
            str(issue_map_path),
            "--output",
            str(output),
            "--limit",
            "5",
            "--dry-run",
        ]
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert "rows: 1" in printed
    assert "selected candidates: 1" in printed
    assert "acme/widget#42" in printed
    assert "tests: tests/test_widget.py" in printed
    assert not output.exists()

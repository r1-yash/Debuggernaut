import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# main.py loads project secrets at import time; keep the tests hermetic.
with patch("dotenv.load_dotenv"):
    import main as cli
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.validate import (
    BaselineResult,
    summarize_baselines,
    validate_benchmark,
    validate_entry,
)


def _entry(
    test_command: list[str],
    commit: str = "a" * 40,
    test_patch: str | None = None,
    owner: str = "acme",
    repository: str = "widget",
    issue_number: int = 42,
) -> BenchmarkEntry:
    """Return a benchmark entry for the throwaway local repository."""
    return BenchmarkEntry(
        owner=owner,
        repository=repository,
        issue_number=issue_number,
        commit=commit,
        test_command=test_command,
        test_patch=test_patch,
    )


def _write_config(path: Path, entries: list[BenchmarkEntry]) -> Path:
    """Write a benchmark config containing the given entries in order."""
    payload = {
        f"{entry.owner}/{entry.repository}": {
            "issue_number": entry.issue_number,
            "commit": entry.commit,
            "test_command": entry.test_command,
            "test_patch": entry.test_patch,
        }
        for entry in entries
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _baseline(
    entry: BenchmarkEntry,
    *,
    status: str = "fails_as_expected",
    detail: str = "ok",
    return_code: int | None = 1,
    output_tail: str = "",
) -> BaselineResult:
    """Return a completed baseline result for validation-loop tests."""
    return BaselineResult(
        entry=entry,
        status=status,
        detail=detail,
        return_code=return_code,
        output_tail=output_tail,
    )


def _make_repo(path: Path, calc_content: str, test_content: str) -> Path:
    """Create a tiny repository with one module and one test file."""
    path.mkdir()
    (path / "calc.py").write_text(calc_content)
    (path / "test_calc.py").write_text(test_content)
    return path


def _failing_repo(path: Path) -> Path:
    """Create a repository whose pinned test fails."""
    return _make_repo(
        path,
        "def add(a, b):\n    return a - b\n",
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )


def _pytest_command(path: Path) -> list[str]:
    """Return a pytest command bound to the current interpreter."""
    return [sys.executable, "-m", "pytest", "-q", str(path / "test_calc.py")]


def _git(repo_path: Path, *arguments: str) -> None:
    """Run a git command in a test repository."""
    subprocess.run(
        ["git", *arguments],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )


def _git_repo(path: Path, calc_content: str, test_content: str) -> Path:
    """Create a committed repository so test patches can be applied."""
    repo = _make_repo(path, calc_content, test_content)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def test_failing_tests_reproduce_baseline(tmp_path: Path) -> None:
    repo = _failing_repo(tmp_path / "widget")
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "fails_as_expected"
    assert result.return_code == 1


def test_applies_test_patch_before_running_tests(tmp_path: Path) -> None:
    repo = _git_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a - b\n",
        "# placeholder\n",
    )
    patch_file = tmp_path / "add_test.diff"
    patch_file.write_text(
        """diff --git a/test_regression.py b/test_regression.py
new file mode 100644
--- /dev/null
+++ b/test_regression.py
@@ -0,0 +1,4 @@
+from calc import add
+
+def test_add():
+    assert add(1, 2) == 3
"""
    )
    entry = _entry(
        [sys.executable, "-m", "pytest", "-q", str(repo / "test_regression.py")],
        test_patch=str(patch_file),
    )

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "fails_as_expected"
    assert result.return_code == 1
    assert (repo / "test_regression.py").is_file()


def test_applies_test_patch_that_modifies_existing_test(tmp_path: Path) -> None:
    repo = _git_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a - b\n",
        "# placeholder\n",
    )
    patch_file = tmp_path / "modify_test.diff"
    patch_file.write_text(
        """diff --git a/test_calc.py b/test_calc.py
--- a/test_calc.py
+++ b/test_calc.py
@@ -1 +1,4 @@
-# placeholder
+from calc import add
+
+def test_add():
+    assert add(1, 2) == 3
"""
    )
    entry = _entry(_pytest_command(repo), test_patch=str(patch_file))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "fails_as_expected"
    assert (repo / "test_calc.py").read_text().startswith("from calc import add")


def test_patch_failure_is_not_passes_unexpectedly_or_could_not_run(
    tmp_path: Path,
) -> None:
    repo = _git_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a - b\n",
        "def test_ok():\n    assert True\n",
    )
    patch_file = tmp_path / "bad.diff"
    patch_file.write_text(
        """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a * b
"""
    )
    passing_entry = _entry(_pytest_command(repo), test_patch=str(patch_file))
    missing_entry = _entry(
        ["definitely-not-a-real-binary-xyz"], test_patch=str(patch_file)
    )

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        passing_result = validate_entry(
            passing_entry, str(tmp_path / "workspace")
        )
        missing_result = validate_entry(
            missing_entry, str(tmp_path / "workspace")
        )

    assert passing_result.status == "test_patch_failed"
    assert missing_result.status == "test_patch_failed"
    assert passing_result.return_code is None
    assert missing_result.return_code is None
    assert "failed to apply" in passing_result.detail


def test_patched_tests_can_pass_unexpectedly(tmp_path: Path) -> None:
    repo = _git_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a + b\n",
        "# placeholder\n",
    )
    patch_file = tmp_path / "passing_test.diff"
    patch_file.write_text(
        """diff --git a/test_calc.py b/test_calc.py
--- a/test_calc.py
+++ b/test_calc.py
@@ -1 +1,4 @@
-# placeholder
+from calc import add
+
+def test_add():
+    assert add(1, 2) == 3
"""
    )
    entry = _entry(_pytest_command(repo), test_patch=str(patch_file))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "passes_unexpectedly"
    assert result.return_code == 0


def test_passing_tests_make_entry_invalid(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a + b\n",
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "passes_unexpectedly"
    assert result.return_code == 0


def test_pytest_exit_code_two_is_could_not_run(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a - b\n",
        "def test_interrupt():\n    raise KeyboardInterrupt\n",
    )
    # Pytest exit code 2 means test execution was interrupted.
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "could_not_run"
    assert result.return_code == 2


def test_pytest_exit_code_five_is_could_not_run(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path / "widget",
        "def add(a, b):\n    return a - b\n",
        "# No tests collected.\n",
    )
    # Pytest exit code 5 means no tests were collected.
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "could_not_run"
    assert result.return_code == 5


def test_missing_test_command_is_could_not_run(tmp_path: Path) -> None:
    repo = _failing_repo(tmp_path / "widget")
    entry = _entry(["definitely-not-a-real-binary-xyz"])

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "could_not_run"
    assert result.return_code is None
    assert "definitely-not-a-real-binary-xyz" in result.detail


def test_unfetchable_commit_is_could_not_run(tmp_path: Path) -> None:
    entry = _entry(["pytest"], commit="f" * 40)

    with patch(
        "swe_agent.benchmark.validate.clone_repository",
        side_effect=ValueError(
            f"Could not fetch commit {'f' * 40} from /upstream: not our ref"
        ),
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "could_not_run"
    assert result.return_code is None
    assert "f" * 40 in result.detail


def test_report_includes_exit_code_and_output_tail(tmp_path: Path) -> None:
    repo = _failing_repo(tmp_path / "widget")
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    report = summarize_baselines([result])
    assert "acme/widget#42: fails_as_expected (exit code 1)" in report
    assert "Last test output lines:" in report


def test_summary_counts_reproducible_entries() -> None:
    entry = _entry(["pytest"])
    results = [
        BaselineResult(
            entry=entry,
            status="fails_as_expected",
            detail="ok",
            return_code=1,
            output_tail="",
        ),
        BaselineResult(
            entry=entry,
            status="could_not_run",
            detail="boom",
            return_code=None,
            output_tail="",
        ),
    ]

    report = summarize_baselines(results)

    assert "Total entries: 2" in report
    assert "Reproduce as failing: 1" in report
    assert "Not reproducible: 1" in report


def test_summary_reports_test_patch_failure_as_not_reproducible() -> None:
    entry = _entry(["pytest"])
    result = BaselineResult(
        entry=entry,
        status="test_patch_failed",
        detail="Test patch failed to apply.",
        return_code=None,
        output_tail="",
    )

    report = summarize_baselines([result])

    assert "Reproduce as failing: 0" in report
    assert "Not reproducible: 1" in report
    assert "test_patch_failed (exit code unknown)" in report


def test_validate_benchmark_prints_progress_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(["pytest"])
    config_path = _write_config(tmp_path / "bench.json", [entry])
    result = _baseline(entry)
    times = iter([100.0, 108.3])

    monkeypatch.setattr(
        "swe_agent.benchmark.validate.validate_entry",
        lambda entry, workspace: result,
    )
    monkeypatch.setattr(
        "swe_agent.benchmark.validate.time.monotonic",
        lambda: next(times),
    )

    results = validate_benchmark(str(config_path))

    assert results == [result]
    assert (
        capsys.readouterr().out
        == "[1/1] acme/widget fails_as_expected (8.3s)\n"
    )


def test_validate_limit_uses_first_entries_in_config_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(["pytest"], repository="first")
    second = _entry(["pytest"], repository="second")
    config_path = _write_config(tmp_path / "bench.json", [first, second])
    validated: list[str] = []

    def fake_validate_entry(
        entry: BenchmarkEntry, workspace_dir: str
    ) -> BaselineResult:
        validated.append(entry.repository)
        return _baseline(entry)

    monkeypatch.setattr(
        "swe_agent.benchmark.validate.validate_entry", fake_validate_entry
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--validate",
            "--limit",
            "1",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert validated == ["first"]
    assert "[1/1] acme/first fails_as_expected" in output
    assert "Total entries: 1" in output
    assert "acme/second" not in output


def test_validate_results_file_appends_json_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(["pytest"])
    config_path = _write_config(tmp_path / "bench.json", [entry])
    output_tail = "\n".join(f"line {index}" for index in range(25))
    result = _baseline(entry, output_tail=output_tail)
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"existing": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        "swe_agent.benchmark.validate.validate_entry",
        lambda entry, workspace: result,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--validate",
            "--results-file",
            str(results_path),
        ],
    )

    cli.main()
    capsys.readouterr()

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"existing": true}'
    record = json.loads(lines[1])
    assert record == {
        "repo": "acme/widget",
        "status": "fails_as_expected",
        "return_code": 1,
        "detail": "ok",
        "output_tail": "\n".join(
            f"line {index}" for index in range(5, 25)
        ),
    }


def test_validate_ctrl_c_prints_partial_summary_and_keeps_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry(["pytest"], repository="first")
    second = _entry(["pytest"], repository="second")
    config_path = _write_config(tmp_path / "bench.json", [first, second])
    results_path = tmp_path / "results.jsonl"

    def fake_validate_entry(
        entry: BenchmarkEntry, workspace_dir: str
    ) -> BaselineResult:
        if entry.repository == "second":
            raise KeyboardInterrupt
        return _baseline(entry)

    monkeypatch.setattr(
        "swe_agent.benchmark.validate.validate_entry", fake_validate_entry
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--validate",
            "--results-file",
            str(results_path),
        ],
    )

    with pytest.raises(SystemExit) as interrupted:
        cli.main()

    output = capsys.readouterr().out
    assert interrupted.value.code == 130
    assert "[1/2] acme/first fails_as_expected" in output
    assert "Total entries: 1" in output
    assert "acme/second" not in output
    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 1

import sys
from pathlib import Path
from unittest.mock import patch

from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.validate import (
    BaselineResult,
    summarize_baselines,
    validate_entry,
)


def _entry(test_command: list[str], commit: str = "a" * 40) -> BenchmarkEntry:
    """Return a benchmark entry for the throwaway local repository."""
    return BenchmarkEntry(
        owner="acme",
        repository="widget",
        issue_number=42,
        commit=commit,
        test_command=test_command,
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


def test_failing_tests_reproduce_baseline(tmp_path: Path) -> None:
    repo = _failing_repo(tmp_path / "widget")
    entry = _entry(_pytest_command(repo))

    with patch(
        "swe_agent.benchmark.validate.clone_repository", return_value=str(repo)
    ):
        result = validate_entry(entry, str(tmp_path / "workspace"))

    assert result.status == "fails_as_expected"
    assert result.return_code == 1


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

import json
from datetime import UTC, datetime
from pathlib import Path

from swe_agent.agent.diagnose import Diagnosis
from swe_agent.agent.executor import TestOutcome
from swe_agent.agent.fixer import ExplorationStep, FixResult
from swe_agent.agent.loop import Attempt, LoopResult
from swe_agent.benchmark.baseline import BaselineResult
from swe_agent.benchmark.config import BenchmarkEntry
from swe_agent.benchmark.environment import EnvironmentSummary
from swe_agent.benchmark.reporting import (
    RunTiming,
    format_run_summary,
    write_benchmark_summary,
    write_run_report,
)
from swe_agent.benchmark.test_patch import TestPatchResult


def _entry() -> BenchmarkEntry:
    return BenchmarkEntry(
        owner="acme",
        repository="widget",
        issue_number=7,
        commit="a" * 40,
        test_command=["pytest"],
    )


def _timing() -> RunTiming:
    return RunTiming(
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        finished_at=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
        duration_seconds=2.0,
    )


def _attempt(
    *,
    original: str = "def add(a, b):\n    return a - b\n",
    new: str = "def add(a, b):\n    return a + b\n",
    passed: bool = False,
    stdout: str = "",
    stderr: str = "",
    diagnosis: Diagnosis | None = None,
    avoid: list[str] | None = None,
    steps: list[ExplorationStep] | None = None,
) -> Attempt:
    return Attempt(
        fix_result=FixResult(
            file_path="calc.py",
            original_content=original,
            new_content=new,
            reasoning="The subtraction operands are inverted.",
        ),
        outcome=TestOutcome(
            passed=passed,
            return_code=0 if passed else 1,
            stdout=stdout,
            stderr=stderr,
        ),
        diagnosis=diagnosis,
        avoid_file_paths_after=avoid or [],
        exploration_steps=steps or [],
    )


def _write(tmp_path: Path, loop: LoopResult, *, status: str = "failed", error: str | None = None) -> dict:
    path = write_run_report(
        str(tmp_path / "runs"), _entry(), "gemini", "gemini-test", loop,
        status, None, _timing(), error=error,
    )
    return json.loads(Path(path).read_text())


def test_report_contains_complete_attempt_record(tmp_path: Path) -> None:
    attempt = _attempt(
        stdout="1 failed in 0.02s",
        diagnosis=Diagnosis(reason="wrong_fix", explanation="Operator still inverted."),
        avoid=["calc.py"],
        steps=[
            ExplorationStep(
                tool="read_file", arguments={"path": "calc.py"}, ok=True,
                result_size=41, result_unit="characters",
            ),
            ExplorationStep(
                tool="search_code", arguments={"query": "def add"}, ok=True,
                result_size=1, result_unit="lines",
            ),
        ],
    )
    payload = _write(
        tmp_path, LoopResult(succeeded=False, final_attempt=attempt, attempts=[attempt])
    )

    assert payload["schema_version"] == 4
    assert payload["repository"] == "acme/widget"
    assert payload["issue"] == 7
    assert payload["commit"] == "a" * 40
    assert payload["provider"] == "gemini"
    assert payload["model"] == "gemini-test"
    assert payload["status"] == "failed"
    assert payload["failure_class"] == "agent"
    assert payload["status_reason"] is None
    assert payload["error"] is None
    assert payload["baseline"] is None
    assert payload["environment"] is None
    assert payload["duration_seconds"] == 2.0
    assert payload["started_at"] == "2026-01-01T00:00:00+00:00"
    assert payload["finished_at"] == "2026-01-01T00:00:02+00:00"
    assert payload["attempts"] == 1

    record = payload["attempt_records"][0]
    assert record["attempt"] == 1
    assert record["file_path"] == "calc.py"
    assert record["reasoning"] == "The subtraction operands are inverted."
    assert "-    return a - b" in record["diff"]
    assert "+    return a + b" in record["diff"]
    assert record["diff_truncated"] is False
    assert record["test"] == {
        "passed": False,
        "return_code": 1,
        "timed_out": False,
        "started": True,
        "timeout_seconds": None,
        "output_tail": "1 failed in 0.02s",
        "output_truncated": False,
    }
    assert record["diagnosis"] == {
        "reason": "wrong_fix",
        "explanation": "Operator still inverted.",
    }
    assert record["avoid_file_paths_after"] == ["calc.py"]
    assert [step["tool"] for step in record["exploration"]] == [
        "read_file",
        "search_code",
    ]
    assert record["exploration"][0]["arguments"] == {"path": "calc.py"}
    assert record["exploration"][1]["result_size"] == 1
    assert record["exploration"][1]["result_unit"] == "lines"
    assert record["exploration_truncated"] is False
    assert payload["guard_events"] == []


def test_report_excludes_file_content_outside_the_diff(tmp_path: Path) -> None:
    original_lines = [f"line {number}\n" for number in range(50)]
    new_lines = list(original_lines)
    new_lines[25] = "changed line\n"
    attempt = _attempt(
        original="".join(original_lines), new="".join(new_lines), passed=True
    )

    payload = _write(
        tmp_path,
        LoopResult(succeeded=True, final_attempt=attempt, attempts=[attempt]),
        status="success",
    )

    serialized = json.dumps(payload)
    record = payload["attempt_records"][0]
    assert "line 0\n" not in serialized
    assert "-line 25" in record["diff"]
    assert "+changed line" in record["diff"]
    assert "original_content" not in record
    assert "new_content" not in record


def test_diff_is_truncated_and_flagged(tmp_path: Path) -> None:
    attempt = _attempt(original="a" * 20_000, new="b" * 20_000)

    record = _write(
        tmp_path, LoopResult(succeeded=False, final_attempt=attempt, attempts=[attempt])
    )["attempt_records"][0]

    assert record["diff_truncated"] is True
    assert len(record["diff"]) == 8_000


def test_output_tail_is_bounded_and_flagged(tmp_path: Path) -> None:
    attempt = _attempt(
        stdout="\n".join(f"line {number}" for number in range(1, 101))
    )

    record = _write(
        tmp_path, LoopResult(succeeded=False, final_attempt=attempt, attempts=[attempt])
    )["attempt_records"][0]

    tail_lines = record["test"]["output_tail"].splitlines()
    assert len(tail_lines) == 40
    assert tail_lines[0] == "line 61"
    assert tail_lines[-1] == "line 100"
    assert record["test"]["output_truncated"] is True


def test_exploration_is_capped_and_flagged(tmp_path: Path) -> None:
    steps = [
        ExplorationStep(
            tool="read_file", arguments={"path": f"file_{number}.py"}, ok=True,
            result_size=number, result_unit="characters",
        )
        for number in range(101)
    ]
    attempt = _attempt(steps=steps)

    record = _write(
        tmp_path, LoopResult(succeeded=False, final_attempt=attempt, attempts=[attempt])
    )["attempt_records"][0]

    assert len(record["exploration"]) == 100
    assert record["exploration"][-1]["arguments"] == {"path": "file_99.py"}
    assert record["exploration_truncated"] is True


def test_errored_report_records_error(tmp_path: Path) -> None:
    payload = _write(
        tmp_path,
        LoopResult(succeeded=False, final_attempt=None, attempts=[]),
        status="errored",
        error="Git clone failed",
    )

    assert payload["status"] == "errored"
    assert payload["failure_class"] == "infrastructure"
    assert payload["error"] == "Git clone failed"
    assert payload["attempt_records"] == []
    assert payload["input_tokens"] is None
    assert payload["test_patch"] == {"applied": False, "file_paths": []}


def test_report_records_applied_test_patch(tmp_path: Path) -> None:
    attempt = _attempt(passed=True)
    path = write_run_report(
        str(tmp_path / "runs"),
        _entry(),
        "gemini",
        "gemini-test",
        LoopResult(succeeded=True, final_attempt=attempt, attempts=[attempt]),
        "success",
        None,
        _timing(),
        test_patch=TestPatchResult(
            applied=True,
            file_paths=["tests/test_calc.py", "tests/helpers.py"],
            baseline_commit="b" * 40,
        ),
    )
    payload = json.loads(Path(path).read_text())

    assert payload["test_patch"] == {
        "applied": True,
        "file_paths": ["tests/test_calc.py", "tests/helpers.py"],
    }
    assert "baseline_commit" not in payload["test_patch"]


def test_format_run_summary_reads_new_payload(tmp_path: Path) -> None:
    attempt = _attempt(passed=True)
    payload = _write(
        tmp_path,
        LoopResult(succeeded=True, final_attempt=attempt, attempts=[attempt]),
        status="success",
    )

    summary = format_run_summary(payload)

    assert "DEBUGGERNAUT RUN SUMMARY" in summary
    assert "Repository:  acme/widget" in summary
    assert "Status:      SUCCESS" in summary
    assert "Attempts:    1" in summary


def test_benchmark_summary_counts_all_statuses_and_paths(tmp_path: Path) -> None:
    results = [
        (str(tmp_path / "runs" / "a.json"), "success"),
        (str(tmp_path / "runs" / "b.json"), "failed"),
        (str(tmp_path / "runs" / "c.json"), "test_timeout"),
        (str(tmp_path / "runs" / "d.json"), "could_not_install"),
        (str(tmp_path / "runs" / "e.json"), "errored"),
    ]
    path = write_benchmark_summary(str(tmp_path / "runs"), results, _timing())
    payload = json.loads(Path(path).read_text())

    assert path.name.endswith("_benchmark_summary.json")
    assert payload["schema_version"] == 4
    assert payload["duration_seconds"] == 2.0
    assert payload["counts"] == {
        "total": 5,
        "succeeded": 1,
        "failed": 1,
        "test_timeout": 1,
        "context_too_large": 0,
        "could_not_install": 1,
        "provider_unavailable": 0,
        "errored": 1,
    }
    assert payload["reports"] == [report for report, _ in results]


def test_schema_v3_report_includes_baseline_environment_and_reason(
    tmp_path: Path,
) -> None:
    environment = EnvironmentSummary(
        cleanup_errors=["Could not remove repo: PermissionError: denied"]
    )
    baseline = BaselineResult(
        entry=_entry(),
        status="test_timeout",
        detail="The baseline test command timed out after 300 s.",
        return_code=-1,
        output_tail="baseline output",
        environment=environment,
    )
    path = write_run_report(
        str(tmp_path / "runs"),
        _entry(),
        "gemini",
        "gemini-test",
        LoopResult(succeeded=False, final_attempt=None, attempts=[]),
        "errored",
        None,
        _timing(),
        status_reason="Baseline tests timed out.",
        baseline=baseline,
        environment=environment,
        failure_class="infrastructure",
    )
    payload = json.loads(Path(path).read_text())

    assert payload["schema_version"] == 4
    assert payload["status_reason"] == "Baseline tests timed out."
    assert payload["baseline"] == {
        "status": "test_timeout",
        "detail": "The baseline test command timed out after 300 s.",
        "return_code": -1,
        "output_tail": "baseline output",
        "output_truncated": False,
    }
    assert payload["environment"]["cleanup_errors"] == [
        "Could not remove repo: PermissionError: denied"
    ]
    assert payload["failure_class"] == "infrastructure"


def test_agent_timeout_report_is_labeled_agent_failure(tmp_path: Path) -> None:
    attempt = _attempt()
    attempt.outcome.timed_out = True
    attempt.outcome.timeout_seconds = 120.0
    path = write_run_report(
        str(tmp_path / "runs"),
        _entry(),
        "gemini",
        "gemini-test",
        LoopResult(succeeded=False, final_attempt=attempt, attempts=[attempt]),
        "test_timeout",
        None,
        _timing(),
        status_reason="Agent tests timed out after 120 s.",
        failure_class="agent",
    )
    payload = json.loads(Path(path).read_text())

    assert payload["status"] == "test_timeout"
    assert payload["failure_class"] == "agent"
    assert payload["status_reason"] == "Agent tests timed out after 120 s."
    assert payload["attempt_records"][0]["test"]["timed_out"] is True
    assert payload["attempt_records"][0]["test"]["timeout_seconds"] == 120.0

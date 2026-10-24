from swe_agent.agent.executor import TestOutcome
from swe_agent.benchmark.baseline import classify_baseline


def _outcome(
    *,
    passed: bool = False,
    return_code: int | None = 1,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    started: bool = True,
    timeout_seconds: float | None = None,
) -> TestOutcome:
    return TestOutcome(
        passed=passed,
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        started=started,
        timeout_seconds=timeout_seconds,
    )


def _classify(outcome: TestOutcome) -> str:
    status, _detail = classify_baseline(
        outcome,
        own_top_level_modules={"widget"},
    )
    return status


def test_timeout_baseline_is_test_timeout() -> None:
    assert (
        _classify(
            _outcome(
                return_code=-1,
                timed_out=True,
                timeout_seconds=300.0,
                stderr="",
            )
        )
        == "test_timeout"
    )


def test_process_that_could_not_start_is_could_not_install() -> None:
    assert (
        _classify(
            _outcome(
                return_code=None,
                started=False,
                stderr="FileNotFoundError: pytest",
            )
        )
        == "could_not_install"
    )


def test_passing_baseline_is_passes_unexpectedly() -> None:
    assert (
        _classify(_outcome(passed=True, return_code=0))
        == "passes_unexpectedly"
    )


def test_missing_third_party_module_is_could_not_install() -> None:
    assert (
        _classify(
            _outcome(
                stderr=(
                    "E   ModuleNotFoundError: No module named 'requests'\n"
                )
            )
        )
        == "could_not_install"
    )


def test_missing_own_package_is_fails_as_expected() -> None:
    assert (
        _classify(
            _outcome(
                stderr=(
                    "E   ModuleNotFoundError: No module named 'widget.parser'\n"
                )
            )
        )
        == "fails_as_expected"
    )


def test_cannot_import_name_is_fails_as_expected() -> None:
    assert (
        _classify(
            _outcome(
                stderr=(
                    "E   ImportError: cannot import name 'parse' from 'widget'\n"
                )
            )
        )
        == "fails_as_expected"
    )


def test_pytest_collection_error_exit_code_two_is_fails_as_expected() -> None:
    assert (
        _classify(
            _outcome(
                return_code=2,
                stderr="ERROR: found no collectors for test_widget.py",
            )
        )
        == "fails_as_expected"
    )


def test_ordinary_failing_tests_are_fails_as_expected() -> None:
    assert (
        _classify(
            _outcome(
                return_code=1,
                stdout="1 failed in 0.02s",
            )
        )
        == "fails_as_expected"
    )

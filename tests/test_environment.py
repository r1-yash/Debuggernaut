import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from unittest.mock import Mock, patch

import pytest

from swe_agent.benchmark.environment import (
    ENVIRONMENT_DIR_NAME,
    EnvironmentSetupError,
    CommandResult,
    EntryEnvironment,
    SubprocessRunner,
    build_install_plan,
    build_subprocess_env,
    discover_requirement_files,
    ensure_python_installed,
    entry_environment,
    execute_install_plan,
    sweep_stale_workspaces,
    venv_python_path,
)


@dataclass(frozen=True)
class RecordedCall:
    command: list[str]
    cwd: str
    env: dict[str, str]
    timeout: float | None


class SequenceRunner:
    """A deterministic subprocess boundary for environment unit tests."""

    def __init__(
        self,
        outcomes: Sequence[
            CommandResult
            | tuple[int | None, str, str, bool]
            | BaseException
            | Callable[[Sequence[str]], CommandResult]
        ],
    ) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[RecordedCall] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandResult:
        self.calls.append(
            RecordedCall(
                command=list(command),
                cwd=os.fspath(cwd),
                env=dict(env),
                timeout=timeout,
            )
        )
        if not self._outcomes:
            raise AssertionError(f"Unexpected command: {list(command)!r}")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, CommandResult):
            return outcome.model_copy(update={"command": list(command)})
        if callable(outcome):
            return outcome(command)
        return_code, stdout, stderr, timed_out = outcome
        return CommandResult(
            command=list(command),
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=0.01,
        )

    def remaining(self) -> int:
        return len(self._outcomes)


def _ok(stdout: str = "", stderr: str = "") -> tuple[int, str, str, bool]:
    return 0, stdout, stderr, False


def _fail(
    stdout: str = "",
    stderr: str = "failed",
) -> tuple[int, str, str, bool]:
    return 1, stdout, stderr, False


def _timeout(
    stdout: str = "",
    stderr: str = "timed out",
) -> tuple[int, str, str, bool]:
    return -9, stdout, stderr, True


def _repo(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "module.py").write_text("value = 1\n", encoding="utf-8")
    return repository


def _environment_success_runner() -> SequenceRunner:
    return SequenceRunner(
        [
            _ok("uv 0.11.32 (deadbeef)\n"),
            _ok(),
            _ok(),
            _ok("3.11.9\n"),
            _ok("editable test install\n"),
            _ok("pytest install\n"),
            _ok("tests passed\n"),
        ]
    )


def test_subprocess_environment_excludes_secrets_and_uses_throwaway_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("PARENT_ONLY", "do-not-pass")

    home = tmp_path / "home"
    temp_dir = tmp_path / "tmp"
    venv = tmp_path / "venv"
    managed_python = tmp_path / "uv-python"
    env = build_subprocess_env(
        home,
        temp_dir,
        venv_path=venv,
        python_install_dir=managed_python,
    )

    assert env["HOME"] == os.fspath(home)
    assert env["TMPDIR"] == os.fspath(temp_dir)
    assert env["TEMP"] == os.fspath(temp_dir)
    assert env["TMP"] == os.fspath(temp_dir)
    assert env["LANG"] == "C.UTF-8"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["VIRTUAL_ENV"] == os.fspath(venv)
    assert env["UV_PYTHON_INSTALL_DIR"] == os.fspath(managed_python)
    assert env["PATH"].split(os.pathsep)[0] == os.fspath(
        venv_python_path(venv).parent
    )
    assert "GITHUB_TOKEN" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PARENT_ONLY" not in env
    assert env["HOME"] != os.environ.get("HOME")


def test_requirement_files_are_root_only_and_ordered(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    for name in (
        "requirements-test.txt",
        "requirements-test-extra.txt",
        "requirements-dev.txt",
        "test-requirements.txt",
        "requirements-prod.txt",
    ):
        (repository / name).write_text("", encoding="utf-8")
    (repository / "requirements-test-directory").mkdir()
    nested = repository / "nested"
    nested.mkdir()
    (nested / "requirements-test.txt").write_text("", encoding="utf-8")

    discovered = discover_requirement_files(repository)

    assert [path.name for path in discovered] == [
        "requirements-test-extra.txt",
        "requirements-test.txt",
        "requirements-dev.txt",
        "test-requirements.txt",
    ]


def test_install_plan_uses_editable_extras_then_requirements_then_pytest(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    (repository / "test-requirements.txt").write_text("", encoding="utf-8")
    python_path = tmp_path / "venv" / "bin" / "python"

    plan = build_install_plan(repository, python_path)

    assert [step.name for step in plan.editable] == [
        "editable:test",
        "editable:tests",
        "editable:testing",
        "editable:dev",
        "editable:plain",
    ]
    assert [step.command[-2:] for step in plan.editable] == [
        ("-e", ".[test]"),
        ("-e", ".[tests]"),
        ("-e", ".[testing]"),
        ("-e", ".[dev]"),
        ("-e", "."),
    ]
    assert [step.name for step in plan.requirements] == [
        "requirements:requirements-test.txt",
        "requirements:test-requirements.txt",
    ]
    assert [step.command[-2:] for step in plan.requirements] == [
        ("-r", "requirements-test.txt"),
        ("-r", "test-requirements.txt"),
    ]
    assert plan.pytest.command[-1] == "pytest"
    assert all(
        step.command[:5]
        == ("uv", "pip", "install", "--python", os.fspath(python_path))
        for step in (*plan.editable, *plan.requirements, plan.pytest)
    )


def test_install_stops_after_first_editable_and_adds_requirements(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    (repository / "requirements-dev.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _fail(stderr="no test extra"),
            _ok("tests extra installed\n"),
            _ok("test requirements installed\n"),
            _ok("dev requirements installed\n"),
            _ok("pytest installed\n"),
        ]
    )

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={"PATH": "/usr/bin"},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 60,
        command_runner=runner,
    )

    assert result.succeeded
    assert result.successful_install == "editable:tests"
    assert [attempt.name for attempt in result.attempts] == [
        "editable:test",
        "editable:tests",
        "requirements:requirements-test.txt",
        "requirements:requirements-dev.txt",
        "pytest",
    ]
    assert result.log_tail == "tests extra installed"
    assert all(call.cwd == os.fspath(repository) for call in runner.calls)


def test_install_fails_when_every_editable_attempt_fails(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner([_fail(stderr=f"failure {index}") for index in range(5)])

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 60,
        command_runner=runner,
    )

    assert not result.succeeded
    assert result.successful_install is None
    assert len(result.attempts) == 5
    assert all(attempt.phase == "editable" for attempt in result.attempts)
    assert runner.remaining() == 0


def test_install_timeout_aborts_setup_immediately(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner([_timeout(stderr="build hung")])

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 60,
        command_runner=runner,
    )

    assert not result.succeeded
    assert result.successful_install is None
    assert len(result.attempts) == 1
    assert result.attempts[0].timed_out
    assert len(runner.calls) == 1


def test_install_failure_keeps_the_successful_editable_attempt(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _ok(),
            _fail(stderr="bad requirements file"),
        ]
    )

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 60,
        command_runner=runner,
    )

    assert not result.succeeded
    assert result.successful_install == "editable:test"
    assert [attempt.name for attempt in result.attempts] == [
        "editable:test",
        "requirements:requirements-test.txt",
    ]
    assert result.log_tail == "bad requirements file"


class MutableClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_total_setup_deadline_prevents_the_next_command(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    plan = build_install_plan(repository, "/venv/bin/python")
    clock = MutableClock()

    def slow_success(command: Sequence[str]) -> CommandResult:
        clock.value = 200.0
        return CommandResult(
            command=list(command),
            return_code=0,
            duration_seconds=0.01,
        )

    runner = SequenceRunner([slow_success])
    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=100.0,
        command_runner=runner,
        clock=clock,
    )

    assert not result.succeeded
    assert result.successful_install == "editable:test"
    assert len(result.attempts) == 2
    assert not result.attempts[0].deadline_exceeded
    assert result.attempts[1].deadline_exceeded
    assert result.attempts[1].name == "pytest"
    assert len(runner.calls) == 1


def test_python_install_time_is_not_charged_to_the_setup_deadline(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    clock = MutableClock()

    def slow_python_install(command: Sequence[str]) -> CommandResult:
        clock.value = 1_000.0
        return CommandResult(
            command=list(command),
            return_code=0,
            duration_seconds=1_000.0,
        )

    runner = SequenceRunner(
        [
            _ok("uv 0.11.32\n"),
            slow_python_install,
            _ok(),
            _ok("3.11.9\n"),
            _ok(),
            _ok(),
        ]
    )

    with entry_environment(
        entry_id="acme/widget#42",
        repo_path=repository,
        workspace_dir=tmp_path / "workspace",
        setup_deadline_seconds=10,
        command_runner=runner,
        clock=clock,
    ) as environment:
        assert environment.summary.python_version == "3.11.9"

    assert runner.calls[1].timeout is None
    assert runner.calls[2].command[1:2] == ["venv"]
    assert runner.calls[2].timeout == 10


def test_install_log_tail_is_capped(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    plan = build_install_plan(repository, "/venv/bin/python")
    noisy = "\n".join(f"line {index}" for index in range(100))
    runner = SequenceRunner([_ok(stdout=noisy), _ok()])

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 60,
        command_runner=runner,
    )

    assert result.succeeded
    assert result.attempts[0].log_truncated
    assert len(result.attempts[0].log_tail.splitlines()) == 40
    assert result.attempts[0].log_tail.endswith("line 99")


def test_entry_environment_creates_isolated_paths_and_cleans_up(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    runner = _environment_success_runner()

    with entry_environment(
        entry_id="acme/widget#42",
        repo_path=repository,
        workspace_dir=workspace,
        command_runner=runner,
    ) as environment:
        assert environment.repo_path == repository
        assert environment.home_path.is_dir()
        assert environment.temp_dir.is_dir()
        assert environment.summary.uv_version == "uv 0.11.32 (deadbeef)"
        assert environment.summary.python_version == "3.11.9"
        assert environment.summary.python_install is not None
        assert environment.summary.venv is not None
        assert environment.summary.venv.succeeded
        assert environment.summary.install is not None
        assert environment.summary.install.successful_install == "editable:test"
        assert environment.summary.cleanup_errors == []
        assert environment.subprocess_env["HOME"] == os.fspath(
            environment.home_path
        )
        assert environment.subprocess_env["UV_PYTHON_INSTALL_DIR"] == os.fspath(
            environment.python_install_dir
        )
        assert environment.python_install_dir == workspace / ".uv-python"
        assert "GITHUB_TOKEN" not in environment.subprocess_env

        test_result = environment.run_test_command(["pytest", "-q"])
        assert test_result.succeeded
        assert test_result.stdout == "tests passed\n"

        environment_root = environment.environment_root

    assert not environment_root.exists()
    assert not repository.exists()
    assert runner.calls[1].command == ["uv", "python", "install", "3.11"]
    assert runner.calls[1].timeout is None
    assert runner.calls[-1].command == ["pytest", "-q"]
    assert runner.calls[-1].timeout == 120
    assert runner.calls[-1].env["HOME"] == os.fspath(
        environment.home_path
    )


def test_environment_creation_failure_cleans_clone_and_environment(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    runner = SequenceRunner(
        [
            _ok("uv 0.11.32\n"),
            _ok(),
            _ok(),
            _ok("3.11.9\n"),
            *[_fail(stderr=f"extra failed {index}") for index in range(5)],
        ]
    )

    with pytest.raises(EnvironmentSetupError) as exc_info:
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=workspace,
            command_runner=runner,
        )

    error = exc_info.value
    assert error.status == "could_not_install"
    assert error.infrastructure is False
    assert error.reason == "could_not_install"
    assert error.summary.install is not None
    assert not error.summary.install.succeeded
    assert len(error.summary.install.attempts) == 5
    assert error.summary.cleanup_errors == []
    assert not repository.exists()
    assert not (workspace / ENVIRONMENT_DIR_NAME / "acme__widget__42").exists()


def test_venv_timeout_is_a_setup_error_and_cleans_up(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    runner = SequenceRunner(
        [
            _ok("uv 0.11.32\n"),
            _ok(),
            _timeout(stderr="venv hung"),
        ]
    )

    with pytest.raises(EnvironmentSetupError) as exc_info:
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=workspace,
            command_runner=runner,
        )

    error = exc_info.value
    assert error.reason == "venv_failed"
    assert error.status == "errored"
    assert error.infrastructure is True
    assert error.summary.venv is not None
    assert error.summary.venv.timed_out
    assert not repository.exists()
    assert not (workspace / ENVIRONMENT_DIR_NAME / "acme__widget__42").exists()


def test_missing_uv_is_an_infrastructure_error(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    runner = SequenceRunner([FileNotFoundError("uv")])

    with pytest.raises(EnvironmentSetupError) as exc_info:
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=tmp_path / "workspace",
            command_runner=runner,
        )

    error = exc_info.value
    assert error.reason == "uv_unavailable"
    assert error.status == "errored"
    assert error.infrastructure is True
    assert error.summary.uv_version is None
    assert not repository.exists()


def test_environment_setup_keyboard_interrupt_still_cleans_up(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    workspace = tmp_path / "workspace"
    runner = SequenceRunner([KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=workspace,
            command_runner=runner,
        )

    assert not repository.exists()
    assert not (workspace / ENVIRONMENT_DIR_NAME / "acme__widget__42").exists()


def test_test_command_timeout_is_returned_not_raised(tmp_path: Path) -> None:
    repository = _repo(tmp_path)
    runner = _environment_success_runner()
    runner._outcomes[-1] = _timeout(stderr="agent tests hung")

    with entry_environment(
        entry_id="acme/widget#42",
        repo_path=repository,
        workspace_dir=tmp_path / "workspace",
        command_runner=runner,
    ) as environment:
        result = environment.run_test_command(["pytest"], timeout_seconds=3)

    assert result.timed_out
    assert result.return_code == -9
    assert not result.succeeded
    assert not repository.exists()


def test_cleanup_errors_are_recorded_without_masking_the_result(
    tmp_path: Path,
) -> None:
    repository = _repo(tmp_path)
    environment = EntryEnvironment.create(
        entry_id="acme/widget#42",
        repo_path=repository,
        workspace_dir=tmp_path / "workspace",
        command_runner=_environment_success_runner(),
    )

    with patch(
        "swe_agent.benchmark.environment.shutil.rmtree",
        side_effect=OSError("busy"),
    ):
        environment.close()

    assert len(environment.summary.cleanup_errors) == 2
    assert all(
        "Could not remove" in error
        for error in environment.summary.cleanup_errors
    )


def test_stale_workspace_sweep_removes_only_known_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    stale_environment = workspace / ENVIRONMENT_DIR_NAME / "old-entry"
    stale_environment.mkdir(parents=True)
    (stale_environment / "venv").mkdir()
    stale_clone = workspace / "acme__widget"
    stale_clone.mkdir()
    managed_python = workspace / ".uv-python"
    managed_python.mkdir()
    unrelated = workspace / "keep-me"
    unrelated.mkdir()

    errors = sweep_stale_workspaces(
        workspace,
        clone_paths=[stale_clone],
    )

    assert errors == []
    assert not (workspace / ENVIRONMENT_DIR_NAME).exists()
    assert not stale_clone.exists()
    assert managed_python.exists()
    assert unrelated.exists()
    assert sweep_stale_workspaces(tmp_path / "missing") == []


def test_ensure_python_installed_has_no_entry_timeout(tmp_path: Path) -> None:
    runner = SequenceRunner([_ok()])

    result = ensure_python_installed(
        "uv",
        "3.11",
        cwd=tmp_path,
        env={"PATH": "/usr/bin"},
        command_runner=runner,
    )

    assert result.succeeded
    assert runner.calls[0].command == ["uv", "python", "install", "3.11"]
    assert runner.calls[0].timeout is None


def test_subprocess_runner_starts_new_session_and_captures_output(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 123
    process.returncode = 0
    process.communicate.return_value = ("stdout text", "stderr text")

    with patch(
        "swe_agent.benchmark.environment.subprocess.Popen",
        return_value=process,
    ) as popen:
        result = SubprocessRunner().run(
            ["echo", "hello"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin"},
            timeout=5,
        )

    assert result.succeeded
    assert result.stdout == "stdout text"
    assert result.stderr == "stderr text"
    if os.name == "posix":
        assert popen.call_args.kwargs["start_new_session"] is True


def test_subprocess_runner_kills_the_process_group_on_timeout(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 456
    process.returncode = -15
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sleep"], timeout=1),
        ("partial stdout", "partial stderr"),
    ]

    with patch(
        "swe_agent.benchmark.environment.subprocess.Popen",
        return_value=process,
    ), patch("swe_agent.benchmark.environment.os.killpg") as killpg:
        result = SubprocessRunner().run(
            ["sleep", "5"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin"},
            timeout=1,
        )

    assert result.timed_out
    assert not result.succeeded
    assert result.stdout == "partial stdout"
    assert killpg.call_args_list == [
        ((456, signal.SIGTERM),),
    ]


def test_subprocess_runner_kills_the_group_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 789
    process.communicate.side_effect = KeyboardInterrupt()
    process.wait.return_value = 0

    with patch(
        "swe_agent.benchmark.environment.subprocess.Popen",
        return_value=process,
    ), patch("swe_agent.benchmark.environment.os.killpg") as killpg:
        with pytest.raises(KeyboardInterrupt):
            SubprocessRunner().run(
                ["pytest"],
                cwd=tmp_path,
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

    assert killpg.call_args_list == [
        ((789, signal.SIGKILL),),
    ]
    process.wait.assert_called_once()


def test_environment_create_rejects_nonpositive_timeouts(tmp_path: Path) -> None:
    repository = _repo(tmp_path)

    with pytest.raises(ValueError, match="install_timeout_seconds"):
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=tmp_path / "workspace",
            install_timeout_seconds=0,
            command_runner=SequenceRunner([]),
        )
    with pytest.raises(ValueError, match="test_timeout_seconds"):
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=tmp_path / "workspace",
            test_timeout_seconds=0,
            command_runner=SequenceRunner([]),
        )
    with pytest.raises(ValueError, match="setup_deadline_seconds"):
        EntryEnvironment.create(
            entry_id="acme/widget#42",
            repo_path=repository,
            workspace_dir=tmp_path / "workspace",
            setup_deadline_seconds=0,
            command_runner=SequenceRunner([]),
        )

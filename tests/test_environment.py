import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from unittest.mock import patch

import pytest

from swe_agent.benchmark.environment import (
    ENVIRONMENT_DIR_NAME,
    EnvironmentSetupError,
    CommandResult,
    EntryEnvironment,
    build_install_plan,
    build_subprocess_env,
    discover_requirement_files,
    ensure_python_installed,
    entry_environment,
    execute_install_plan,
    has_packaging_files,
    prepare_test_command,
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
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    return repository


def _environment_success_runner() -> SequenceRunner:
    return SequenceRunner(
        [
            _ok("uv 0.11.32 (deadbeef)\n"),
            _ok(),
            _ok(),
            _ok("3.11.9\n"),
            _ok("pytest install\n"),
            _ok("editable test install\n"),
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
    monkeypatch.setenv("PIP_INDEX_URL", "https://private.example/simple")
    monkeypatch.setenv("PIP_CACHE_DIR", "/parent/pip-cache")
    monkeypatch.setenv("UV_CACHE_DIR", "/parent/uv-cache")
    monkeypatch.setenv("UV_INDEX_URL", "https://private.example/simple")

    home = tmp_path / "home"
    temp_dir = tmp_path / "tmp"
    venv = tmp_path / "venv"
    managed_python = tmp_path / "uv-python"
    cache = tmp_path / "cache"
    env = build_subprocess_env(
        home,
        temp_dir,
        venv_path=venv,
        python_install_dir=managed_python,
        cache_dir=cache,
    )

    assert env["HOME"] == os.fspath(home)
    assert env["TMPDIR"] == os.fspath(temp_dir)
    assert env["TEMP"] == os.fspath(temp_dir)
    assert env["TMP"] == os.fspath(temp_dir)
    assert env["LANG"] == "C.UTF-8"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CI"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["VIRTUAL_ENV"] == os.fspath(venv)
    assert env["UV_PYTHON_INSTALL_DIR"] == os.fspath(managed_python)
    assert env["UV_CACHE_DIR"] == os.fspath(cache / "uv")
    assert env["PIP_CACHE_DIR"] == os.fspath(cache / "pip")
    assert env["PATH"].split(os.pathsep)[0] == os.fspath(
        venv_python_path(venv).parent
    )
    assert "GITHUB_TOKEN" not in env
    assert "GOOGLE_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "PARENT_ONLY" not in env
    assert "PIP_INDEX_URL" not in env
    assert "UV_INDEX_URL" not in env
    assert env["HOME"] != os.environ.get("HOME")


def test_requirement_files_follow_validator_order_and_include_nested(
    tmp_path: Path,
) -> None:
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
    nested = repository / "requirements"
    nested.mkdir()
    for name in ("test.txt", "tests.txt", "test-extra.txt"):
        (nested / name).write_text("", encoding="utf-8")
    unrelated = repository / "nested"
    unrelated.mkdir()
    (unrelated / "requirements-test.txt").write_text("", encoding="utf-8")

    discovered = discover_requirement_files(repository)

    assert [
        path.relative_to(repository).as_posix() for path in discovered
    ] == [
        "requirements-test.txt",
        "requirements-dev.txt",
        "test-requirements.txt",
        "requirements/test.txt",
        "requirements/tests.txt",
        "requirements-test-extra.txt",
        "requirements/test-extra.txt",
    ]


def test_install_plan_puts_pytest_first_and_uses_relative_requirements(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    (repository / "test-requirements.txt").write_text("", encoding="utf-8")
    nested = repository / "requirements"
    nested.mkdir()
    (nested / "tests.txt").write_text("", encoding="utf-8")
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
        "requirements:requirements/tests.txt",
    ]
    assert [step.command[-2:] for step in plan.requirements] == [
        ("-r", "requirements-test.txt"),
        ("-r", "test-requirements.txt"),
        ("-r", "requirements/tests.txt"),
    ]
    assert plan.pytest.name == "pytest"
    assert plan.pytest.command[-1] == "pytest"
    assert plan.editable_skipped is False
    assert all(
        step.command[:5]
        == ("uv", "pip", "install", "--python", os.fspath(python_path))
        for step in (*plan.editable, *plan.requirements, plan.pytest)
    )


def test_install_plan_skips_editable_without_packaging_metadata(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "module.py").write_text("value = 1\n", encoding="utf-8")

    assert has_packaging_files(repository) is False
    plan = build_install_plan(repository, "/venv/bin/python")

    assert plan.editable == ()
    assert plan.editable_skipped is True
    assert plan.pytest.name == "pytest"


@pytest.mark.parametrize("filename", ["pyproject.toml", "setup.py", "setup.cfg"])
def test_has_packaging_files_accepts_supported_metadata(
    tmp_path: Path,
    filename: str,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / filename).write_text("", encoding="utf-8")

    assert has_packaging_files(repository) is True
    assert build_install_plan(repository, "/venv/bin/python").editable


def test_install_stops_after_first_editable_and_adds_requirements(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    (repository / "requirements-dev.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _ok("pytest installed\n"),
            _fail(stderr="no test extra"),
            _ok("tests extra installed\n"),
            _ok("test requirements installed\n"),
            _ok("dev requirements installed\n"),
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
        "pytest",
        "editable:test",
        "editable:tests",
        "requirements:requirements-test.txt",
        "requirements:requirements-dev.txt",
    ]
    assert result.log_tail == "tests extra installed"
    assert all(call.cwd == os.fspath(repository) for call in runner.calls)
    assert runner.calls[0].command[-1] == "pytest"


def test_install_fails_when_every_editable_attempt_fails(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [_ok("pytest installed\n"), *[_fail(stderr=f"failure {index}") for index in range(5)]]
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
    assert result.successful_install is None
    assert len(result.attempts) == 6
    assert result.attempts[0].name == "pytest"
    assert all(
        attempt.phase == "editable" for attempt in result.attempts[1:]
    )
    assert runner.remaining() == 0


def test_missing_uv_extra_is_not_a_successful_editable_install(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _ok("pytest installed\n"),
            _ok(
                stdout=(
                    "warning: demo does not have an extra named `test`\n"
                    + "\n".join(f"noise {index}" for index in range(100))
                )
            ),
            _ok(stderr="warning: demo does not have an extra named `tests`\n"),
            _ok(stderr="warning: demo does not have an extra named `testing`\n"),
            _ok(stderr="warning: demo does not have an extra named `dev`\n"),
            _ok("plain editable installed\n"),
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

    assert result.succeeded
    assert result.successful_install == "editable:plain"
    assert result.attempts[1].name == "editable:test"
    assert result.attempts[1].succeeded
    assert result.attempts[1].accepted is False
    assert result.attempts[1].missing_extra is True
    assert result.attempts[1].effective_success is False


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


def test_requirement_failure_is_recorded_but_setup_continues(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    (repository / "requirements-dev.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _ok("pytest installed\n"),
            _ok("editable installed\n"),
            _fail(stderr="bad requirements file"),
            _ok("dev requirements installed\n"),
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

    assert result.succeeded
    assert result.successful_install == "editable:test"
    assert result.failed_requirements == [
        "requirements:requirements-test.txt"
    ]
    assert [attempt.name for attempt in result.attempts] == [
        "pytest",
        "editable:test",
        "requirements:requirements-test.txt",
        "requirements:requirements-dev.txt",
    ]
    assert result.log_tail == "bad requirements file"


def test_requirement_timeout_is_recorded_but_setup_continues(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
    (repository / "requirements-test.txt").write_text("", encoding="utf-8")
    plan = build_install_plan(repository, "/venv/bin/python")
    runner = SequenceRunner(
        [
            _ok("pytest installed\n"),
            _ok("editable installed\n"),
            _timeout(stderr="requirements hung"),
        ]
    )

    result = execute_install_plan(
        plan,
        repo_path=repository,
        env={},
        install_timeout_seconds=600,
        deadline=time.monotonic() + 1_200,
        command_runner=runner,
    )

    assert result.succeeded
    assert result.failed_requirements == [
        "requirements:requirements-test.txt"
    ]
    assert result.attempts[-1].timed_out
    assert not result.attempts[-1].effective_success


class MutableClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_total_setup_deadline_prevents_the_next_command(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
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
    assert result.successful_install is None
    assert len(result.attempts) == 2
    assert not result.attempts[0].deadline_exceeded
    assert result.attempts[0].name == "pytest"
    assert result.attempts[1].deadline_exceeded
    assert result.attempts[1].name == "editable:test"
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
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\nversion = '0.1'\n",
        encoding="utf-8",
    )
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
        assert environment.cache_dir == workspace / ".cache"
        assert environment.subprocess_env["UV_CACHE_DIR"] == os.fspath(
            workspace / ".cache" / "uv"
        )
        assert environment.subprocess_env["PIP_CACHE_DIR"] == os.fspath(
            workspace / ".cache" / "pip"
        )
        assert "GITHUB_TOKEN" not in environment.subprocess_env

        test_result = environment.run_test_command(["pytest", "-q"])
        assert test_result.succeeded
        assert test_result.stdout == "tests passed\n"

        environment_root = environment.environment_root
        venv_path = environment.venv_path

    assert not environment_root.exists()
    assert not repository.exists()
    assert environment.cache_dir.exists()
    assert runner.calls[1].command == ["uv", "python", "install", "3.11"]
    assert runner.calls[1].timeout is None
    assert runner.calls[-1].command == [
        os.fspath(venv_python_path(venv_path)),
        "-m",
        "pytest",
        "-q",
    ]
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
            _ok("pytest installed\n"),
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
    assert len(error.summary.install.attempts) == 6
    assert error.summary.install.attempts[0].name == "pytest"
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


def test_cleanup_records_every_exception_without_masking_the_original(
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
        side_effect=RuntimeError("cleanup exploded"),
    ):
        with pytest.raises(ValueError, match="original failure"):
            try:
                raise ValueError("original failure")
            finally:
                environment.close()

    assert len(environment.summary.cleanup_errors) == 2
    assert all(
        "RuntimeError: cleanup exploded" in error
        for error in environment.summary.cleanup_errors
    )


def test_prepare_test_command_binds_python_and_pytest_to_the_venv(
    tmp_path: Path,
) -> None:
    python = tmp_path / "venv" / "bin" / "python"

    assert prepare_test_command(["pytest", "-q"], python) == [
        os.fspath(python),
        "-m",
        "pytest",
        "-q",
    ]
    assert prepare_test_command(["python", "-m", "pytest"], python) == [
        os.fspath(python),
        "-m",
        "pytest",
    ]
    assert prepare_test_command(["python3", "script.py"], python) == [
        os.fspath(python),
        "script.py",
    ]
    assert prepare_test_command(["/usr/bin/python", "script.py"], python) == [
        "/usr/bin/python",
        "script.py",
    ]


def test_stale_workspace_sweep_removes_only_known_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    stale_environment = workspace / ENVIRONMENT_DIR_NAME / "old-entry"
    stale_environment.mkdir(parents=True)
    (stale_environment / "venv").mkdir()
    stale_clone = workspace / "acme__widget"
    stale_clone.mkdir()
    managed_python = workspace / ".uv-python"
    managed_python.mkdir()
    cache = workspace / ".cache" / "uv"
    cache.mkdir(parents=True)
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
    assert cache.exists()
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

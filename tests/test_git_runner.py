from pathlib import Path
from typing import Mapping, Sequence

import pytest

from swe_agent.git_runner import DEFAULT_GIT_TIMEOUT_SECONDS, GitRunner
from swe_agent.process import CommandResult


class RecordingRunner:
    def __init__(
        self,
        result: CommandResult | BaseException | None = None,
    ) -> None:
        self.result = result or CommandResult(command=[], return_code=0)
        self.calls: list[tuple[list[str], str, dict[str, str], float | None]] = []

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandResult:
        self.calls.append(
            (list(command), str(cwd), dict(env), timeout)
        )
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result.model_copy(update={"command": list(command)})


def test_git_runner_uses_default_timeout_and_noninteractive_environment(
    tmp_path: Path,
) -> None:
    command_runner = RecordingRunner()
    git_runner = GitRunner(
        command_runner=command_runner,
        env={"PATH": "/usr/bin", "GIT_TERMINAL_PROMPT": "1"},
    )

    result = git_runner.run(["status", "--short"], cwd=tmp_path)

    assert result.succeeded
    command, cwd, env, timeout = command_runner.calls[0]
    assert command == ["git", "status", "--short"]
    assert cwd == str(tmp_path)
    assert timeout == DEFAULT_GIT_TIMEOUT_SECONDS
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_git_runner_can_override_timeout_per_command(tmp_path: Path) -> None:
    command_runner = RecordingRunner()
    git_runner = GitRunner(command_runner=command_runner, timeout_seconds=60)

    git_runner.run(["fetch"], cwd=tmp_path, timeout_seconds=7)

    assert command_runner.calls[0][3] == 7


def test_git_runner_reports_missing_git_without_raising(tmp_path: Path) -> None:
    command_runner = RecordingRunner(FileNotFoundError("git"))

    result = GitRunner(command_runner=command_runner).run(
        ["status"],
        cwd=tmp_path,
    )

    assert result.return_code is None
    assert not result.succeeded
    assert "FileNotFoundError" in result.stderr


def test_git_runner_rejects_empty_or_invalid_commands(tmp_path: Path) -> None:
    git_runner = GitRunner(command_runner=RecordingRunner())

    with pytest.raises(ValueError, match="must not be empty"):
        git_runner.run([], cwd=tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        GitRunner(command_runner=RecordingRunner(), timeout_seconds=0)
    with pytest.raises(ValueError, match="must be positive"):
        git_runner.run(["status"], cwd=tmp_path, timeout_seconds=-1)

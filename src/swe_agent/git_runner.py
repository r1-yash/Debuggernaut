"""Timed, process-group-safe execution of git commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from swe_agent.process import CommandResult, CommandRunner, SubprocessRunner


DEFAULT_GIT_TIMEOUT_SECONDS = 300.0
DEFAULT_PATCH_GIT_TIMEOUT_SECONDS = 60.0


class GitRunner:
    """Run git commands through the shared subprocess boundary."""

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        *,
        timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.timeout_seconds = timeout_seconds
        self._command_runner = command_runner or SubprocessRunner()
        self._env = dict(os.environ if env is None else env)
        self._env.setdefault("GIT_TERMINAL_PROMPT", "0")

    def run(
        self,
        arguments: Sequence[str],
        *,
        cwd: str | Path,
        timeout_seconds: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        """Run one git command and capture its output without raising."""
        if not arguments:
            raise ValueError("Git command arguments must not be empty.")
        timeout = (
            self.timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive.")
        command = ["git", *arguments]
        child_env = dict(self._env)
        if env is not None:
            child_env.update(env)
        child_env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            return self._command_runner.run(
                command,
                cwd=cwd,
                env=child_env,
                timeout=timeout,
            )
        except OSError as error:
            return CommandResult(
                command=command,
                return_code=None,
                stderr=f"{type(error).__name__}: {error}",
            )


def git_failure_message(label: str, result: CommandResult) -> str:
    """Return a concise error message for a failed git invocation."""
    if result.timed_out:
        suffix = f": {result.stderr.strip()}" if result.stderr.strip() else ""
        return f"{label} timed out{suffix}"
    detail = result.stderr.strip() or result.stdout.strip()
    suffix = f": {detail}" if detail else ""
    return f"{label} failed with exit code {result.return_code}{suffix}"

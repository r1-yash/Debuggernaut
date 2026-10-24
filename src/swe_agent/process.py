"""Shared subprocess execution with process-group timeouts."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from pydantic import BaseModel


_TERMINATION_GRACE_SECONDS = 5.0


class CommandResult(BaseModel):
    """The captured result of one subprocess invocation."""

    command: list[str]
    return_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Return whether the command exited successfully without timing out."""
        return not self.timed_out and self.return_code == 0


class CommandRunner(Protocol):
    """Minimal subprocess boundary used by environments and git commands."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandResult:
        """Run one command and return its captured result."""
        ...


class SubprocessRunner:
    """Run commands in a new session and kill the whole group on timeout."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        timeout: float | None,
    ) -> CommandResult:
        command_list = list(command)
        started = time.monotonic()

        popen_kwargs: dict[str, object] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:  # pragma: no cover - the benchmark host is expected to be POSIX.
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creationflags:
                popen_kwargs["creationflags"] = creationflags

        process = subprocess.Popen(
            command_list,
            cwd=os.fspath(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_process_group(process, force=False)
            try:
                stdout, stderr = process.communicate(
                    timeout=_TERMINATION_GRACE_SECONDS
                )
            except subprocess.TimeoutExpired:
                _signal_process_group(process, force=True)
                stdout, stderr = process.communicate()
        except BaseException:
            # Do not leave a test/build process running after Ctrl-C.
            _signal_process_group(process, force=True)
            try:
                process.wait(timeout=_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            raise

        return CommandResult(
            command=command_list,
            return_code=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=timed_out,
            duration_seconds=time.monotonic() - started,
        )


def _signal_process_group(
    process: subprocess.Popen[str],
    *,
    force: bool,
) -> None:
    if os.name == "posix":
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:  # pragma: no cover - defensive fallback.
            process.kill()
    else:  # pragma: no cover - POSIX benchmark host.
        if force:
            process.kill()
        else:
            process.terminate()

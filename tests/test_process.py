import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

from swe_agent.process import SubprocessRunner


def test_subprocess_runner_starts_new_session_and_captures_output(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 123
    process.returncode = 0
    process.communicate.return_value = ("stdout text", "stderr text")

    with patch(
        "swe_agent.process.subprocess.Popen",
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
        "swe_agent.process.subprocess.Popen",
        return_value=process,
    ), patch("swe_agent.process.os.killpg") as killpg:
        result = SubprocessRunner().run(
            ["sleep", "5"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin"},
            timeout=1,
        )

    assert result.timed_out
    assert not result.succeeded
    assert result.stdout == "partial stdout"
    assert killpg.call_args_list == [call(456, signal.SIGTERM)]


def test_subprocess_runner_force_kills_the_process_group_after_grace(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 654
    process.returncode = -signal.SIGKILL
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd=["sleep"], timeout=1),
        subprocess.TimeoutExpired(cmd=["sleep"], timeout=5),
        ("", "still running"),
    ]

    with patch(
        "swe_agent.process.subprocess.Popen",
        return_value=process,
    ), patch("swe_agent.process.os.killpg") as killpg:
        result = SubprocessRunner().run(
            ["sleep", "30"],
            cwd=tmp_path,
            env={"PATH": "/usr/bin"},
            timeout=1,
        )

    assert result.timed_out
    assert result.return_code == -signal.SIGKILL
    assert result.stderr == "still running"
    assert killpg.call_args_list == [
        call(654, signal.SIGTERM),
        call(654, signal.SIGKILL),
    ]


def test_subprocess_runner_kills_the_group_on_keyboard_interrupt(
    tmp_path: Path,
) -> None:
    process = Mock()
    process.pid = 789
    process.communicate.side_effect = KeyboardInterrupt()
    process.wait.return_value = 0

    with patch(
        "swe_agent.process.subprocess.Popen",
        return_value=process,
    ), patch("swe_agent.process.os.killpg") as killpg:
        with pytest.raises(KeyboardInterrupt):
            SubprocessRunner().run(
                ["pytest"],
                cwd=tmp_path,
                env={"PATH": "/usr/bin"},
                timeout=5,
            )

    assert killpg.call_args_list == [call(789, signal.SIGKILL)]
    process.wait.assert_called_once()


def _process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    status = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True,
        text=True,
        check=False,
    )
    return status.returncode != 0 or status.stdout.strip().startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups only")
def test_real_grandchild_is_reaped_after_sigkill_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("swe_agent.process._TERMINATION_GRACE_SECONDS", 0.2)
    pid_file = tmp_path / "pids.txt"
    ready_file = tmp_path / "grandchild-ready.txt"
    grandchild_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready_file)!r}).write_text('ready')\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"child = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
        "for _ in range(100):\n"
        f"    if Path({str(ready_file)!r}).exists():\n"
        "        break\n"
        "    time.sleep(0.05)\n"
        f"Path({str(pid_file)!r}).write_text(f'{{os.getpid()}} {{child.pid}}')\n"
        "time.sleep(60)\n"
    )

    result = SubprocessRunner().run(
        [sys.executable, "-c", parent_code],
        cwd=tmp_path,
        env=dict(os.environ),
        timeout=1.5,
    )

    assert result.timed_out
    assert result.return_code != 0
    assert pid_file.exists()
    parent_pid, grandchild_pid = (
        int(part) for part in pid_file.read_text().split()
    )
    print(
        "\nreal process-group test:",
        f"timed_out={result.timed_out}",
        f"return_code={result.return_code}",
        f"parent_pid={parent_pid}",
        f"grandchild_pid={grandchild_pid}",
    )

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_is_gone(parent_pid) and _process_is_gone(grandchild_pid):
            break
        time.sleep(0.05)

    assert _process_is_gone(parent_pid)
    assert _process_is_gone(grandchild_pid)

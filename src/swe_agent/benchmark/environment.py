"""Isolated per-entry Python environments for benchmark execution.

Each benchmark entry gets its own virtual environment and throwaway ``HOME``.
The environment layer never mutates the harness process environment and never
passes parent secrets to child processes.

``uv python install 3.11`` is a one-time host setup step.  Python download time
is deliberately excluded from the per-entry setup deadline; the manager invokes
the command idempotently before starting its setup clock so a cold cache does
not consume an entry's installation budget.  Managed Python installations are
kept in a shared workspace directory via ``UV_PYTHON_INSTALL_DIR`` so the
throwaway per-entry ``HOME`` does not force a Python download for every entry.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, Field


BENCHMARK_PYTHON_VERSION = "3.11"
DEFAULT_INSTALL_TIMEOUT_SECONDS = 600.0
DEFAULT_TEST_TIMEOUT_SECONDS = 120.0
DEFAULT_SETUP_DEADLINE_SECONDS = 1_800.0
OUTPUT_TAIL_MAX_LINES = 40
OUTPUT_TAIL_MAX_CHARS = 8_000
ENVIRONMENT_DIR_NAME = ".environments"

_EDITABLE_EXTRAS = ("test", "tests", "testing", "dev")
_REQUIREMENT_PATTERNS = (
    "requirements-test*",
    "requirements-dev*",
    "test-requirements*",
)
_TERMINATION_GRACE_SECONDS = 5.0

# Explicit allowlist.  HOME and the temp variables are always replaced with
# per-entry directories; the parent values are intentionally never forwarded.
_ALLOWED_ENV_KEYS = frozenset(
    {
        "PATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }
)


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


class CommandAttempt(BaseModel):
    """A capped, serializable record of one setup or installation command."""

    name: str
    phase: Literal["python", "venv", "editable", "requirements", "pytest"]
    command: list[str]
    return_code: int | None
    succeeded: bool
    timed_out: bool
    deadline_exceeded: bool = False
    duration_seconds: float = 0.0
    log_tail: str = ""
    log_truncated: bool = False


class InstallResult(BaseModel):
    """The outcome of the editable, requirements, and pytest install phases."""

    succeeded: bool
    successful_install: str | None = None
    attempts: list[CommandAttempt] = Field(default_factory=list)
    requirement_files: list[str] = Field(default_factory=list)
    log_tail: str = ""
    log_truncated: bool = False
    duration_seconds: float = 0.0


class EnvironmentSummary(BaseModel):
    """Machine-readable environment metadata for a run or baseline record."""

    python_version: str | None = None
    uv_version: str | None = None
    python_install: CommandAttempt | None = None
    venv: CommandAttempt | None = None
    install: InstallResult | None = None
    setup_duration_seconds: float = 0.0
    cleanup_errors: list[str] = Field(default_factory=list)


class EnvironmentSetupError(RuntimeError):
    """Raised when an entry's isolated environment cannot be prepared.

    ``summary`` is always populated so the caller can persist a partial
    record, including install logs and timing.  ``status`` distinguishes a
    genuine entry installation failure from infrastructure failures such as a
    missing ``uv`` binary or a timeout creating the virtual environment.
    """

    def __init__(
        self,
        message: str,
        *,
        summary: EnvironmentSummary,
        reason: str,
        status: Literal["could_not_install", "errored"] = "errored",
        infrastructure: bool = True,
    ) -> None:
        super().__init__(message)
        self.summary = summary
        self.reason = reason
        self.status = status
        self.infrastructure = infrastructure


class CommandRunner(Protocol):
    """Minimal subprocess boundary used by environments and tests."""

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


@dataclass(frozen=True)
class InstallStep:
    """One command in the deterministic installation recipe."""

    name: str
    phase: Literal["editable", "requirements", "pytest"]
    command: tuple[str, ...]


@dataclass(frozen=True)
class InstallPlan:
    """The ordered installation recipe for one repository."""

    editable: tuple[InstallStep, ...]
    requirements: tuple[InstallStep, ...]
    pytest: InstallStep
    requirement_files: tuple[str, ...]


def build_subprocess_env(
    home: str | Path,
    temp_dir: str | Path,
    *,
    venv_path: str | Path | None = None,
    python_install_dir: str | Path | None = None,
    parent_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal child environment for uv, Python, and tests.

    Only the explicit allowlist is copied from the parent.  ``HOME`` and all
    temporary-directory variables point at the entry's throwaway directories.
    Secrets such as ``GITHUB_TOKEN`` and ``GOOGLE_API_KEY`` are never included.
    ``python_install_dir`` points uv at shared managed-Python storage so a
    throwaway HOME does not force a Python download for every entry.
    """
    parent = os.environ if parent_env is None else parent_env
    home_path = Path(home).expanduser().absolute()
    temp_path = Path(temp_dir).expanduser().absolute()
    env = {
        key: parent[key]
        for key in _ALLOWED_ENV_KEYS
        if key in parent and parent[key]
    }
    env["PATH"] = parent.get("PATH") or os.defpath
    env["HOME"] = os.fspath(home_path)
    env["TMPDIR"] = os.fspath(temp_path)
    env["TEMP"] = os.fspath(temp_path)
    env["TMP"] = os.fspath(temp_path)
    env["PYTHONNOUSERSITE"] = "1"
    if python_install_dir is not None:
        env["UV_PYTHON_INSTALL_DIR"] = os.fspath(
            Path(python_install_dir).expanduser().absolute()
        )

    if venv_path is not None:
        venv = Path(venv_path).expanduser().absolute()
        env["VIRTUAL_ENV"] = os.fspath(venv)
        env["PATH"] = (
            os.fspath(_venv_bin_dir(venv)) + os.pathsep + env["PATH"]
        )
    return env


def venv_python_path(venv_path: str | Path) -> Path:
    """Return the Python executable inside a virtual environment."""
    venv = Path(venv_path)
    if os.name == "nt":  # pragma: no cover - POSIX benchmark host.
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def discover_requirement_files(repo_path: str | Path) -> tuple[Path, ...]:
    """Return root-level test/development requirement files in install order."""
    repository = Path(repo_path)
    discovered: list[Path] = []
    seen: set[str] = set()
    for pattern in _REQUIREMENT_PATTERNS:
        matches = sorted(
            (
                path
                for path in repository.glob(pattern)
                if path.is_file() and path.name not in seen
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
        for path in matches:
            seen.add(path.name)
            discovered.append(path)
    return tuple(discovered)


def build_install_plan(
    repo_path: str | Path,
    python_path: str | Path,
    *,
    uv_executable: str = "uv",
) -> InstallPlan:
    """Build the editable-first installation recipe for a repository.

    Editable extras are attempted in the order test, tests, testing, dev, then
    a plain editable install.  Requirement files are additive after the first
    successful editable install, followed by a mandatory pytest installation.
    """
    repository = Path(repo_path)
    prefix = (uv_executable, "pip", "install", "--python", os.fspath(python_path))

    editable: list[InstallStep] = []
    for extra in _EDITABLE_EXTRAS:
        editable.append(
            InstallStep(
                name=f"editable:{extra}",
                phase="editable",
                command=(*prefix, "-e", f".[{extra}]"),
            )
        )
    editable.append(
        InstallStep(
            name="editable:plain",
            phase="editable",
            command=(*prefix, "-e", "."),
        )
    )

    requirement_paths = discover_requirement_files(repository)
    requirements = tuple(
        InstallStep(
            name=f"requirements:{path.name}",
            phase="requirements",
            command=(*prefix, "-r", path.name),
        )
        for path in requirement_paths
    )
    pytest_step = InstallStep(
        name="pytest",
        phase="pytest",
        command=(*prefix, "pytest"),
    )
    return InstallPlan(
        editable=tuple(editable),
        requirements=requirements,
        pytest=pytest_step,
        requirement_files=tuple(path.name for path in requirement_paths),
    )


def ensure_python_installed(
    uv_executable: str,
    python_version: str,
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    command_runner: CommandRunner,
) -> CommandResult:
    """Run the one-time ``uv python install`` step outside entry deadlines.

    The command is idempotent and uses uv's cache after the first invocation.
    Its download time must not be charged to an individual benchmark entry.
    """
    return _safe_run(
        command_runner,
        [uv_executable, "python", "install", python_version],
        cwd=cwd,
        env=env,
        timeout=None,
    )


def execute_install_plan(
    plan: InstallPlan,
    *,
    repo_path: str | Path,
    env: Mapping[str, str],
    install_timeout_seconds: float,
    deadline: float,
    command_runner: CommandRunner,
    clock: Callable[[], float] = time.monotonic,
) -> InstallResult:
    """Execute an install plan, stopping at the first successful editable spec.

    A timeout aborts setup immediately.  Non-timeout failures move to the next
    editable extra.  Once an editable install succeeds, every discovered
    requirement file and pytest itself must install successfully.
    """
    if install_timeout_seconds <= 0:
        raise ValueError("install_timeout_seconds must be positive.")

    repository = Path(repo_path)
    started = clock()
    attempts: list[CommandAttempt] = []

    successful_step: InstallStep | None = None
    for step in plan.editable:
        attempt = _attempt_step(
            step,
            repo_path=repository,
            env=env,
            install_timeout_seconds=install_timeout_seconds,
            deadline=deadline,
            command_runner=command_runner,
            clock=clock,
        )
        attempts.append(attempt)
        if attempt.deadline_exceeded or attempt.timed_out:
            return _install_failure(
                attempts,
                plan,
                attempt,
                successful_install=None,
                duration_seconds=clock() - started,
            )
        if attempt.succeeded:
            successful_step = step
            break

    if successful_step is None:
        return _install_failure(
            attempts,
            plan,
            attempts[-1] if attempts else None,
            successful_install=None,
            duration_seconds=clock() - started,
        )

    for step in plan.requirements:
        attempt = _attempt_step(
            step,
            repo_path=repository,
            env=env,
            install_timeout_seconds=install_timeout_seconds,
            deadline=deadline,
            command_runner=command_runner,
            clock=clock,
        )
        attempts.append(attempt)
        if not attempt.succeeded:
            return _install_failure(
                attempts,
                plan,
                attempt,
                successful_install=successful_step.name,
                duration_seconds=clock() - started,
            )

    pytest_attempt = _attempt_step(
        plan.pytest,
        repo_path=repository,
        env=env,
        install_timeout_seconds=install_timeout_seconds,
        deadline=deadline,
        command_runner=command_runner,
        clock=clock,
    )
    attempts.append(pytest_attempt)
    if not pytest_attempt.succeeded:
        return _install_failure(
            attempts,
            plan,
            pytest_attempt,
            successful_install=successful_step.name,
            duration_seconds=clock() - started,
        )

    successful_attempt = _attempt_for_name(attempts, successful_step.name)
    return InstallResult(
        succeeded=True,
        successful_install=successful_step.name,
        attempts=attempts,
        requirement_files=list(plan.requirement_files),
        log_tail=successful_attempt.log_tail,
        log_truncated=successful_attempt.log_truncated,
        duration_seconds=clock() - started,
    )


class EntryEnvironment:
    """A disposable venv and HOME for one benchmark entry."""

    def __init__(
        self,
        *,
        entry_id: str,
        repo_path: Path,
        workspace_dir: Path,
        environment_root: Path,
        venv_path: Path,
        home_path: Path,
        temp_dir: Path,
        python_install_dir: Path,
        subprocess_env: dict[str, str],
        test_timeout_seconds: float,
        command_runner: CommandRunner,
        clock: Callable[[], float],
        cleanup_repo: bool,
        summary: EnvironmentSummary,
    ) -> None:
        self.entry_id = entry_id
        self.repo_path = repo_path
        self.workspace_dir = workspace_dir
        self.environment_root = environment_root
        self.venv_path = venv_path
        self.home_path = home_path
        self.temp_dir = temp_dir
        self.python_install_dir = python_install_dir
        self.subprocess_env = subprocess_env
        self.test_timeout_seconds = test_timeout_seconds
        self.summary = summary
        self._command_runner = command_runner
        self._clock = clock
        self._cleanup_repo = cleanup_repo
        self._closed = False
        self._setup_started: float | None = None

    @classmethod
    def create(
        cls,
        *,
        entry_id: str,
        repo_path: str | Path,
        workspace_dir: str | Path = "workspace",
        python_version: str = BENCHMARK_PYTHON_VERSION,
        uv_executable: str = "uv",
        install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
        test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
        python_install_dir: str | Path | None = None,
        command_runner: CommandRunner | None = None,
        clock: Callable[[], float] | None = None,
        cleanup_repo: bool = True,
    ) -> EntryEnvironment:
        """Create, install, and return one isolated entry environment."""
        _validate_positive(install_timeout_seconds, "install_timeout_seconds")
        _validate_positive(test_timeout_seconds, "test_timeout_seconds")
        _validate_positive(setup_deadline_seconds, "setup_deadline_seconds")
        if not entry_id.strip():
            raise ValueError("entry_id must not be empty.")
        if not python_version.strip():
            raise ValueError("python_version must not be empty.")

        repository = Path(repo_path).expanduser().absolute()
        if not repository.is_dir():
            raise ValueError(f"Repository path is not a directory: {repository}")

        workspace = Path(workspace_dir).expanduser().absolute()
        environment_root = workspace / ENVIRONMENT_DIR_NAME / _entry_slug(entry_id)
        venv_path = environment_root / "venv"
        home_path = environment_root / "home"
        temp_dir = environment_root / "tmp"
        managed_python_dir = (
            Path(python_install_dir).expanduser().resolve()
            if python_install_dir is not None
            else workspace / ".uv-python"
        )
        runner = command_runner or SubprocessRunner()
        current_clock = clock or time.monotonic
        summary = EnvironmentSummary()

        environment = cls(
            entry_id=entry_id,
            repo_path=repository,
            workspace_dir=workspace,
            environment_root=environment_root,
            venv_path=venv_path,
            home_path=home_path,
            temp_dir=temp_dir,
            python_install_dir=managed_python_dir,
            subprocess_env=build_subprocess_env(
                home_path,
                temp_dir,
                venv_path=venv_path,
                python_install_dir=managed_python_dir,
            ),
            test_timeout_seconds=test_timeout_seconds,
            command_runner=runner,
            clock=current_clock,
            cleanup_repo=cleanup_repo,
            summary=summary,
        )

        try:
            environment._prepare(
                python_version=python_version,
                uv_executable=uv_executable,
                install_timeout_seconds=install_timeout_seconds,
                setup_deadline_seconds=setup_deadline_seconds,
            )
        except EnvironmentSetupError:
            environment._finish_setup_timing()
            environment.close()
            raise
        except Exception as error:
            environment._finish_setup_timing()
            environment.close()
            raise EnvironmentSetupError(
                f"Environment setup failed: {error}",
                summary=summary,
                reason="unexpected",
            ) from error
        except BaseException:
            # KeyboardInterrupt/SystemExit must still remove the clone and venv.
            environment._finish_setup_timing()
            environment.close()
            raise
        return environment

    def _prepare(
        self,
        *,
        python_version: str,
        uv_executable: str,
        install_timeout_seconds: float,
        setup_deadline_seconds: float,
    ) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        stale_errors = _remove_paths(
            [self.environment_root],
            protected={self.workspace_dir},
        )
        if stale_errors:
            self.summary.cleanup_errors.extend(stale_errors)
            raise EnvironmentSetupError(
                "Could not clear a stale benchmark environment.",
                summary=self.summary,
                reason="stale_cleanup_failed",
            )
        self.environment_root.mkdir(parents=True)
        self.home_path.mkdir()
        self.temp_dir.mkdir()

        version_result = _safe_run(
            self._command_runner,
            [uv_executable, "--version"],
            cwd=self.environment_root,
            env=self.subprocess_env,
            timeout=install_timeout_seconds,
        )
        if not version_result.succeeded:
            raise EnvironmentSetupError(
                _command_failure_message("uv --version", version_result),
                summary=self.summary,
                reason="uv_unavailable",
            )
        self.summary.uv_version = (
            _first_line(version_result.stdout)
            or _first_line(version_result.stderr)
            or None
        )

        python_result = ensure_python_installed(
            uv_executable,
            python_version,
            cwd=self.environment_root,
            env=self.subprocess_env,
            command_runner=self._command_runner,
        )
        self.summary.python_install = _attempt_from_result(
            name=f"python:{python_version}",
            phase="python",
            command=python_result.command,
            result=python_result,
        )
        if not python_result.succeeded:
            raise EnvironmentSetupError(
                _command_failure_message("uv python install", python_result),
                summary=self.summary,
                reason="python_unavailable",
            )

        self._setup_started = self._clock()
        deadline = self._setup_started + setup_deadline_seconds
        venv_command = [
            uv_executable,
            "venv",
            "--python",
            python_version,
            os.fspath(self.venv_path),
        ]
        venv_attempt = _attempt_command(
            name="venv",
            phase="venv",
            command=venv_command,
            cwd=self.environment_root,
            env=self.subprocess_env,
            timeout_seconds=install_timeout_seconds,
            deadline=deadline,
            command_runner=self._command_runner,
            clock=self._clock,
        )
        self.summary.venv = venv_attempt
        if not venv_attempt.succeeded:
            self._finish_setup_timing()
            raise EnvironmentSetupError(
                "Could not create the benchmark virtual environment.",
                summary=self.summary,
                reason="venv_failed",
            )

        python_path = venv_python_path(self.venv_path)
        version_probe = [
            os.fspath(python_path),
            "-c",
            "import platform; print(platform.python_version())",
        ]
        probe_attempt = _attempt_command(
            name="python-version",
            phase="python",
            command=version_probe,
            cwd=self.environment_root,
            env=self.subprocess_env,
            timeout_seconds=install_timeout_seconds,
            deadline=deadline,
            command_runner=self._command_runner,
            clock=self._clock,
        )
        if not probe_attempt.succeeded:
            self._finish_setup_timing()
            raise EnvironmentSetupError(
                "Could not query the benchmark virtual environment's Python version.",
                summary=self.summary,
                reason="python_probe_failed",
            )
        self.summary.python_version = (
            _first_line(probe_attempt.log_tail) or None
        )

        plan = build_install_plan(
            self.repo_path,
            python_path,
            uv_executable=uv_executable,
        )
        install_result = execute_install_plan(
            plan,
            repo_path=self.repo_path,
            env=self.subprocess_env,
            install_timeout_seconds=install_timeout_seconds,
            deadline=deadline,
            command_runner=self._command_runner,
            clock=self._clock,
        )
        self.summary.install = install_result
        self._finish_setup_timing()
        if not install_result.succeeded:
            spawn_failure = any(
                attempt.return_code is None
                and not attempt.timed_out
                and not attempt.deadline_exceeded
                for attempt in install_result.attempts
            )
            if spawn_failure:
                raise EnvironmentSetupError(
                    "Could not launch a uv installation command.",
                    summary=self.summary,
                    reason="uv_unavailable",
                )
            raise EnvironmentSetupError(
                "No installation attempt succeeded.",
                summary=self.summary,
                reason="could_not_install",
                status="could_not_install",
                infrastructure=False,
            )

    def _finish_setup_timing(self) -> None:
        if self._setup_started is not None:
            self.summary.setup_duration_seconds = max(
                0.0, self._clock() - self._setup_started
            )

    def run_test_command(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        """Run a test command inside the isolated environment."""
        if self._closed:
            raise RuntimeError("The benchmark environment is already closed.")
        if not command:
            raise ValueError("Test command must not be empty.")
        timeout = (
            self.test_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        _validate_positive(timeout, "timeout_seconds")
        return _safe_run(
            self._command_runner,
            list(command),
            cwd=self.repo_path,
            env=self.subprocess_env,
            timeout=timeout,
        )

    def close(self) -> None:
        """Remove the clone and the entire per-entry environment."""
        if self._closed:
            return
        self._closed = True
        targets = [self.environment_root]
        if self._cleanup_repo:
            targets.append(self.repo_path)
        self.summary.cleanup_errors.extend(
            _remove_paths(targets, protected={self.workspace_dir})
        )

    def __enter__(self) -> EntryEnvironment:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@contextmanager
def entry_environment(
    *,
    entry_id: str,
    repo_path: str | Path,
    workspace_dir: str | Path = "workspace",
    python_version: str = BENCHMARK_PYTHON_VERSION,
    uv_executable: str = "uv",
    install_timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
    setup_deadline_seconds: float = DEFAULT_SETUP_DEADLINE_SECONDS,
    python_install_dir: str | Path | None = None,
    command_runner: CommandRunner | None = None,
    clock: Callable[[], float] | None = None,
    cleanup_repo: bool = True,
) -> Iterator[EntryEnvironment]:
    """Yield an isolated environment and always clean up on exit."""
    environment = EntryEnvironment.create(
        entry_id=entry_id,
        repo_path=repo_path,
        workspace_dir=workspace_dir,
        python_version=python_version,
        uv_executable=uv_executable,
        install_timeout_seconds=install_timeout_seconds,
        test_timeout_seconds=test_timeout_seconds,
        setup_deadline_seconds=setup_deadline_seconds,
        python_install_dir=python_install_dir,
        command_runner=command_runner,
        clock=clock,
        cleanup_repo=cleanup_repo,
    )
    try:
        yield environment
    finally:
        environment.close()


def sweep_stale_workspaces(
    workspace_dir: str | Path,
    *,
    clone_paths: Iterable[str | Path] = (),
) -> list[str]:
    """Remove environment roots and configured clone paths left by a crash.

    Only the environment directory and the explicitly supplied clone paths are
    touched.  Unrelated directories in the workspace are preserved.
    """
    workspace = Path(workspace_dir)
    if not workspace.exists():
        return []

    targets = [workspace / ENVIRONMENT_DIR_NAME, *clone_paths]
    return _remove_paths(targets, protected={workspace})


def _attempt_step(
    step: InstallStep,
    *,
    repo_path: Path,
    env: Mapping[str, str],
    install_timeout_seconds: float,
    deadline: float,
    command_runner: CommandRunner,
    clock: Callable[[], float],
) -> CommandAttempt:
    return _attempt_command(
        name=step.name,
        phase=step.phase,
        command=list(step.command),
        cwd=repo_path,
        env=env,
        timeout_seconds=install_timeout_seconds,
        deadline=deadline,
        command_runner=command_runner,
        clock=clock,
    )


def _attempt_command(
    *,
    name: str,
    phase: Literal["python", "venv", "editable", "requirements", "pytest"],
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    deadline: float,
    command_runner: CommandRunner,
    clock: Callable[[], float],
) -> CommandAttempt:
    remaining = deadline - clock()
    if remaining <= 0:
        return CommandAttempt(
            name=name,
            phase=phase,
            command=command,
            return_code=None,
            succeeded=False,
            timed_out=False,
            deadline_exceeded=True,
            log_tail="Per-entry setup deadline exceeded before this command.",
        )

    deadline_limited = remaining < timeout_seconds
    result = _safe_run(
        command_runner,
        command,
        cwd=cwd,
        env=env,
        timeout=min(timeout_seconds, remaining),
    )
    attempt = _attempt_from_result(
        name=name,
        phase=phase,
        command=command,
        result=result,
    )
    if deadline_limited and result.timed_out:
        attempt.deadline_exceeded = True
    return attempt


def _attempt_from_result(
    *,
    name: str,
    phase: Literal["python", "venv", "editable", "requirements", "pytest"],
    command: list[str],
    result: CommandResult,
) -> CommandAttempt:
    log_tail, log_truncated = _output_tail(result.stdout, result.stderr)
    return CommandAttempt(
        name=name,
        phase=phase,
        command=command,
        return_code=result.return_code,
        succeeded=result.succeeded,
        timed_out=result.timed_out,
        duration_seconds=result.duration_seconds,
        log_tail=log_tail,
        log_truncated=log_truncated,
    )


def _install_failure(
    attempts: list[CommandAttempt],
    plan: InstallPlan,
    failed_attempt: CommandAttempt | None,
    *,
    successful_install: str | None,
    duration_seconds: float,
) -> InstallResult:
    return InstallResult(
        succeeded=False,
        successful_install=successful_install,
        attempts=attempts,
        requirement_files=list(plan.requirement_files),
        log_tail=failed_attempt.log_tail if failed_attempt is not None else "",
        log_truncated=(
            failed_attempt.log_truncated if failed_attempt is not None else False
        ),
        duration_seconds=duration_seconds,
    )


def _attempt_for_name(
    attempts: list[CommandAttempt], name: str
) -> CommandAttempt:
    for attempt in attempts:
        if attempt.name == name:
            return attempt
    raise ValueError(f"Attempt not found: {name}")


def _safe_run(
    command_runner: CommandRunner,
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    timeout: float | None,
) -> CommandResult:
    command_list = list(command)
    try:
        return command_runner.run(
            command_list,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
    except OSError as error:
        return CommandResult(
            command=command_list,
            return_code=None,
            stderr=f"{type(error).__name__}: {error}",
            duration_seconds=0.0,
        )


def _command_failure_message(label: str, result: CommandResult) -> str:
    detail = _first_line(result.stderr) or _first_line(result.stdout)
    suffix = f": {detail}" if detail else ""
    return f"{label} failed{suffix}"


def _output_tail(stdout: str, stderr: str) -> tuple[str, bool]:
    combined = "\n".join(
        part for part in (stdout.strip(), stderr.strip()) if part
    )
    lines = combined.splitlines()
    truncated = len(lines) > OUTPUT_TAIL_MAX_LINES
    tail = "\n".join(lines[-OUTPUT_TAIL_MAX_LINES:])
    if len(tail) > OUTPUT_TAIL_MAX_CHARS:
        return tail[-OUTPUT_TAIL_MAX_CHARS:], True
    return tail, truncated


def _first_line(output: str) -> str:
    lines = output.strip().splitlines()
    return lines[0].strip() if lines else ""


def _venv_bin_dir(venv_path: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - POSIX benchmark host.
        return venv_path / "Scripts"
    return venv_path / "bin"


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


def _entry_slug(entry_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "__"
        for character in entry_id
    ).strip("._-")
    if not safe:
        raise ValueError("entry_id must contain at least one safe character.")
    return safe


def _validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _remove_paths(
    paths: Iterable[Path],
    *,
    protected: set[Path],
) -> list[str]:
    errors: list[str] = []
    protected_resolved = {path.resolve() for path in protected}
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if resolved in seen or resolved in protected_resolved:
            continue
        seen.add(resolved)
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
        except FileNotFoundError:
            continue
        except OSError as error:
            errors.append(f"Could not remove {path}: {error}")
    return errors

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
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Mapping, Sequence

from pydantic import BaseModel, Field

from swe_agent.process import CommandResult, CommandRunner, SubprocessRunner


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
_SCRIPT_REQUIREMENT_FILES = (
    "requirements-test.txt",
    "requirements-dev.txt",
    "test-requirements.txt",
    "requirements/test.txt",
    "requirements/tests.txt",
)
_NESTED_REQUIREMENT_PATTERNS = ("requirements/test*.txt",)
_PACKAGING_FILES = ("pyproject.toml", "setup.py", "setup.cfg")

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


class CommandAttempt(BaseModel):
    """A capped, serializable record of one setup or installation command."""

    name: str
    phase: Literal["python", "venv", "editable", "requirements", "pytest"]
    command: list[str]
    return_code: int | None
    succeeded: bool
    timed_out: bool
    accepted: bool = True
    missing_extra: bool = False
    deadline_exceeded: bool = False
    duration_seconds: float = 0.0
    log_tail: str = ""
    log_truncated: bool = False

    @property
    def effective_success(self) -> bool:
        """Return whether the command succeeded and counts as an install."""
        return self.succeeded and self.accepted


class InstallResult(BaseModel):
    """The outcome of the editable, requirements, and pytest install phases."""

    succeeded: bool
    successful_install: str | None = None
    editable_skipped: bool = False
    attempts: list[CommandAttempt] = Field(default_factory=list)
    requirement_files: list[str] = Field(default_factory=list)
    failed_requirements: list[str] = Field(default_factory=list)
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
    editable_skipped: bool = False


def build_subprocess_env(
    home: str | Path,
    temp_dir: str | Path,
    *,
    venv_path: str | Path | None = None,
    python_install_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    parent_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal child environment for uv, Python, and tests.

    Only the explicit allowlist is copied from the parent.  ``HOME`` and all
    temporary-directory variables point at the entry's throwaway directories.
    Secrets such as ``GITHUB_TOKEN`` and ``GOOGLE_API_KEY`` are never included.
    ``python_install_dir`` points uv at shared managed-Python storage so a
    throwaway HOME does not force a Python download for every entry.  Package
    caches point at a shared cache directory so they survive per-entry cleanup.
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
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CI"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    if python_install_dir is not None:
        env["UV_PYTHON_INSTALL_DIR"] = os.fspath(
            Path(python_install_dir).expanduser().absolute()
        )
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser().absolute()
        env["UV_CACHE_DIR"] = os.fspath(cache_path / "uv")
        env["PIP_CACHE_DIR"] = os.fspath(cache_path / "pip")

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


def prepare_test_command(
    command: Sequence[str],
    python_path: str | Path,
) -> list[str]:
    """Bind a configured test command to the entry's virtual environment."""
    prepared = list(command)
    if not prepared:
        raise ValueError("Test command must not be empty.")
    python = os.fspath(python_path)
    if prepared[0] in {"python", "python3"}:
        prepared[0] = python
    elif prepared[0] == "pytest":
        prepared = [python, "-m", "pytest", *prepared[1:]]
    return prepared


def discover_requirement_files(repo_path: str | Path) -> tuple[Path, ...]:
    """Return test/development requirement files in deterministic order.

    The validator's exact files come first, followed by additional root-level
    ``requirements-test*``/``requirements-dev*``/``test-requirements*`` matches
    and nested ``requirements/test*.txt`` files.
    """
    repository = Path(repo_path)
    discovered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError:
            return
        if relative in seen or not path.is_file():
            return
        seen.add(relative)
        discovered.append(path)

    for relative in _SCRIPT_REQUIREMENT_FILES:
        add(repository / relative)

    for pattern in (*_REQUIREMENT_PATTERNS, *_NESTED_REQUIREMENT_PATTERNS):
        for path in sorted(
            repository.glob(pattern),
            key=lambda candidate: (
                candidate.relative_to(repository).as_posix().casefold(),
                candidate.relative_to(repository).as_posix(),
            ),
        ):
            add(path)
    return tuple(discovered)


def has_packaging_files(repo_path: str | Path) -> bool:
    """Return whether the repository has a supported install metadata file."""
    repository = Path(repo_path)
    return any((repository / name).is_file() for name in _PACKAGING_FILES)


def build_install_plan(
    repo_path: str | Path,
    python_path: str | Path,
    *,
    uv_executable: str = "uv",
) -> InstallPlan:
    """Build the validator-compatible installation recipe for a repository.

    Pytest is installed first.  Editable extras are then attempted in the order
    test, tests, testing, dev, then a plain editable install.  Requirement
    files are additive after the first successful editable install.  A
    repository without packaging metadata skips editable installation.
    """
    repository = Path(repo_path)
    prefix = (uv_executable, "pip", "install", "--python", os.fspath(python_path))

    editable: list[InstallStep] = []
    editable_skipped = not has_packaging_files(repository)
    if not editable_skipped:
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
            name=f"requirements:{path.relative_to(repository).as_posix()}",
            phase="requirements",
            command=(
                *prefix,
                "-r",
                path.relative_to(repository).as_posix(),
            ),
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
        requirement_files=tuple(
            path.relative_to(repository).as_posix()
            for path in requirement_paths
        ),
        editable_skipped=editable_skipped,
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
    """Execute the validator-compatible install plan.

    Pytest is installed first.  A pytest timeout or failure aborts setup.
    Editable extras are tried until one effectively succeeds; a uv
    ``does not have an extra`` warning is not an install.  Requirement files
    are additive and non-fatal: failures are recorded and installation
    continues.  A deadline overrun stops setup.
    """
    if install_timeout_seconds <= 0:
        raise ValueError("install_timeout_seconds must be positive.")

    repository = Path(repo_path)
    started = clock()
    attempts: list[CommandAttempt] = []
    failed_requirements: list[str] = []

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
    if (
        pytest_attempt.deadline_exceeded
        or pytest_attempt.timed_out
        or not pytest_attempt.effective_success
    ):
        return _install_failure(
            attempts,
            plan,
            pytest_attempt,
            successful_install=None,
            failed_requirements=failed_requirements,
            duration_seconds=clock() - started,
        )

    successful_step: InstallStep | None = None
    if not plan.editable_skipped:
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
            if attempt.succeeded and _is_missing_extra(attempt):
                attempt = attempt.model_copy(update={"accepted": False})
            attempts.append(attempt)
            if attempt.deadline_exceeded or attempt.timed_out:
                return _install_failure(
                    attempts,
                    plan,
                    attempt,
                    successful_install=None,
                    failed_requirements=failed_requirements,
                    duration_seconds=clock() - started,
                )
            if attempt.effective_success:
                successful_step = step
                break

        if successful_step is None:
            return _install_failure(
                attempts,
                plan,
                attempts[-1] if attempts else None,
                successful_install=None,
                failed_requirements=failed_requirements,
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
        if attempt.deadline_exceeded:
            return _install_failure(
                attempts,
                plan,
                attempt,
                successful_install=(
                    successful_step.name if successful_step is not None else None
                ),
                failed_requirements=failed_requirements,
                duration_seconds=clock() - started,
            )
        if not attempt.effective_success:
            failed_requirements.append(step.name)

    successful_install = (
        successful_step.name if successful_step is not None else None
    )
    detail_attempt = (
        _attempt_for_name(attempts, successful_install)
        if successful_install is not None
        else pytest_attempt
    )
    for name in failed_requirements:
        detail_attempt = _attempt_for_name(attempts, name)
    return InstallResult(
        succeeded=True,
        successful_install=successful_install,
        editable_skipped=plan.editable_skipped,
        attempts=attempts,
        requirement_files=list(plan.requirement_files),
        failed_requirements=failed_requirements,
        log_tail=detail_attempt.log_tail,
        log_truncated=detail_attempt.log_truncated,
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
        cache_dir: Path,
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
        self.cache_dir = cache_dir
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
        cache_dir: str | Path | None = None,
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
        cache_root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else workspace / ".cache"
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
            cache_dir=cache_root,
            subprocess_env=build_subprocess_env(
                home_path,
                temp_dir,
                venv_path=venv_path,
                python_install_dir=managed_python_dir,
                cache_dir=cache_root,
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
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        prepared = prepare_test_command(
            command,
            venv_python_path(self.venv_path),
        )
        return _safe_run(
            self._command_runner,
            prepared,
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
        try:
            errors = _remove_paths(targets, protected={self.workspace_dir})
        except Exception as error:  # pragma: no cover - defensive fallback.
            errors = [
                "Cleanup failed before removing paths: "
                f"{type(error).__name__}: {error}"
            ]
        self.summary.cleanup_errors.extend(errors)

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
    cache_dir: str | Path | None = None,
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
        cache_dir=cache_dir,
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
    combined_output = f"{result.stdout}\n{result.stderr}".casefold()
    return CommandAttempt(
        name=name,
        phase=phase,
        command=command,
        return_code=result.return_code,
        succeeded=result.succeeded,
        timed_out=result.timed_out,
        missing_extra=(
            result.succeeded
            and "does not have an extra" in combined_output
        ),
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
    failed_requirements: list[str],
    duration_seconds: float,
) -> InstallResult:
    return InstallResult(
        succeeded=False,
        successful_install=successful_install,
        editable_skipped=plan.editable_skipped,
        attempts=attempts,
        requirement_files=list(plan.requirement_files),
        failed_requirements=list(failed_requirements),
        log_tail=failed_attempt.log_tail if failed_attempt is not None else "",
        log_truncated=(
            failed_attempt.log_truncated if failed_attempt is not None else False
        ),
        duration_seconds=duration_seconds,
    )


def _is_missing_extra(attempt: CommandAttempt) -> bool:
    """Return whether uv succeeded only because an extra was not defined."""
    return attempt.missing_extra


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
    protected_resolved: set[Path] = set()
    for path in protected:
        try:
            protected_resolved.add(path.resolve())
        except Exception:
            protected_resolved.add(path.absolute())
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            resolved = path.resolve()
        except Exception:
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
        except Exception as error:
            errors.append(
                f"Could not remove {path}: {type(error).__name__}: {error}"
            )
    return errors

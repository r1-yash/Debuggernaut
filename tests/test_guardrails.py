from pathlib import Path

from swe_agent.agent.guardrails import (
    proposal_rejection,
    protected_path_reason,
)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    return repository


def _write(repository: Path, relative_path: str, content: str = "x\n") -> Path:
    path = repository / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_blocks_structural_test_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for relative_path in [
        "tests/test_calc.py",
        "test/test_calc.py",
        "conftest.py",
        "pkg/test_utils.py",
        "pkg/utils_test.py",
        "tests/helpers.py",
    ]:
        _write(repository, relative_path)
        assert protected_path_reason(str(repository), relative_path) is not None


def test_allows_non_test_source_paths(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for relative_path in [
        "calc.py",
        "src/calculator.py",
        "src/testing_utils.py",
        "contest.py",
        "attest.py",
        "testing/helpers.py",
    ]:
        _write(repository, relative_path)
        assert protected_path_reason(str(repository), relative_path) is None


def test_testing_directory_allowed_when_command_targets_tests(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/test_calc.py")
    helpers = _write(repository, "testing/helpers.py")

    reason = protected_path_reason(
        str(repository), "testing/helpers.py", ["python", "-m", "pytest", "tests/"]
    )

    assert reason is None
    assert helpers.is_file()


def test_testing_directory_protected_when_command_names_it(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "testing/helpers.py")

    reason = protected_path_reason(
        str(repository), "testing/helpers.py",
        ["python", "-m", "pytest", "testing/"],
    )

    assert reason is not None
    assert "test command" in reason


def test_testing_file_protected_when_command_names_it(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "testing/check_api.py")
    _write(repository, "testing/helpers.py")
    command = ["python", "-m", "pytest", "testing/check_api.py"]

    assert protected_path_reason(
        str(repository), "testing/check_api.py", command
    ) is not None
    assert protected_path_reason(
        str(repository), "testing/helpers.py", command
    ) is None


def test_command_named_file_is_protected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "check_math.py")

    reason = protected_path_reason(
        str(repository), "check_math.py",
        ["python", "-m", "pytest", "check_math.py"],
    )

    assert reason is not None


def test_node_ids_and_flags_are_ignored(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "check_math.py")

    reason = protected_path_reason(
        str(repository), "check_math.py",
        ["pytest", "-q", "--rootdir=tests", "check_math.py::test_add"],
    )

    assert reason is not None


def test_symlink_into_tests_is_protected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/test_calc.py")
    (repository / "src").mkdir()
    (repository / "src" / "alias.py").symlink_to(
        repository / "tests" / "test_calc.py"
    )

    assert protected_path_reason(str(repository), "src/alias.py") is not None


def test_rejection_classifies_missing_path(tmp_path: Path) -> None:
    repository = _repository(tmp_path)

    rejection = proposal_rejection(str(repository), "missing.py")

    assert rejection is not None
    kind, reason = rejection
    assert kind == "nonexistent_path"
    assert "does not exist" in reason


def test_rejection_classifies_test_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "tests/test_calc.py")

    rejection = proposal_rejection(
        str(repository), "tests/test_calc.py", ["pytest"]
    )

    assert rejection is not None
    kind, reason = rejection
    assert kind == "test_file"
    assert "test file" in reason


def test_rejection_allows_source_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "calc.py")

    assert proposal_rejection(str(repository), "calc.py", ["pytest"]) is None


def test_explicit_test_patch_path_is_blocked(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "fixtures/helper.py")

    reason = protected_path_reason(
        str(repository),
        "fixtures/helper.py",
        ["pytest"],
        ["fixtures/helper.py"],
    )

    assert reason is not None
    assert "benchmark test patch" in reason


def test_explicit_test_patch_protection_is_case_insensitive(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    _write(repository, "fixtures/helper.py")

    assert protected_path_reason(
        str(repository),
        "fixtures/helper.py",
        ["pytest"],
        ["Fixtures/Helper.py"],
    ) is not None


def test_unlisted_path_remains_allowed(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "fixtures/helper.py")
    _write(repository, "fixtures/other.py")

    reason = protected_path_reason(
        str(repository),
        "fixtures/other.py",
        ["pytest"],
        ["fixtures/helper.py"],
    )

    assert reason is None


def test_rejection_classifies_test_patch_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository, "fixtures/helper.py")

    rejection = proposal_rejection(
        str(repository),
        "fixtures/helper.py",
        ["pytest"],
        ["fixtures/helper.py"],
    )

    assert rejection is not None
    kind, reason = rejection
    assert kind == "test_patch_file"
    assert "benchmark test patch" in reason

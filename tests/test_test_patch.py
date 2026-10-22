import subprocess
from pathlib import Path

import pytest

from swe_agent.benchmark.test_patch import apply_test_patch


def _git(repo_path: Path, *arguments: str) -> str:
    """Run a git command in a repository and return its stripped stdout."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository(path: Path) -> Path:
    """Create a tiny committed repository with one module."""
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(path, "add", "calc.py")
    _git(path, "commit", "-q", "-m", "base")
    return path


def _write_patch(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_modifies_existing_file_and_commits_baseline(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    original_head = _git(repository, "rev-parse", "HEAD")
    patch = _write_patch(
        tmp_path / "modify.diff",
        """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
""",
    )

    result = apply_test_patch(str(repository), str(patch))

    assert result.applied is True
    assert result.file_paths == ["calc.py"]
    assert result.baseline_commit == _git(repository, "rev-parse", "HEAD")
    assert result.baseline_commit != original_head
    assert (repository / "calc.py").read_text() == (
        "def add(a, b):\n    return a + b\n"
    )
    assert _git(repository, "status", "--porcelain") == ""
    assert [
        line
        for line in _git(repository, "show", "--name-only", "--format=", "HEAD").splitlines()
        if line
    ] == ["calc.py"]


def test_creates_new_test_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    patch = _write_patch(
        tmp_path / "new_file.diff",
        """diff --git a/tests/test_calc.py b/tests/test_calc.py
new file mode 100644
--- /dev/null
+++ b/tests/test_calc.py
@@ -0,0 +1,4 @@
+from calc import add
+
+def test_add():
+    assert add(1, 2) == 3
""",
    )

    result = apply_test_patch(str(repository), str(patch))

    assert result.file_paths == ["tests/test_calc.py"]
    assert (repository / "tests" / "test_calc.py").is_file()
    assert _git(repository, "status", "--porcelain") == ""


def test_failing_patch_leaves_repository_unchanged(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    original_head = _git(repository, "rev-parse", "HEAD")
    patch = _write_patch(
        tmp_path / "bad.diff",
        """diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a + b
+    return a * b
""",
    )

    with pytest.raises(ValueError, match="Test patch failed to apply"):
        apply_test_patch(str(repository), str(patch))

    assert (repository / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    assert _git(repository, "status", "--porcelain") == ""
    assert _git(repository, "rev-parse", "HEAD") == original_head


def test_rejects_empty_patch(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    original_head = _git(repository, "rev-parse", "HEAD")
    patch = _write_patch(tmp_path / "empty.diff", "")

    with pytest.raises(ValueError, match="did not change any files"):
        apply_test_patch(str(repository), str(patch))

    assert _git(repository, "status", "--porcelain") == ""
    assert _git(repository, "rev-parse", "HEAD") == original_head


def test_rejects_missing_patch_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")

    with pytest.raises(ValueError, match="does not exist"):
        apply_test_patch(str(repository), str(tmp_path / "missing.diff"))


def test_handles_rename(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    _git(repository, "mv", "calc.py", "maths.py")
    patch_text = _git(repository, "diff", "--cached")
    _git(repository, "reset", "--hard", "HEAD")
    patch = _write_patch(tmp_path / "rename.diff", patch_text + "\n")

    result = apply_test_patch(str(repository), str(patch))

    assert result.file_paths == ["maths.py"]
    assert not (repository / "calc.py").exists()
    assert (repository / "maths.py").read_text() == (
        "def add(a, b):\n    return a - b\n"
    )


def test_handles_deletion(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    _git(repository, "rm", "-q", "calc.py")
    patch_text = _git(repository, "diff", "--cached")
    _git(repository, "reset", "--hard", "HEAD")
    patch = _write_patch(tmp_path / "delete.diff", patch_text + "\n")

    result = apply_test_patch(str(repository), str(patch))

    assert result.file_paths == ["calc.py"]
    assert not (repository / "calc.py").exists()
    assert _git(repository, "status", "--porcelain") == ""

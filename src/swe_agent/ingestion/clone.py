"""Clone GitHub repositories into a local workspace for ingestion."""

from __future__ import annotations

import os
import shutil
import subprocess


def clone_repository(
    owner: str,
    repository: str,
    workspace_dir: str = "workspace",
) -> str:
    """Clone a GitHub repository and return its absolute local path."""
    if not owner.strip() or not repository.strip():
        raise ValueError("GitHub owner and repository must be provided.")

    target_path = os.path.join(workspace_dir, f"{owner}__{repository}")
    if os.path.exists(target_path):
        shutil.rmtree(target_path)

    os.makedirs(workspace_dir, exist_ok=True)
    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            f"https://github.com/{owner}/{repository}.git",
            target_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Git clone failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    return os.path.abspath(target_path)

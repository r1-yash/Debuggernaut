"""Utilities for accepting GitHub repository URLs."""

from __future__ import annotations

import re


_GITHUB_REPOSITORY_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[^/\s?#]+)/(?P<repository>[^/\s?#]+?)"
    r"(?:\.git)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)


def parse_repository_url(url: str) -> str:
    """Return ``owner/repository`` from a pasted GitHub repository URL.

    Accepted examples include ``https://github.com/owner/repository``, a URL
    with a trailing slash or ``.git`` suffix, and ``github.com/owner/repository``.
    """
    match = _GITHUB_REPOSITORY_URL.fullmatch(url.strip())
    if not match:
        raise ValueError("Enter a valid GitHub repository URL.")

    return f"{match['owner']}/{match['repository']}"

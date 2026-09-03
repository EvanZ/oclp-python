"""Helpers for resolving an implementation source at observation time."""

from __future__ import annotations

import subprocess
from pathlib import Path

from oclp.models import GitSource, OpaqueSource


def source_from_git_checkout(
    project_root: Path,
    *,
    path: str = ".",
) -> GitSource | OpaqueSource:
    """Resolve the checked-out Git revision that supplies an implementation.

    A source record needs a retrievable repository and immutable commit.  When
    either cannot be read from the checkout, return an explicit
    :class:`OpaqueSource` rather than inventing source provenance.
    """

    try:
        commit = _git_output(project_root, "rev-parse", "HEAD")
        repository = _git_output(project_root, "config", "--get", "remote.origin.url")
    except (OSError, subprocess.CalledProcessError):
        return OpaqueSource(
            reason="Git source metadata was unavailable at observation time."
        )
    if not repository:
        return OpaqueSource(
            reason="Git source metadata has no configured remote.origin.url."
        )
    return GitSource(repository=repository, commit=commit, path=path)


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=project_root,
        capture_output=True,
        text=True,
    ).stdout.strip()

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

    A source record needs a retrievable repository and immutable base commit.
    A checkout with uncommitted changes remains a valid source basis and is
    explicitly represented with ``dirty=True``.  When the repository or commit
    cannot be read, return an explicit :class:`OpaqueSource` rather than
    inventing source provenance.
    """

    try:
        commit = _git_output(project_root, "rev-parse", "HEAD")
        repository = _git_output(project_root, "config", "--get", "remote.origin.url")
        dirty = bool(_git_output(project_root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return OpaqueSource(
            reason="Git source metadata was unavailable at observation time."
        )
    if not repository:
        return OpaqueSource(
            reason="Git source metadata has no configured remote.origin.url."
        )
    return GitSource(
        repository=repository,
        commit=commit,
        path=path,
        dirty=dirty,
    )


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=project_root,
        capture_output=True,
        text=True,
    ).stdout.strip()

"""Helpers for resolving and durably capturing implementation source."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from oclp.models import (
    ArtifactSet,
    ArtifactSetMember,
    GitSource,
    OpaqueSource,
    new_record_id,
)

if TYPE_CHECKING:
    from oclp.publishing import LocalArtifactPublisher


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


def capture_git_source_overlay(
    project_root: Path,
    *,
    source: GitSource,
    publisher: LocalArtifactPublisher,
    name: str,
    relative_path: str,
    untracked_files: Iterable[str | Path] = (),
    captured_at: datetime | None = None,
) -> GitSource:
    """Capture one dirty Git worktree and bind it as ``source.overlay``.

    The resulting immutable ArtifactSet contains the complete binary diff
    against ``HEAD`` plus every selected untracked file.  Ignored files are
    never considered.  If Git reports untracked files, callers must select
    all of them explicitly; omitting one would make the overlay look more
    reproducible than it is.

    ``name`` remains application-owned because the ArtifactSet is displayed as
    an ordinary OCLP record. ``relative_path`` is a storage prefix local to the
    supplied publisher. Content-digest suffixes keep previously published
    payload locations immutable when a later capture uses the same prefix.
    """

    if not source.dirty:
        return source
    if not name:
        raise ValueError("Git source overlay names must be non-empty strings")
    _validate_relative_path(relative_path)

    project_root = project_root.resolve()
    patch = _git_output_bytes(
        project_root, "diff", "--binary", "--no-ext-diff", "HEAD", "--"
    )
    discovered_untracked = _untracked_paths(project_root)
    selected_untracked = _selected_untracked_paths(
        project_root,
        discovered=discovered_untracked,
        requested=untracked_files,
    )

    if not patch and not selected_untracked:
        raise ValueError(
            "Git reported a dirty checkout, but no tracked diff or selected "
            "untracked files were available to capture"
        )

    created_at = captured_at or datetime.now(UTC)
    members: list[ArtifactSetMember] = []
    if patch:
        patch_digest = sha256(patch).hexdigest()
        published = publisher.artifact_for_bytes(
            name=f"{name} patch",
            relative_path=f"{relative_path}/{patch_digest}.patch",
            content=patch,
            media_type="text/x-diff",
            created_at=created_at,
            annotations={"source_basis": "git-diff-head"},
        )
        members.append(
            ArtifactSetMember(
                name="tracked.patch",
                artifact=published.reference,
                role="tracked-source-changes",
            )
        )

    for path in selected_untracked:
        content = (project_root / path).read_bytes()
        content_digest = sha256(content).hexdigest()
        published = publisher.artifact_for_bytes(
            name=f"{name}: {path}",
            relative_path=(
                f"{relative_path}/untracked/{content_digest}-{path.name}"
            ),
            content=content,
            media_type="application/octet-stream",
            created_at=created_at,
            annotations={"source_path": path.as_posix()},
        )
        members.append(
            ArtifactSetMember(
                name=f"untracked/{path.as_posix()}",
                artifact=published.reference,
                role="untracked-source-file",
            )
        )

    overlay = ArtifactSet(
        id=new_record_id(),
        name=name,
        members=tuple(members),
        created_at=created_at,
    )
    return source.model_copy(update={"overlay": publisher.publish(overlay)})


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=project_root,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_output_bytes(project_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        cwd=project_root,
        capture_output=True,
    ).stdout


def _untracked_paths(project_root: Path) -> tuple[Path, ...]:
    output = _git_output_bytes(
        project_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    return tuple(
        Path(value.decode("utf-8"))
        for value in output.split(b"\0")
        if value
    )


def _selected_untracked_paths(
    project_root: Path,
    *,
    discovered: tuple[Path, ...],
    requested: Iterable[str | Path],
) -> tuple[Path, ...]:
    discovered_set = set(discovered)
    selected: set[Path] = set()
    for requested_path in requested:
        candidate = Path(requested_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (project_root / candidate).resolve()
        )
        try:
            relative = resolved.relative_to(project_root)
        except ValueError as error:
            raise ValueError(
                f"untracked source file {requested_path!r} is outside project_root"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"untracked source file {requested_path!r} is not a file")
        selected.add(relative)

    unexpected = selected - discovered_set
    if unexpected:
        paths = ", ".join(sorted(path.as_posix() for path in unexpected))
        raise ValueError(f"requested files are not untracked Git files: {paths}")
    missing = discovered_set - selected
    if missing:
        paths = ", ".join(sorted(path.as_posix() for path in missing))
        raise ValueError(
            "dirty Git source capture requires explicit selection of every "
            f"untracked file: {paths}"
        )
    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def _validate_relative_path(relative_path: str) -> None:
    path = Path(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "Git source overlay relative_path must be a non-empty relative path"
        )

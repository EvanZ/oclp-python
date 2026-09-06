"""Tests for observing Git implementation-source metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from oclp.models import Artifact, ArtifactSet, GitSource
from oclp.publishing import LocalArtifactPublisher
from oclp.sources import capture_git_source_overlay, source_from_git_checkout


@pytest.mark.parametrize(
    ("status", "expected_dirty"),
    (("", False), (" M src/example.py", True)),
)
def test_source_from_git_checkout_labels_dirty_worktrees(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected_dirty: bool,
) -> None:
    """A dirty checkout remains a GitSource instead of becoming opaque."""

    def git_output(_: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): "a" * 40,
            ("config", "--get", "remote.origin.url"): (
                "https://github.com/example/reports.git"
            ),
            ("status", "--porcelain"): status,
        }
        return values[arguments]

    monkeypatch.setattr("oclp.sources._git_output", git_output)

    source = source_from_git_checkout(Path("/example"), path="src/reports.py")

    assert source == GitSource(
        repository="https://github.com/example/reports.git",
        commit="a" * 40,
        path="src/reports.py",
        dirty=expected_dirty,
    )
    assert ("dirty" in source.model_dump(mode="json")) is expected_dirty


def test_capture_git_source_overlay_persists_tracked_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dirty source may bind its exact tracked diff as an ArtifactSet."""

    def git_output(_: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): "a" * 40,
            ("config", "--get", "remote.origin.url"): (
                "https://github.com/example/reports.git"
            ),
            ("status", "--porcelain"): " M src/example.py",
        }
        return values[arguments]

    def git_output_bytes(_: Path, *arguments: str) -> bytes:
        if arguments[0] == "diff":
            return b"diff --git a/src/example.py b/src/example.py\n"
        if arguments[0] == "ls-files":
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr("oclp.sources._git_output", git_output)
    monkeypatch.setattr("oclp.sources._git_output_bytes", git_output_bytes)
    source = source_from_git_checkout(tmp_path)
    assert isinstance(source, GitSource)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        captured = capture_git_source_overlay(
            tmp_path,
            source=source,
            publisher=publisher,
            name="Working source overlay",
            relative_path="source-overlays/example",
        )
        records = publisher.records()

    assert captured.overlay is not None
    artifact = next(record for record in records if isinstance(record, Artifact))
    artifact_set = next(record for record in records if isinstance(record, ArtifactSet))
    assert artifact.media_type == "text/x-diff"
    assert artifact_set.id == captured.overlay.id
    assert artifact_set.members[0].name == "tracked.patch"


def test_capture_git_source_overlay_requires_explicit_untracked_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Untracked files are never silently included in a captured overlay."""

    (tmp_path / "local.py").write_text("answer = 42\n")

    def git_output(_: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): "a" * 40,
            ("config", "--get", "remote.origin.url"): (
                "https://github.com/example/reports.git"
            ),
            ("status", "--porcelain"): "?? local.py",
        }
        return values[arguments]

    def git_output_bytes(_: Path, *arguments: str) -> bytes:
        if arguments[0] == "diff":
            return b""
        if arguments[0] == "ls-files":
            return b"local.py\0"
        raise AssertionError(arguments)

    monkeypatch.setattr("oclp.sources._git_output", git_output)
    monkeypatch.setattr("oclp.sources._git_output_bytes", git_output_bytes)
    source = source_from_git_checkout(tmp_path)
    assert isinstance(source, GitSource)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with pytest.raises(ValueError, match="explicit selection"):
            capture_git_source_overlay(
                tmp_path,
                source=source,
                publisher=publisher,
                name="Working source overlay",
                relative_path="source-overlays/example",
            )
        captured = capture_git_source_overlay(
            tmp_path,
            source=source,
            publisher=publisher,
            name="Working source overlay",
            relative_path="source-overlays/example",
            untracked_files=("local.py",),
        )

    assert captured.overlay is not None

"""Tests for observing Git implementation-source metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from oclp.models import GitSource
from oclp.sources import source_from_git_checkout


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

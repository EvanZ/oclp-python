"""Conformance tests for the portable dataset-snapshot profile."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp import canonical_json_bytes, record_digest
from oclp.profiles import DatasetSnapshotBinding, DatasetSnapshotManifest


def _profile_root() -> Path:
    configured = os.environ.get("OCLP_PROFILES_ROOT")
    if configured is None:
        pytest.skip("set OCLP_PROFILES_ROOT to run the profile conformance suite")
    return Path(configured).resolve() / "tests" / "profiles" / "dataset-snapshot"


def test_dataset_snapshot_binding_requires_the_declared_version() -> None:
    assert (
        DatasetSnapshotBinding.model_validate({"version": "0.2.0-draft"}).version
        == "0.2.0-draft"
    )
    with pytest.raises(ValidationError):
        DatasetSnapshotBinding.model_validate({"version": "0.1.0-draft"})


def _manifest() -> dict[str, object]:
    return json.loads((_profile_root() / "manifest.json").read_text())


def test_dataset_snapshot_valid_vectors_have_canonical_digests() -> None:
    for entry in _manifest()["valid"]:
        assert isinstance(entry, dict)
        snapshot = DatasetSnapshotManifest.model_validate(
            json.loads((_profile_root() / entry["path"]).read_text())
        )
        assert canonical_json_bytes(snapshot).decode() == entry["canonical_json"]
        assert str(record_digest(snapshot)) == entry["digest"]


def test_dataset_snapshot_invalid_vectors_are_rejected() -> None:
    for fixture_path in _manifest()["invalid"]:
        with pytest.raises(ValidationError):
            DatasetSnapshotManifest.model_validate(
                json.loads((_profile_root() / fixture_path).read_text())
            )

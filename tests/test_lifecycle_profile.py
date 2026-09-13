"""Conformance coverage for the portable lifecycle profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from oclp import canonical_json_bytes, record_digest
from oclp.profiles import (
    LifecycleBinding,
    lifecycle_from_id,
    lifecycle_from_profiles,
    new_lifecycle,
)


def _profile_root() -> Path:
    configured = os.environ.get("OCLP_PROFILES_ROOT")
    if configured is None:
        pytest.skip("set OCLP_PROFILES_ROOT to run the profile conformance suite")
    return Path(configured).resolve() / "tests" / "profiles" / "lifecycle"


def _manifest() -> dict[str, object]:
    return json.loads((_profile_root() / "manifest.json").read_text())


def test_valid_lifecycle_profile_vector_is_accepted() -> None:
    for entry in _manifest()["valid"]:
        assert isinstance(entry, dict)
        binding = LifecycleBinding.model_validate(
            json.loads((_profile_root() / entry["path"]).read_text())
        )

        assert canonical_json_bytes(binding).decode() == entry["canonical_json"]
        assert str(record_digest(binding)) == entry["digest"]


def test_invalid_lifecycle_profile_vectors_are_rejected() -> None:
    for name in _manifest()["invalid"]:
        assert isinstance(name, str)
        value = json.loads((_profile_root() / name).read_text())

        with pytest.raises(ValidationError):
            LifecycleBinding.model_validate(value)


def test_lifecycle_can_be_minted_reused_and_rehydrated_from_records() -> None:
    lifecycle = new_lifecycle()
    second = new_lifecycle()

    assert lifecycle.lifecycle_id != second.lifecycle_id
    assert lifecycle_from_id(str(lifecycle.lifecycle_id)) == lifecycle
    assert lifecycle_from_profiles(lifecycle.profile_bindings()) == lifecycle
    assert UUID(str(lifecycle.lifecycle_id)).version == 4


def test_lifecycle_requires_an_explicit_valid_profile_binding() -> None:
    with pytest.raises(ValueError, match="missing the lifecycle profile"):
        lifecycle_from_profiles(None)
    with pytest.raises(ValidationError):
        lifecycle_from_id("not-a-uuid")

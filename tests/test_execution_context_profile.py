"""Conformance coverage for the execution-context profile."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp import canonical_json_bytes, record_digest
from oclp.profiles import ExecutionContextBinding, ExecutionContextManifest


def _profile_root() -> Path:
    configured = os.environ.get("OCLP_PROFILES_ROOT")
    if configured is None:
        pytest.skip("set OCLP_PROFILES_ROOT to run the profile conformance suite")
    return Path(configured).resolve() / "tests" / "profiles" / "execution-context"


def _manifest() -> dict[str, object]:
    return json.loads((_profile_root() / "manifest.json").read_text())


def test_valid_execution_context_profile_vector_is_accepted() -> None:
    for entry in _manifest()["valid"]:
        assert isinstance(entry, dict)
        profile = ExecutionContextManifest.model_validate(
            json.loads((_profile_root() / entry["path"]).read_text())
        )

        assert canonical_json_bytes(profile).decode() == entry["canonical_json"]
        assert str(record_digest(profile)) == entry["digest"]


def test_invalid_execution_context_profile_vectors_are_rejected() -> None:
    for name in _manifest()["invalid"]:
        assert isinstance(name, str)
        value = json.loads((_profile_root() / name).read_text())

        with pytest.raises(ValidationError):
            ExecutionContextManifest.model_validate(value)


def test_execution_context_binding_requires_an_exact_manifest_reference() -> None:
    binding = ExecutionContextBinding.model_validate(
        {
            "version": "0.1.0-draft",
            "manifest": {
                "id": "urn:example:artifact:execution-context",
                "digest": {"value": "a" * 64},
            },
        }
    )

    assert binding.manifest.digest is not None
    with pytest.raises(ValidationError):
        ExecutionContextBinding.model_validate(
            {
                "version": "0.1.0-draft",
                "manifest": {"id": "urn:example:artifact:execution-context"},
            }
        )
    with pytest.raises(ValidationError):
        ExecutionContextBinding.model_validate(
            {
                "manifest": {
                    "id": "urn:example:artifact:execution-context",
                    "digest": {"value": "a" * 64},
                }
            }
        )

"""Conformance coverage for the execution-context profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp.profiles import ExecutionContextBinding, ExecutionContextManifest

PROFILE_ROOT = Path(__file__).parent / "profiles" / "execution-context"


def test_valid_execution_context_profile_vector_is_accepted() -> None:
    value = json.loads((PROFILE_ROOT / "valid" / "manifest.json").read_text())

    profile = ExecutionContextManifest.model_validate(value)

    assert profile.runtime.interpreter == "CPython 3.12.11"
    assert profile.configuration is not None


@pytest.mark.parametrize(
    "name",
    ["configuration-unbound.json", "dependency-lock-unbound.json"],
)
def test_invalid_execution_context_profile_vectors_are_rejected(name: str) -> None:
    value = json.loads((PROFILE_ROOT / "invalid" / name).read_text())

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

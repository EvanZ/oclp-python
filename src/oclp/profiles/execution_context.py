"""Portable runtime and configuration provenance for one OCLP Execution."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from oclp.models import OclpModel, RecordReference

EXECUTION_CONTEXT_PROFILE = "execution-context"
EXECUTION_CONTEXT_PROFILE_VERSION = "0.3.0-draft"


class ExecutionRuntime(OclpModel):
    """Portable resolver and platform facts for one execution context."""

    interpreter: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    dependency_lock: RecordReference

class ExecutionContextManifest(OclpModel):
    """Canonical profile payload describing an Execution's execution context."""

    oclp_profile: Literal["execution-context"] = EXECUTION_CONTEXT_PROFILE
    oclp_profile_version: Literal["0.3.0-draft"] = EXECUTION_CONTEXT_PROFILE_VERSION
    runtime: ExecutionRuntime
    configuration: RecordReference | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

class ExecutionContextArtifactBinding(OclpModel):
    """The value carried under an Artifact's ``profiles.execution-context`` key."""

    version: Literal["0.3.0-draft"]


class ExecutionContextBinding(OclpModel):
    """The value carried under an Execution's ``profiles.execution-context`` key."""

    version: Literal["0.3.0-draft"]
    manifest: RecordReference

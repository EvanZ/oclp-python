"""Portable runtime and configuration provenance for one OCLP Invocation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from oclp.models import OclpModel, RecordReference

EXECUTION_CONTEXT_PROFILE = "execution-context"
EXECUTION_CONTEXT_PROFILE_VERSION = "0.1.0-draft"


class ExecutionRuntime(OclpModel):
    """Portable resolver and platform facts for one execution context."""

    interpreter: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    dependency_lock: RecordReference

    @model_validator(mode="after")
    def dependency_lock_is_content_bound(self) -> ExecutionRuntime:
        if self.dependency_lock.digest is None:
            raise ValueError("execution dependency locks must include a record digest")
        return self


class ExecutionContextManifest(OclpModel):
    """Canonical profile payload describing an Invocation's execution context."""

    oclp_profile: Literal["execution-context"] = EXECUTION_CONTEXT_PROFILE
    oclp_profile_version: Literal["0.1.0-draft"] = EXECUTION_CONTEXT_PROFILE_VERSION
    runtime: ExecutionRuntime
    configuration: RecordReference | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def configuration_is_content_bound(self) -> ExecutionContextManifest:
        if self.configuration is not None and self.configuration.digest is None:
            raise ValueError("execution configurations must include a record digest")
        return self


class ExecutionContextArtifactBinding(OclpModel):
    """The value carried under an Artifact's ``profiles.execution-context`` key."""

    version: Literal["0.1.0-draft"]


class ExecutionContextBinding(OclpModel):
    """The value carried under an Invocation's ``profiles.execution-context`` key."""

    version: Literal["0.1.0-draft"]
    manifest: RecordReference

    @model_validator(mode="after")
    def profile_is_content_bound(self) -> ExecutionContextBinding:
        if self.manifest.digest is None:
            raise ValueError("execution context profiles must include a record digest")
        return self

"""Typed records in the experimental OCLP core vocabulary."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

OCLP_DRAFT_VERSION = "0.1.0-draft"


class OclpModel(BaseModel):
    """Strict immutable base model for protocol records and value objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ProfileBindings = Annotated[
    dict[str, dict[str, JsonValue]],
    Field(min_length=1),
]


class Digest(OclpModel):
    algorithm: Literal["sha256"] = "sha256"
    value: str = Field(pattern=r"^[0-9a-f]{64}$")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.value}"


class RecordReference(OclpModel):
    id: str = Field(min_length=1)
    digest: Digest | None = None


class PortDefinition(OclpModel):
    name: str = Field(min_length=1)
    cardinality: Literal["one", "many"] = "one"
    required: bool = True
    media_types: tuple[str, ...] = ()


class GitSource(OclpModel):
    """An immutable Git source revision selected by an implementation."""

    kind: Literal["git"] = "git"
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    path: str = Field(default=".", min_length=1)
    overlay: RecordReference | None = None

    @model_validator(mode="after")
    def overlay_is_content_bound(self) -> GitSource:
        if self.overlay is not None and self.overlay.digest is None:
            raise ValueError("git source overlays must include a record digest")
        return self


class ArtifactSource(OclpModel):
    """An immutable Artifact selected as an implementation's source basis."""

    kind: Literal["artifact"] = "artifact"
    artifact: RecordReference

    @model_validator(mode="after")
    def artifact_reference_is_content_bound(self) -> ArtifactSource:
        if self.artifact.digest is None:
            raise ValueError("artifact sources must include a record digest")
        return self


class ServiceSource(OclpModel):
    """A versioned external service selected as an implementation's source basis."""

    kind: Literal["service"] = "service"
    locator: str = Field(min_length=1)
    version: str = Field(min_length=1)


class OpaqueSource(OclpModel):
    """An explicitly unavailable implementation source basis."""

    kind: Literal["opaque"] = "opaque"
    reason: str = Field(min_length=1)


ImplementationSource = Annotated[
    GitSource | ArtifactSource | ServiceSource | OpaqueSource,
    Field(discriminator="kind"),
]


class Implementation(OclpModel):
    kind: Literal["python-callable", "container", "command", "other"]
    locator: str = Field(min_length=1)
    digest: Digest | None = None
    artifact: RecordReference | None = None
    source: ImplementationSource

    @model_validator(mode="after")
    def artifact_reference_is_content_bound(self) -> Implementation:
        if self.artifact is not None and self.artifact.digest is None:
            raise ValueError("implementation artifacts must include a record digest")
        return self


class ContractReference(OclpModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Diagnostic(OclpModel):
    """A compact, portable explanation attached to an Event or Evidence record."""

    code: str | None = Field(default=None, min_length=1)
    message: str | None = Field(default=None, min_length=1)
    stage: str | None = Field(default=None, min_length=1)
    artifact: RecordReference | None = None

    @model_validator(mode="after")
    def has_content_and_binds_detail_artifact(self) -> Diagnostic:
        if not any((self.code, self.message, self.stage, self.artifact)):
            raise ValueError(
                "diagnostics must include code, message, stage, or artifact"
            )
        if self.artifact is not None and self.artifact.digest is None:
            raise ValueError("diagnostic artifacts must include a record digest")
        return self


class CoreRecord(OclpModel):
    oclp_version: Literal["0.1.0-draft"] = OCLP_DRAFT_VERSION
    id: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    profiles: ProfileBindings | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def profile_names_are_nonempty(self) -> CoreRecord:
        if self.profiles is not None and any(
            not profile_id for profile_id in self.profiles
        ):
            raise ValueError("profile identifiers must be non-empty")
        return self


class Artifact(CoreRecord):
    kind: Literal["artifact"] = "artifact"
    media_type: str = Field(min_length=1)
    digest: Digest
    size: int = Field(ge=0)
    created_at: datetime | None = None
    locations: tuple[str, ...] = ()
    schema_uri: str | None = None

    @field_validator("created_at")
    @classmethod
    def created_at_has_explicit_offset(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("artifact created_at must include a UTC offset")
        return value

    @model_validator(mode="after")
    def identifier_is_independent_of_content_digest(self) -> Artifact:
        if self.id == f"urn:{self.digest.algorithm}:{self.digest.value}":
            raise ValueError(
                "artifact ID must be a logical identifier, not its content digest"
            )
        return self


class ArtifactSetMember(OclpModel):
    """One immutable Artifact assigned a stable name and semantic role."""

    name: str = Field(min_length=1)
    artifact: RecordReference
    role: str | None = Field(default=None, min_length=1)
    required: bool = True

    @model_validator(mode="after")
    def artifact_reference_is_content_bound(self) -> ArtifactSetMember:
        if self.artifact.digest is None:
            raise ValueError("artifact set members must include an artifact digest")
        return self


class ArtifactSet(CoreRecord):
    """An immutable, named collection of exact Artifact references."""

    kind: Literal["artifact_set"] = "artifact_set"
    members: tuple[ArtifactSetMember, ...] = Field(min_length=1)
    created_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def created_at_has_explicit_offset(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("artifact set created_at must include a UTC offset")
        return value

    @model_validator(mode="after")
    def member_names_are_unique(self) -> ArtifactSet:
        names = [member.name for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("artifact set member names must be unique")
        return self


class ComputationDefinition(CoreRecord):
    kind: Literal["definition"] = "definition"
    implementation: Implementation
    input_ports: tuple[PortDefinition, ...] = ()
    output_ports: tuple[PortDefinition, ...] = ()

    @model_validator(mode="after")
    def port_names_are_unique(self) -> ComputationDefinition:
        for ports in (self.input_ports, self.output_ports):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError("port names must be unique within each direction")
        return self


class Invocation(CoreRecord):
    kind: Literal["invocation"] = "invocation"
    definition: RecordReference
    parent_invocation: RecordReference | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    inputs: dict[str, tuple[RecordReference, ...]] = Field(default_factory=dict)
    outputs: dict[str, tuple[RecordReference, ...]] | None = None
    requested_outputs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def output_references_are_content_bound(self) -> Invocation:
        if self.parent_invocation is not None and self.parent_invocation.id == self.id:
            raise ValueError("an invocation cannot be its own parent")
        for references in (self.outputs or {}).values():
            if any(reference.digest is None for reference in references):
                raise ValueError("invocation outputs must include record digests")
        return self


class Evidence(CoreRecord):
    kind: Literal["evidence"] = "evidence"
    subject: RecordReference
    contract: ContractReference
    outcome: Literal["pass", "fail", "error"]
    observed_at: datetime
    diagnostic: Diagnostic | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class GitCheckout(OclpModel):
    """The local Git worktree observed for one execution attempt."""

    kind: Literal["git"] = "git"
    worktree: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    dirty: bool = False


class ExecutionContext(OclpModel):
    """Execution-local context attached to a lifecycle observation."""

    git_checkout: GitCheckout | None = None


class LifecycleEvent(CoreRecord):
    kind: Literal["event"] = "event"
    invocation: RecordReference
    event_type: str = Field(min_length=1)
    occurred_at: datetime
    sequence: int = Field(ge=0)
    attempt_id: str | None = None
    execution: ExecutionContext | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None
    diagnostic: Diagnostic | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


OclpRecord = Annotated[
    Artifact
    | ArtifactSet
    | ComputationDefinition
    | Invocation
    | Evidence
    | LifecycleEvent,
    Field(discriminator="kind"),
]

OCLP_RECORD_ADAPTER = TypeAdapter(OclpRecord)

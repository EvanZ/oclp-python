"""Typed records in the experimental OCLP core vocabulary."""

from __future__ import annotations

import warnings
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

OCLP_DRAFT_VERSION = "0.2.0-draft"


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


with warnings.catch_warnings():
    # ``schema`` is the normative protocol field. Pydantic retains a deprecated
    # BaseModel.schema() compatibility method, so suppress only that expected
    # field-shadow warning while preserving the canonical field name.
    warnings.filterwarnings(
        "ignore",
        message='Field name "schema" in "ParameterDefinition" shadows an attribute',
        category=UserWarning,
    )

    class ParameterDefinition(OclpModel):
        """One JSON-valued argument in a reusable Computation interface.

        Artifacts belong on input ports; this value object describes the remaining
        concrete arguments needed to reproduce an Execution.  ``schema`` uses a
        JSON Schema subschema so its vocabulary remains language-neutral.
        """

        name: str = Field(min_length=1)
        schema: dict[str, JsonValue]
        required: bool = True


class GitSource(OclpModel):
    """A Git source basis selected by an implementation.

    ``commit`` is always the immutable base revision.  ``dirty`` makes an
    observed local checkout with uncommitted changes explicit; a producer can
    additionally bind an exact ``overlay`` when those changes are captured.
    """

    kind: Literal["git"] = "git"
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    path: str = Field(default=".", min_length=1)
    dirty: bool = Field(default=False, exclude_if=lambda value: not value)
    overlay: RecordReference | None = None

    @model_validator(mode="after")
    def overlay_is_content_bound(self) -> GitSource:
        if self.overlay is not None and self.overlay.digest is None:
            raise ValueError("git source overlays must include a record digest")
        if self.overlay is not None and not self.dirty:
            raise ValueError("git source overlays require dirty=true")
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


EvidenceOutcome = Literal["pass", "fail", "error"]


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
    oclp_version: Literal["0.2.0-draft"] = OCLP_DRAFT_VERSION
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


class Computation(CoreRecord):
    """A reusable, source-bound computation interface and implementation."""

    kind: Literal["computation"] = "computation"
    implementation: Implementation
    input_ports: tuple[PortDefinition, ...] = ()
    output_ports: tuple[PortDefinition, ...] = ()
    parameter_definitions: tuple[ParameterDefinition, ...] = ()
    required_evidence: tuple[Implementation, ...] | None = Field(
        default=None,
        min_length=1,
    )

    @model_validator(mode="after")
    def interface_and_evidence_requirements_are_unique(self) -> Computation:
        for ports in (self.input_ports, self.output_ports):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError("port names must be unique within each direction")
        parameter_names = [parameter.name for parameter in self.parameter_definitions]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("computation parameter names must be unique")
        input_names = {port.name for port in self.input_ports}
        if input_names.intersection(parameter_names):
            raise ValueError(
                "computation parameter names must not overlap input port names"
            )
        evaluators = self.required_evidence or ()
        evaluator_keys = [
            evaluator.model_dump_json(exclude_none=True) for evaluator in evaluators
        ]
        if len(evaluator_keys) != len(set(evaluator_keys)):
            raise ValueError("required evidence evaluators must be unique")
        return self


class Execution(CoreRecord):
    """One concrete run of a Computation with immutable I/O bindings."""

    kind: Literal["execution"] = "execution"
    computation: RecordReference
    parent_execution: RecordReference | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    inputs: dict[str, tuple[RecordReference, ...]] = Field(default_factory=dict)
    outputs: dict[str, tuple[RecordReference, ...]] | None = None
    requested_outputs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def output_references_are_content_bound(self) -> Execution:
        if self.parent_execution is not None and self.parent_execution.id == self.id:
            raise ValueError("an execution cannot be its own parent")
        for references in (self.outputs or {}).values():
            if any(reference.digest is None for reference in references):
                raise ValueError("execution outputs must include record digests")
        return self


class Evidence(CoreRecord):
    kind: Literal["evidence"] = "evidence"
    subject: RecordReference
    evaluator: Implementation
    outcome: EvidenceOutcome
    observed_at: datetime
    diagnostic: Diagnostic | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class GitCheckout(OclpModel):
    """The local Git worktree observed for one Execution."""

    kind: Literal["git"] = "git"
    worktree: str = Field(min_length=1)
    commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    dirty: bool = False


class ExecutionContext(OclpModel):
    """Execution-local runtime context attached to a lifecycle observation."""

    git_checkout: GitCheckout | None = None


class Event(CoreRecord):
    kind: Literal["event"] = "event"
    execution: RecordReference
    event_type: str = Field(min_length=1)
    occurred_at: datetime
    sequence: int = Field(ge=0)
    runtime: ExecutionContext | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None
    diagnostic: Diagnostic | None = None
    data: dict[str, JsonValue] = Field(default_factory=dict)


OclpRecord = Annotated[
    Artifact | ArtifactSet | Computation | Execution | Evidence | Event,
    Field(discriminator="kind"),
]

OCLP_RECORD_ADAPTER = TypeAdapter(OclpRecord)

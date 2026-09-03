"""Reference Python execution runtime for OCLP observation.

The protocol defines immutable lifecycle records; this module is the SDK
implementation that creates and binds them for decorated Python calls. An
active :class:`OclpRun` materializes declared Computation outputs and acquired
Artifact values as immutable payloads and records, while ordinary calls retain
normal Python behavior outside the observation context.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints
from urllib.parse import unquote, urlparse

from oclp.artifacts import (
    DEFAULT_ARTIFACT_ADAPTERS,
    ArtifactAdapterRegistry,
    ArtifactHandle,
    ArtifactType,
)
from oclp.canonical import canonical_json_bytes, record_digest
from oclp.computations import (
    ArtifactSetInput,
    ComputationTemplate,
    ManyArtifacts,
    computation_input_artifact_types,
    computation_record,
)
from oclp.evidence import evaluate_evidence, evidence_template
from oclp.models import (
    Artifact,
    ArtifactSet,
    ArtifactSetMember,
    ArtifactSource,
    Computation,
    Diagnostic,
    Event,
    Evidence,
    Execution,
    GitSource,
    Implementation,
    ImplementationSource,
    JsonValue,
    OclpRecord,
    ProfileBindings,
    RecordReference,
)
from oclp.profiles.lifecycle import LIFECYCLE_PROFILE, LIFECYCLE_PROFILE_VERSION
from oclp.publishing import LocalArtifactPublisher, PublishedArtifact, utc_now

_ACTIVE_RUN: ContextVar[OclpRun | None] = ContextVar("oclp_active_run", default=None)
CallableT = TypeVar("CallableT", bound=Callable[..., object])
_LIFECYCLE_TEMPLATE_ATTRIBUTE = "__oclp_lifecycle_template__"


@dataclass(frozen=True)
class LifecycleTemplate:
    """Static SDK declaration for one application-owned workflow boundary.

    A lifecycle is *not* an OCLP Core record and this declaration does not
    manufacture a parent Execution. It supplies the project namespace and
    human-readable run name used when :func:`observe_lifecycle` activates one
    real :class:`OclpRun` for a decorated workflow.
    """

    namespace: str
    name: str

    def profile_for(self, run_id: str) -> ProfileBindings:
        """Return the portable profile binding for one concrete lifecycle run."""

        return {
            LIFECYCLE_PROFILE: {
                "version": LIFECYCLE_PROFILE_VERSION,
                "run_id": f"{self.namespace}:lifecycle:{run_id}",
                # ``run_id`` is the durable machine identity. ``run_name``
                # is the concise human label used by explorers.
                "run_name": self.name,
            }
        }


def lifecycle(
    *,
    namespace: str,
    name: str,
) -> Callable[[CallableT], CallableT]:
    """Declare an application workflow as one observed lifecycle boundary.

    The decorated callable keeps its normal Python signature and behavior.
    Pair it with :func:`observe_lifecycle` at the application bootstrap point
    to configure a publisher, source basis, and concrete ``run_id``. The SDK
    then derives the shared ``profiles.lifecycle`` binding for every real
    Execution produced by decorated calls inside the workflow.
    """

    if not isinstance(namespace, str) or not namespace.startswith("urn:"):
        raise ValueError("OCLP lifecycle namespaces must be URNs")
    if not isinstance(name, str) or not name:
        raise ValueError("OCLP lifecycle names must be non-empty strings")
    template = LifecycleTemplate(namespace=namespace, name=name)

    def decorate(function: CallableT) -> CallableT:
        if not callable(function):
            raise TypeError("@lifecycle can decorate only callable workflows")
        setattr(function, _LIFECYCLE_TEMPLATE_ATTRIBUTE, template)
        return function

    return decorate


def lifecycle_template(workflow: Callable[..., object]) -> LifecycleTemplate:
    """Return the static lifecycle declaration attached to ``workflow``."""

    template = getattr(workflow, _LIFECYCLE_TEMPLATE_ATTRIBUTE, None)
    if not isinstance(template, LifecycleTemplate):
        name = getattr(workflow, "__qualname__", repr(workflow))
        raise ValueError(f"workflow {name!r} has no OCLP lifecycle declaration")
    return template


@contextmanager
def observe_lifecycle(
    workflow: Callable[..., object],
    *,
    publisher: LocalArtifactPublisher,
    run_id: str,
    source: ImplementationSource,
    parent_execution: RecordReference | None = None,
    profiles: ProfileBindings | None = None,
    artifact_adapters: ArtifactAdapterRegistry = DEFAULT_ARTIFACT_ADAPTERS,
) -> Generator[OclpRun, None, None]:
    """Activate the SDK runtime for one ``@lifecycle``-declared workflow.

    Storage and source selection remain application bootstrap concerns. The
    SDK owns the resulting runtime, lifecycle profile binding, artifact
    materialization, Execution/Event publication, and failure capture. Extra
    profiles may be supplied, but may not replace the lifecycle profile
    derived from the workflow declaration.
    """

    template = lifecycle_template(workflow)
    merged_profiles: ProfileBindings = dict(profiles or {})
    existing = merged_profiles.get(LIFECYCLE_PROFILE)
    generated = template.profile_for(run_id)[LIFECYCLE_PROFILE]
    if existing is not None and existing != generated:
        raise ValueError(
            "observe_lifecycle derives profiles.lifecycle from the decorated "
            "workflow and concrete run_id; do not override it"
        )
    merged_profiles[LIFECYCLE_PROFILE] = generated
    with OclpRun(
        publisher=publisher,
        namespace=template.namespace,
        run_id=run_id,
        source=source,
        parent_execution=parent_execution,
        profiles=merged_profiles,
        artifact_adapters=artifact_adapters,
    ) as observed:
        yield observed


@dataclass(frozen=True)
class _ValueBinding:
    """An in-process value and the immutable Artifact produced from it."""

    value: object
    port: str
    artifact: PublishedArtifact
    execution: RecordReference | None


@dataclass(frozen=True)
class _ExecutionBinding:
    """The returned object for one observed Computation call."""

    value: object
    execution: RecordReference
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class ArtifactSetHandle:
    """SDK handle for one immutable ArtifactSet.

    ArtifactSets have no payload file to load. The handle therefore carries the
    published Core record and its digest-bound reference, allowing it to move
    through application code without a second publisher call. When the caller
    requests SDK-owned release-manifest materialization, ``manifest`` points to
    a durable sidecar JSON Artifact. It is deliberately *not* a set member:
    the sidecar records this set's exact digest-bound reference, which a member
    cannot do without creating a content-digest cycle.
    """

    artifact_set: ArtifactSet
    reference: RecordReference
    members: Mapping[str, ArtifactHandle] = field(default_factory=dict)
    manifest: ArtifactHandle | None = None

    def member(self, name: str) -> ArtifactHandle:
        """Return one named member handle from this exact ArtifactSet."""

        try:
            return self.members[name]
        except KeyError as error:
            available = ", ".join(sorted(self.members)) or "none"
            raise KeyError(
                f"ArtifactSet {self.artifact_set.id!r} has no member {name!r}; "
                f"available members: {available}"
            ) from error

    def load_member(
        self,
        name: str,
        target_type: object,
        *,
        adapters: ArtifactAdapterRegistry = DEFAULT_ARTIFACT_ADAPTERS,
    ) -> object:
        """Materialize one verified member through the SDK adapter registry."""

        return adapters.load(self.member(name), target_type)


@dataclass
class OclpRun:
    """Run-scoped SDK policy for decorated Artifact and Computation observation.

    ``namespace`` supplies the logical project identifier used for generated
    Artifact, Execution, and Event IDs. It intentionally is not part of the
    OCLP protocol: applications choose their own stable ID vocabulary.
    """

    publisher: LocalArtifactPublisher
    namespace: str
    run_id: str
    source: ImplementationSource
    parent_execution: RecordReference | None = None
    profiles: ProfileBindings | None = None
    artifact_adapters: ArtifactAdapterRegistry = DEFAULT_ARTIFACT_ADAPTERS
    _token: Token[OclpRun | None] | None = field(default=None, init=False, repr=False)
    _value_bindings: dict[int, _ValueBinding] = field(
        default_factory=dict, init=False, repr=False
    )
    _call_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _execution_outputs: dict[str, dict[str, ArtifactHandle]] = field(
        default_factory=dict, init=False, repr=False
    )
    _execution_bindings: dict[int, _ExecutionBinding] = field(
        default_factory=dict, init=False, repr=False
    )
    _execution_computations: dict[str, RecordReference] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.namespace.startswith("urn:"):
            raise ValueError("OCLP run namespaces must be URNs")
        if not self.run_id:
            raise ValueError("OCLP run IDs must be non-empty")

    def __enter__(self) -> OclpRun:
        self._token = _ACTIVE_RUN.set(self)
        return self

    def __exit__(self, *_: object) -> None:
        assert self._token is not None
        _ACTIVE_RUN.reset(self._token)
        self._token = None

    def artifact_for(
        self, value: object, *, port: str | None = None
    ) -> PublishedArtifact:
        """Return the Artifact materialized for an exact in-process value."""

        if isinstance(value, ArtifactHandle):
            return value.published
        binding = self._value_bindings.get(id(value))
        if binding is None or binding.value is not value:
            raise ValueError("value has no Artifact binding in this OCLP run")
        if port is not None and binding.port != port:
            raise ValueError(
                f"value is bound to output port {binding.port!r}, not {port!r}"
            )
        return binding.artifact

    def execution_for(self, value: object) -> RecordReference:
        """Return the Execution that materialized an exact returned value."""

        execution_binding = self._execution_bindings.get(id(value))
        if execution_binding is not None and execution_binding.value is value:
            return execution_binding.execution
        binding = self._value_bindings.get(id(value))
        if binding is None or binding.value is not value:
            raise ValueError("value has no Execution binding in this OCLP run")
        if binding.execution is None:
            raise ValueError(
                "value has an Artifact binding but no Execution; it was acquired, "
                "not produced by a Computation"
            )
        return binding.execution

    def computation_for(self, value: object) -> RecordReference:
        """Return the Computation materialized for an observed return value."""

        execution = self.execution_for(value)
        try:
            return self._execution_computations[execution.id]
        except KeyError as error:
            raise ValueError(
                "the Computation binding is unavailable in this OCLP run"
            ) from error

    def outputs_for(self, value: object) -> dict[str, ArtifactHandle]:
        """Return typed handles for all persisted outputs from one Execution.

        This lets a runner pass the exact materialized Artifact downstream.
        Artifact handles expose a payload that an SDK adapter can load on a
        consuming Computation.
        """

        execution = self.execution_for(value)
        try:
            return dict(self._execution_outputs[execution.id])
        except KeyError as error:
            raise ValueError(
                "the Execution output bindings are unavailable in this OCLP run"
            ) from error

    def publish_artifact_set(
        self,
        *,
        key: str,
        name: str,
        members: Mapping[str, tuple[ArtifactHandle, str | None]],
        materialize_manifest: bool = False,
        manifest_name: str | None = None,
    ) -> ArtifactSetHandle:
        """Publish a named collection of exact, previously published Artifacts.

        This is a publication operation, not a Computation: it creates no
        synthetic Execution or lifecycle Event. Each mapping key is the stable
        member name; its tuple contains the resolved Artifact handle and an
        optional semantic role. The resulting ArtifactSet is immutable and
        contains the members' digest-bound references.

        ``materialize_manifest=True`` additionally writes an SDK-owned
        ``release-manifest.json`` sidecar after publishing the ArtifactSet.
        It is not a member of the collection: the sidecar must carry the exact
        digest-bound ArtifactSet reference, while including it in the set would
        create an unsatisfiable self-content cycle. ``manifest_name`` is
        required in that mode because record labels are application-owned. Its
        payload snapshots the exact ArtifactSet record and resolved upstream
        OCLP record closure available to this local publisher. This remains a
        direct collection-publication operation: it does not fabricate a
        Computation, Execution, or Event.
        """

        if not isinstance(key, str) or not key:
            raise ValueError("ArtifactSet keys must be non-empty strings")
        if not isinstance(name, str) or not name:
            raise ValueError("ArtifactSet names must be non-empty strings")
        if not members:
            raise ValueError("ArtifactSets require at least one member")
        if materialize_manifest and (
            not isinstance(manifest_name, str) or not manifest_name
        ):
            raise ValueError(
                "materialized ArtifactSets require an application-supplied "
                "manifest_name"
            )
        if not materialize_manifest and manifest_name is not None:
            raise ValueError(
                "manifest_name is only valid when materialize_manifest=True"
            )

        resolved_members: list[ArtifactSetMember] = []
        member_handles: dict[str, ArtifactHandle] = {}
        for member_name, declaration in members.items():
            if not isinstance(member_name, str) or not member_name:
                raise ValueError("ArtifactSet member names must be non-empty strings")
            if not isinstance(declaration, tuple) or len(declaration) != 2:
                raise TypeError(
                    "ArtifactSet members must be (ArtifactHandle, optional role) tuples"
                )
            artifact, role = declaration
            if not isinstance(artifact, ArtifactHandle):
                raise TypeError("ArtifactSet members must use ArtifactHandle values")
            if role is not None and (not isinstance(role, str) or not role):
                raise ValueError("ArtifactSet member roles must be non-empty strings")
            resolved_members.append(
                ArtifactSetMember(
                    name=member_name,
                    artifact=artifact.reference,
                    role=role,
                )
            )
            member_handles[member_name] = artifact

        artifact_set_id = f"{self.namespace}:artifact-set:{key}:{self.run_id}"
        created_at = utc_now()
        artifact_set = ArtifactSet(
            id=artifact_set_id,
            name=name,
            profiles=self.profiles,
            created_at=created_at,
            members=tuple(resolved_members),
        )
        reference = self.publisher.publish(artifact_set)
        manifest: ArtifactHandle | None = None
        if materialize_manifest:
            assert manifest_name is not None  # validated above for type narrowing.
            manifest = ArtifactHandle(
                published=self.publisher.json_artifact(
                    artifact_id=(
                        f"{self.namespace}:artifact:{key}:release-manifest:"
                        f"{self.run_id}"
                    ),
                    name=manifest_name,
                    relative_path=_release_manifest_relative_path(artifact_set_id),
                    value=_release_manifest_document(
                        artifact_set=artifact_set,
                        artifact_set_reference=reference,
                        records=_release_record_closure(
                            self.publisher.records(),
                            tuple(member.artifact for member in resolved_members),
                        ),
                    ),
                    created_at=created_at,
                    profiles=self.profiles,
                )
            )
        return ArtifactSetHandle(
            artifact_set=artifact_set,
            reference=reference,
            members=member_handles,
            manifest=manifest,
        )

    def evidence_for(self, value: object) -> tuple[Evidence, ...]:
        """Return Evidence emitted for required Contracts on one Execution."""

        binding = self._execution_bindings.get(id(value))
        if binding is None or binding.value is not value:
            raise ValueError("value has no Execution binding in this OCLP run")
        return binding.evidence

    def acquire(
        self,
        function: Callable[..., object],
        artifact_type: ArtifactType,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        """Acquire one Artifact-decorated value without creating an Execution."""

        result = function(*args, **kwargs)
        base_name = _callable_key(function)
        call_index = self._call_counts.get(f"artifact:{base_name}", 0)
        self._call_counts[f"artifact:{base_name}"] = call_index + 1
        suffix = base_name if call_index == 0 else f"{base_name}-{call_index + 1}"
        artifact_id = artifact_type.resolve_id(
            function=function,
            args=args,
            kwargs=dict(kwargs),
            default=f"{self.namespace}:artifact:{suffix}:{self.run_id}",
        )
        if artifact_type.name is None:
            raise ValueError(
                "Artifact-producing decorators require an application-supplied "
                "ArtifactType name"
            )
        storage_name = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        published = self._persist_acquired_artifact(
            artifact_id=artifact_id,
            name=artifact_type.name,
            relative_path=f"acquired/{storage_name}.{artifact_type.suffix}",
            artifact_type=artifact_type,
            value=result,
        )
        handle = artifact_type.handle(published)
        self._value_bindings[id(handle)] = _ValueBinding(
            value=handle,
            port=suffix,
            artifact=published,
            execution=None,
        )
        return handle

    def invoke(
        self,
        function: Callable[..., object],
        template: ComputationTemplate,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        """Call, materialize, and observe one decorated Computation."""

        computation = computation_record(function, source=self.source)
        computation_ref = self.publisher.publish(computation)
        started_at = utc_now()
        stage = _stage_name(computation.id)
        call_index = self._call_counts.get(computation.id, 0)
        self._call_counts[computation.id] = call_index + 1
        suffix = stage if call_index == 0 else f"{stage}-{call_index + 1}"
        parameters, inputs = self._bindings_for_call(function, template, args, kwargs)

        try:
            invocation_args, invocation_kwargs = self._adapt_artifact_inputs(
                function,
                template,
                args,
                kwargs,
            )
            result = function(*invocation_args, **invocation_kwargs)
        except BaseException as error:
            execution, execution_ref = self._publish_execution(
                computation_ref=computation_ref,
                computation_name=computation.name,
                suffix=suffix,
                parameters=parameters,
                inputs=inputs,
                outputs=None,
                requested_outputs=_requested_output_names(template),
            )
            self._publish_failure_events(
                execution=execution,
                execution_ref=execution_ref,
                started_at=started_at,
                error=error,
            )
            raise

        try:
            output_values: dict[str, object] = {}
            materialized_outputs: dict[str, PublishedArtifact] = {}
            for port, spec in template.output_artifacts.items():
                value = _direct_output_value(
                    port=port,
                    result=result,
                    direct_port_count=len(template.output_artifacts),
                )
                output_values[port] = value
                materialized_outputs[port] = self._materialize_output(
                    port=port,
                    spec=spec,
                    value=value,
                    suffix=suffix,
                )
            outputs = {
                port: (artifact.reference,)
                for port, artifact in materialized_outputs.items()
            }
        except BaseException as error:
            execution, execution_ref = self._publish_execution(
                computation_ref=computation_ref,
                computation_name=computation.name,
                suffix=suffix,
                parameters=parameters,
                inputs=inputs,
                outputs=None,
                requested_outputs=_requested_output_names(template),
            )
            self._publish_failure_events(
                execution=execution,
                execution_ref=execution_ref,
                started_at=started_at,
                error=error,
            )
            raise

        execution, execution_ref = self._publish_execution(
            computation_ref=computation_ref,
            computation_name=computation.name,
            suffix=suffix,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            requested_outputs=_requested_output_names(template),
        )
        self._execution_computations[execution_ref.id] = computation_ref
        emitted_evidence = self._evaluate_required_evidence(
            template=template,
            result=result,
            execution_ref=execution_ref,
            suffix=suffix,
        )
        status = (
            "succeeded"
            if all(item.outcome == "pass" for item in emitted_evidence)
            else "failed"
        )
        self._publish_completion_events(
            execution=execution,
            execution_ref=execution_ref,
            started_at=started_at,
            evidence=emitted_evidence,
            status=status,
        )
        output_handles = {
            port: template.output_artifacts[port].handle(artifact)
            for port, artifact in materialized_outputs.items()
        }
        self._execution_outputs[execution_ref.id] = output_handles
        self._execution_bindings[id(result)] = _ExecutionBinding(
            value=result,
            execution=execution_ref,
            evidence=emitted_evidence,
        )
        for port, artifact in materialized_outputs.items():
            self._value_bindings[id(output_values[port])] = _ValueBinding(
                value=output_values[port],
                port=port,
                artifact=artifact,
                execution=execution_ref,
            )
        return result

    def _bindings_for_call(
        self,
        function: Callable[..., object],
        template: ComputationTemplate,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> tuple[dict[str, JsonValue], dict[str, tuple[RecordReference, ...]]]:
        """Bind one call to declared ports and inferred JSON parameters only."""

        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        parameters: dict[str, JsonValue] = {}
        inputs: dict[str, tuple[RecordReference, ...]] = {}
        declarations = computation_input_artifact_types(function)
        input_names = {port.name for port in template.input_ports}
        parameter_names = {
            parameter.name for parameter in template.parameter_definitions
        }
        for name, value in bound.arguments.items():
            if name in input_names:
                declaration = declarations.get(name)
                if isinstance(declaration, ArtifactSetInput):
                    if not isinstance(value, ArtifactSetHandle):
                        raise TypeError(
                            f"ArtifactSet input {name!r} on {function.__qualname__} "
                            "requires an ArtifactSetHandle"
                        )
                    inputs[name] = (value.reference,)
                    continue
                handles = _artifact_handles(value)
                if handles is not None:
                    inputs[name] = tuple(handle.reference for handle in handles)
                    continue
                binding = self._value_bindings.get(id(value))
                if binding is not None and binding.value is value:
                    inputs[name] = (binding.artifact.reference,)
                    continue
            if name in parameter_names:
                json_value = _json_value(value)
                if json_value is None and value is not None:
                    raise TypeError(
                        f"Computation parameter {name!r} must be JSON-compatible"
                    )
                parameters[name] = json_value
        return parameters, inputs

    def _adapt_artifact_inputs(
        self,
        function: Callable[..., object],
        template: ComputationTemplate,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        """Load Artifact handles into the callable's annotated runtime types."""

        bound = inspect.signature(function).bind(*args, **kwargs)
        type_hints = get_type_hints(function)
        declared_artifact_types = computation_input_artifact_types(function)
        ports = {port.name: port for port in template.input_ports}
        for name, value in bound.arguments.items():
            declaration = declared_artifact_types.get(name)
            if isinstance(declaration, ArtifactSetInput):
                if not isinstance(value, ArtifactSetHandle):
                    raise TypeError(
                        f"ArtifactSet input {name!r} on {function.__qualname__} "
                        "requires an ArtifactSetHandle"
                    )
                _validate_artifact_set_handle(
                    handle=value,
                    declaration=declaration,
                    function=function,
                    name=name,
                )
                continue
            handles = _artifact_handles(value)
            if handles is None:
                continue
            is_many = isinstance(declaration, ManyArtifacts)
            if is_many and not handles:
                raise ValueError(
                    f"Artifact input {name!r} on {function.__qualname__} "
                    "requires at least one Artifact"
                )
            if not is_many and len(handles) != 1:
                raise TypeError(
                    f"Artifact input {name!r} on {function.__qualname__} "
                    "requires one Artifact, not a collection"
                )
            expected_type = (
                declaration.artifact_type
                if isinstance(declaration, ManyArtifacts)
                else declaration
            )
            port = ports.get(name)
            if port is not None and port.cardinality == "many" and not is_many:
                raise TypeError(
                    f"Artifact input {name!r} on {function.__qualname__} "
                    "declares many Artifacts but has no many() SDK declaration"
                )
            typed_handles = tuple(
                _validate_artifact_handle(
                    handle=handle,
                    expected_type=expected_type,
                    port=port,
                    function=function,
                    name=name,
                )
                for handle in handles
            )
            try:
                target_type = type_hints[name]
            except KeyError as error:
                raise TypeError(
                    f"Artifact input {name!r} on {function.__qualname__} must "
                    "have an annotated runtime type"
                ) from error
            if is_many:
                item_type = _collection_item_type(
                    target_type,
                    name=name,
                    function=function,
                )
                loaded = tuple(
                    self.artifact_adapters.load(handle, item_type)
                    for handle in typed_handles
                )
                bound.arguments[name] = _collection_value(
                    target_type=target_type,
                    values=loaded,
                )
            else:
                bound.arguments[name] = self.artifact_adapters.load(
                    typed_handles[0], target_type
                )
        return tuple(bound.args), dict(bound.kwargs)

    def _materialize_output(
        self,
        *,
        port: str,
        spec: ArtifactType,
        value: object,
        suffix: str,
    ) -> PublishedArtifact:
        artifact_id = (
            f"{self.namespace}:artifact:{spec.key}:{self.run_id}"
            if spec.key is not None
            else f"{self.namespace}:artifact:{suffix}:{self.run_id}:{port}"
        )
        if spec.name is None:  # guarded when @computation is declared.
            raise ValueError(
                f"Computation output {port!r} requires an application-supplied "
                "ArtifactType name"
            )
        relative_path = spec.path or f"{suffix}/{port}.{spec.suffix}"
        return spec.persist(
            publisher=self.publisher,
            artifact_id=artifact_id,
            name=spec.name,
            relative_path=relative_path,
            value=value,
            created_at=utc_now(),
        )

    def _persist_acquired_artifact(
        self,
        *,
        artifact_id: str,
        name: str,
        relative_path: str,
        artifact_type: ArtifactType,
        value: object,
    ) -> PublishedArtifact:
        return artifact_type.persist(
            publisher=self.publisher,
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            value=value,
            created_at=utc_now(),
        )

    def _publish_execution(
        self,
        *,
        computation_ref: RecordReference,
        computation_name: str,
        suffix: str,
        parameters: dict[str, JsonValue],
        inputs: dict[str, tuple[RecordReference, ...]],
        outputs: dict[str, tuple[RecordReference, ...]] | None,
        requested_outputs: tuple[str, ...],
    ) -> tuple[Execution, RecordReference]:
        execution = Execution(
            id=f"{self.namespace}:execution:{suffix}:{self.run_id}",
            # Executions inherit the explicit, application-owned Computation
            # label. IDs and lifecycle profiles identify the exact call.
            name=computation_name,
            profiles=self.profiles,
            computation=computation_ref,
            parent_execution=self.parent_execution,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            requested_outputs=requested_outputs,
        )
        return execution, self.publisher.publish(execution)

    def _evaluate_required_evidence(
        self,
        *,
        template: ComputationTemplate,
        result: object,
        execution_ref: RecordReference,
        suffix: str,
    ) -> tuple[Evidence, ...]:
        """Evaluate all required Evidence evaluators against same-named outputs."""

        emitted: list[Evidence] = []
        for index, evaluator in enumerate(template.required_evaluators or ()):
            template = evidence_template(evaluator)
            evidence_id = (
                f"{self.namespace}:evidence:{suffix}:{self.run_id}:"
                f"{_evidence_key(evaluator, index)}"
            )
            try:
                value = _evidence_value(evaluator=evaluator, result=result)
            except Exception as error:
                record = evaluate_evidence(
                    evaluator,
                    id=evidence_id,
                    name=template.name,
                    subject=execution_ref,
                    observed_at=utc_now(),
                    source=self.source,
                    evaluation_error=error,
                )
            else:
                record = evaluate_evidence(
                    evaluator,
                    value,
                    id=evidence_id,
                    name=template.name,
                    subject=execution_ref,
                    observed_at=utc_now(),
                    source=self.source,
                )
            self.publisher.publish(record)
            emitted.append(record)
        return tuple(emitted)

    def _publish_completion_events(
        self,
        *,
        execution: Execution,
        execution_ref: RecordReference,
        started_at: datetime,
        evidence: tuple[Evidence, ...],
        status: str,
    ) -> None:
        prefix = execution.id.removeprefix(self.namespace + ":execution:")
        self.publisher.publish(
            Event(
                id=f"{self.namespace}:event:{prefix}:started",
                execution=execution_ref,
                event_type="execution-started",
                occurred_at=started_at,
                sequence=0,
            )
        )
        self.publisher.publish(
            Event(
                id=f"{self.namespace}:event:{prefix}:artifacts-published",
                execution=execution_ref,
                event_type="artifacts-published",
                occurred_at=utc_now(),
                sequence=1,
                data={"outputs": _references_json(execution.outputs or {})},
            )
        )
        for index, record in enumerate(evidence, start=2):
            self.publisher.publish(
                Event(
                    id=f"{self.namespace}:event:{prefix}:evidence-{index - 1}",
                    execution=execution_ref,
                    event_type="evidence-published",
                    occurred_at=utc_now(),
                    sequence=index,
                    data={"evidence": record.model_dump(mode="json")},
                )
            )
        self.publisher.publish(
            Event(
                id=f"{self.namespace}:event:{prefix}:terminal",
                execution=execution_ref,
                event_type="execution-terminal",
                occurred_at=utc_now(),
                sequence=2 + len(evidence),
                status=status,
            )
        )

    def _publish_failure_events(
        self,
        *,
        execution: Execution,
        execution_ref: RecordReference,
        started_at: datetime,
        error: BaseException,
    ) -> None:
        prefix = execution.id.removeprefix(self.namespace + ":execution:")
        self.publisher.publish(
            Event(
                id=f"{self.namespace}:event:{prefix}:started",
                execution=execution_ref,
                event_type="execution-started",
                occurred_at=started_at,
                sequence=0,
            )
        )
        self.publisher.publish(
            Event(
                id=f"{self.namespace}:event:{prefix}:terminal",
                execution=execution_ref,
                event_type="execution-terminal",
                occurred_at=utc_now(),
                sequence=1,
                status="failed",
                diagnostic=Diagnostic(
                    code=type(error).__name__,
                    message=str(error) or type(error).__name__,
                    stage="execution",
                ),
            )
        )


def _release_manifest_relative_path(artifact_set_id: str) -> str:
    """Return a collision-safe, SDK-owned payload path for one manifest."""

    storage_key = hashlib.sha256(artifact_set_id.encode("utf-8")).hexdigest()
    return f"release/{storage_key}/release-manifest.json"


def _release_manifest_document(
    *,
    artifact_set: ArtifactSet,
    artifact_set_reference: RecordReference,
    records: Iterable[OclpRecord],
) -> dict[str, JsonValue]:
    """Build the portable sidecar payload for one exact ArtifactSet.

    The manifest is not an ArtifactSet member, so it can safely carry both the
    canonical ArtifactSet body and its final digest-bound reference. A local or
    remote service can consequently select the entire release as an immutable
    input rather than reconstructing it from a model Artifact plus an
    application-specific string parameter.
    """

    return {
        "oclp_release_manifest_version": "0.2",
        "artifact_set": {
            "reference": artifact_set_reference.model_dump(mode="json"),
            "record": json.loads(canonical_json_bytes(artifact_set)),
        },
        "records": [
            {
                "reference": RecordReference(
                    id=record.id,
                    digest=record_digest(record),
                ).model_dump(mode="json"),
                "record": json.loads(canonical_json_bytes(record)),
            }
            for record in records
        ],
    }


def load_release_manifest(release_manifest_path: Path) -> ArtifactSetHandle:
    """Load one portable release sidecar into an exact local ArtifactSet handle.

    This is SDK infrastructure, not an application-specific service adapter.
    It verifies the manifest's ArtifactSet record against its digest-bound
    reference, resolves every member from the manifest closure, and admits
    only locally available ``file:`` payloads. Applications can then pass the
    returned handle directly to a ``artifact_set_input(...)`` Computation.
    """

    path = release_manifest_path.expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"release manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"release manifest is not valid JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("release manifest must be a JSON object")

    raw_set = document.get("artifact_set")
    if not isinstance(raw_set, dict):
        raise ValueError("release manifest is missing its artifact_set")
    try:
        artifact_set_reference = RecordReference.model_validate(
            raw_set.get("reference")
        )
    except Exception as error:
        raise ValueError(
            "release manifest artifact_set must have a valid digest-bound reference"
        ) from error
    from oclp.validation import parse_record

    try:
        artifact_set_record = parse_record(raw_set.get("record"))
    except Exception as error:
        raise ValueError("release manifest ArtifactSet record is invalid") from error
    if not isinstance(artifact_set_record, ArtifactSet):
        raise ValueError("release manifest artifact_set record must be an ArtifactSet")
    if (
        artifact_set_record.id != artifact_set_reference.id
        or record_digest(artifact_set_record) != artifact_set_reference.digest
    ):
        raise ValueError(
            "release manifest ArtifactSet record does not match its digest-bound "
            "reference"
        )

    raw_records = document.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("release manifest records must be a list")
    artifact_records: dict[RecordReference, Artifact] = {}
    for entry in raw_records:
        if not isinstance(entry, dict):
            raise ValueError("release manifest records must contain JSON objects")
        try:
            reference = RecordReference.model_validate(entry.get("reference"))
            record = parse_record(entry.get("record"))
        except Exception as error:
            raise ValueError(
                "release manifest contains an invalid record entry"
            ) from error
        if record.id != reference.id or record_digest(record) != reference.digest:
            raise ValueError(
                "release manifest record entry does not match its digest-bound "
                "reference"
            )
        if isinstance(record, Artifact):
            artifact_records[reference] = record

    handles: dict[str, ArtifactHandle] = {}
    for member in artifact_set_record.members:
        artifact = artifact_records.get(member.artifact)
        if artifact is None:
            raise ValueError(
                f"release manifest lacks Artifact record for member {member.name!r}"
            )
        handles[member.name] = ArtifactHandle(
            published=PublishedArtifact(
                artifact=artifact,
                path=_local_manifest_artifact_path(artifact),
                reference=member.artifact,
            )
        )

    return ArtifactSetHandle(
        artifact_set=artifact_set_record,
        reference=artifact_set_reference,
        members=handles,
    )


def _local_manifest_artifact_path(artifact: Artifact) -> Path:
    """Return an existing local file location for one manifest Artifact."""

    for location in artifact.locations:
        parsed = urlparse(location)
        if parsed.scheme != "file":
            continue
        candidate = Path(unquote(parsed.path))
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"release Artifact {artifact.id!r} has no available local file location"
    )


def _release_record_closure(
    records: Iterable[OclpRecord],
    member_references: tuple[RecordReference, ...],
) -> tuple[OclpRecord, ...]:
    """Return resolved upstream provenance and lifecycle records for members.

    A manifest is intentionally a snapshot, not another catalog. It carries
    the member Artifacts plus their producing Executions, inputs,
    Computations, Evidence, and Events when those records are available in the
    local publisher. References that resolve only in an external store remain
    present in the included record bodies but cannot contribute a second body
    to this local closure.
    """

    available = tuple(records)
    by_digest = {record_digest(record).value: record for record in available}
    producers_by_output: dict[str, set[str]] = defaultdict(set)
    observations_by_execution: dict[str, set[str]] = defaultdict(set)
    for record in available:
        digest = record_digest(record).value
        if isinstance(record, Execution):
            for references in (record.outputs or {}).values():
                for reference in references:
                    if reference.digest is not None:
                        producers_by_output[reference.digest.value].add(digest)
        elif isinstance(record, Evidence):
            if record.subject.digest is not None:
                observations_by_execution[record.subject.digest.value].add(digest)
        elif isinstance(record, Event):
            if record.execution.digest is not None:
                observations_by_execution[record.execution.digest.value].add(digest)

    pending = {
        reference.digest.value
        for reference in member_references
        if reference.digest is not None and reference.digest.value in by_digest
    }
    included: set[str] = set()
    while pending:
        current_digest = min(pending)
        pending.remove(current_digest)
        if current_digest in included:
            continue
        record = by_digest.get(current_digest)
        if record is None:
            continue
        included.add(current_digest)
        for reference in _release_record_references(record):
            if reference.digest is not None and reference.digest.value in by_digest:
                pending.add(reference.digest.value)
        if isinstance(record, Artifact):
            pending.update(producers_by_output.get(current_digest, ()))
        if isinstance(record, Execution):
            pending.update(observations_by_execution.get(current_digest, ()))

    return tuple(
        sorted(
            (by_digest[digest] for digest in included),
            key=lambda record: (record.kind, record.id, record_digest(record).value),
        )
    )


def _release_record_references(record: OclpRecord) -> tuple[RecordReference, ...]:
    """Return standard Core record references needed for a release snapshot."""

    references: list[RecordReference] = []
    if isinstance(record, ArtifactSet):
        references.extend(member.artifact for member in record.members)
    elif isinstance(record, Computation):
        references.extend(_implementation_references(record.implementation))
    elif isinstance(record, Execution):
        references.append(record.computation)
        if record.parent_execution is not None:
            references.append(record.parent_execution)
        references.extend(
            reference
            for bindings in record.inputs.values()
            for reference in bindings
        )
        references.extend(
            reference
            for bindings in (record.outputs or {}).values()
            for reference in bindings
        )
    elif isinstance(record, Evidence):
        references.append(record.subject)
        references.extend(_implementation_references(record.evaluator))
        references.extend(_diagnostic_references(record.diagnostic))
    elif isinstance(record, Event):
        references.append(record.execution)
        references.extend(_diagnostic_references(record.diagnostic))
    return tuple(references)


def _implementation_references(
    implementation: Implementation,
) -> tuple[RecordReference, ...]:
    """Return record-backed source dependencies declared by an Implementation."""

    references: list[RecordReference] = []
    if implementation.artifact is not None:
        references.append(implementation.artifact)
    source = implementation.source
    if isinstance(source, ArtifactSource):
        references.append(source.artifact)
    elif isinstance(source, GitSource) and source.overlay is not None:
        references.append(source.overlay)
    return tuple(references)


def _diagnostic_references(
    diagnostic: Diagnostic | None,
) -> tuple[RecordReference, ...]:
    """Return a diagnostic's optional durable detail Artifact binding."""

    return (diagnostic.artifact,) if diagnostic and diagnostic.artifact else ()


def active_run() -> OclpRun | None:
    """Return the ambient OCLP run, if automatic observation is active."""

    return _ACTIVE_RUN.get()


def run(**kwargs: Any) -> OclpRun:
    """Create the explicit runtime context used by decorated Computations."""

    return OclpRun(**kwargs)


def _requested_output_names(template: ComputationTemplate) -> tuple[str, ...]:
    """Return every output port the runtime is responsible for publishing."""

    return tuple(template.output_artifacts)


def _direct_output_value(
    *,
    port: str,
    result: object,
    direct_port_count: int,
) -> object:
    """Resolve a direct output from a callable's ordinary domain result."""

    if isinstance(result, Mapping) and port in result:
        return result[port]
    try:
        return getattr(result, port)
    except AttributeError:
        pass
    if direct_port_count == 1:
        return result
    raise ValueError(
        f"multi-output Computation result has no field for port {port!r}; "
        "return an object with same-named fields or a mapping by port"
    )


def _artifact_handles(value: object) -> tuple[ArtifactHandle, ...] | None:
    """Return one or many SDK Artifact handles without treating raw values as inputs."""

    if isinstance(value, ArtifactHandle):
        return (value,)
    if isinstance(value, tuple | list) and all(
        isinstance(item, ArtifactHandle) for item in value
    ):
        return tuple(value)
    return None


def _validate_artifact_handle(
    *,
    handle: ArtifactHandle,
    expected_type: type[ArtifactType] | None,
    port: object,
    function: Callable[..., object],
    name: str,
) -> ArtifactHandle:
    """Check that a resolved payload satisfies its declared Artifact type."""

    if (
        expected_type is not None
        and expected_type.media_types
        and handle.artifact.media_type not in expected_type.media_types
    ):
        accepted = ", ".join(expected_type.media_types)
        raise TypeError(
            f"Artifact input {name!r} on {function.__qualname__} requires "
            f"{expected_type.__name__} ({accepted}), not "
            f"{handle.artifact.media_type!r}"
        )
    media_types = getattr(port, "media_types", ())
    if media_types and handle.artifact.media_type not in media_types:
        accepted = ", ".join(media_types)
        raise TypeError(
            f"Artifact input {name!r} on {function.__qualname__} has media type "
            f"{handle.artifact.media_type!r}; expected one of {accepted}"
        )
    return handle


def _validate_artifact_set_handle(
    *,
    handle: ArtifactSetHandle,
    declaration: ArtifactSetInput,
    function: Callable[..., object],
    name: str,
) -> None:
    """Verify the named, typed members required by one ArtifactSet input."""

    declared_members = {member.name: member for member in handle.artifact_set.members}
    for member_name, expected_type in declaration.members.items():
        member = declared_members.get(member_name)
        if member is None:
            raise TypeError(
                f"ArtifactSet input {name!r} on {function.__qualname__} requires "
                f"member {member_name!r}, but {handle.artifact_set.id!r} does not "
                "declare it"
            )
        resolved = handle.members.get(member_name)
        if resolved is None:
            raise TypeError(
                f"ArtifactSet input {name!r} on {function.__qualname__} requires "
                f"a locally resolvable member {member_name!r}"
            )
        if resolved.reference != member.artifact:
            raise TypeError(
                f"ArtifactSet input {name!r} member {member_name!r} does not match "
                "the ArtifactSet's digest-bound reference"
            )
        _validate_artifact_handle(
            handle=resolved,
            expected_type=expected_type,
            port=None,
            function=function,
            name=f"{name}.{member_name}",
        )


def _collection_item_type(
    target_type: object,
    *,
    name: str,
    function: Callable[..., object],
) -> object:
    """Return the runtime element annotation for a many-Artifact parameter."""

    origin = get_origin(target_type)
    arguments = get_args(target_type)
    if origin not in (tuple, list) or not arguments:
        raise TypeError(
            f"many Artifact input {name!r} on {function.__qualname__} must be "
            "annotated as tuple[T, ...] or list[T]"
        )
    if origin is tuple and len(arguments) == 2 and arguments[1] is Ellipsis:
        return arguments[0]
    if origin is list and len(arguments) == 1:
        return arguments[0]
    raise TypeError(
        f"many Artifact input {name!r} on {function.__qualname__} must be "
        "annotated as tuple[T, ...] or list[T]"
    )


def _collection_value(*, target_type: object, values: tuple[object, ...]) -> object:
    """Return loaded values in the collection form requested by the callable."""

    if get_origin(target_type) is list:
        return list(values)
    return values


def _evidence_value(*, evaluator: Callable[..., object], result: object) -> object:
    """Select a required evaluator's subject from a direct Computation result.

    An Evidence evaluator has one ordinary parameter. For a multi-output
    Computation, that parameter must name the output it evaluates. This makes
    the Evidence decorator itself the local, readable contract binding—no
    second selector declaration is needed on ``@computation``.
    """

    parameters = tuple(inspect.signature(evaluator).parameters.values())
    required = tuple(
        parameter
        for parameter in parameters
        if parameter.default is parameter.empty
        and parameter.kind
        in (
            parameter.POSITIONAL_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.KEYWORD_ONLY,
        )
    )
    if len(required) != 1 or len(parameters) != 1:
        raise TypeError(
            "required Evidence evaluators must accept exactly one parameter"
        )
    output_name = required[0].name
    if isinstance(result, Mapping):
        try:
            return result[output_name]
        except KeyError as error:
            raise ValueError(
                f"Evidence evaluator {evaluator.__qualname__} expects output "
                f"{output_name!r}, but the Computation result has no such key"
            ) from error
    return result


def _evidence_key(evaluator: Callable[..., object], index: int) -> str:
    """Build a readable, collision-safe Evidence suffix from an evaluator."""

    suffix = _callable_key(evaluator)
    return f"{suffix}-{index + 1}"


def _json_value(value: object) -> JsonValue | None:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return None


def _stage_name(computation_id: str) -> str:
    marker = ":computation:"
    if marker in computation_id:
        return computation_id.rsplit(marker, maxsplit=1)[1]
    return computation_id.rsplit(":", maxsplit=1)[-1]


def _callable_key(function: Callable[..., object]) -> str:
    name = getattr(function, "__name__", None)
    if not isinstance(name, str) or not name:
        raise TypeError("Artifact-decorated callable must expose a stable __name__")
    return name.replace("_", "-")


def _references_json(
    references: Mapping[str, tuple[RecordReference, ...]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        port: [reference.model_dump(mode="json") for reference in values]
        for port, values in references.items()
    }

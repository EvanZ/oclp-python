"""Optional Dagster runtime integration for canonical OCLP declarations."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar, cast
from urllib.parse import unquote, urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, ValidationError

from oclp.artifacts import ArtifactHandle, artifact_type
from oclp.computations import artifact_set_assembly_name, computation_template
from oclp.declaration_adapters import DeclarationKind
from oclp.models import (
    Artifact,
    ArtifactSource,
    GitSource,
    ImplementationSource,
    OclpModel,
    OpaqueSource,
    ProfileBindings,
    RecordReference,
    ServiceSource,
)
from oclp.profiles.lifecycle import Lifecycle, LifecycleInput, coerce_lifecycle
from oclp.profiles.run import RUN_PROFILE
from oclp.publishing import LocalArtifactPublisher, PublishedArtifact
from oclp.runtime import (
    ArtifactSetHandle,
    OclpRun,
    RunAdapter,
    active_run,
    observe_run,
    run,
)

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
DagsterContext: TypeAlias = Any
PublisherFactory: TypeAlias = Callable[
    [DagsterContext], AbstractContextManager[LocalArtifactPublisher]
]
SourceFactory: TypeAlias = Callable[[DagsterContext], ImplementationSource]
ApplicationProfiles: TypeAlias = (
    ProfileBindings | Callable[[DagsterContext], ProfileBindings]
)
LifecycleDeclaration: TypeAlias = (
    LifecycleInput | Callable[[DagsterContext], LifecycleInput]
)
DagsterRunName: TypeAlias = str | Callable[[DagsterContext], str]
DagsterOutputBindings: TypeAlias = Mapping[str, str]
"""Map native Dagster output names to differently named OCLP output ports."""

_SOURCE_TYPES = (GitSource, ArtifactSource, ServiceSource, OpaqueSource)

DAGSTER_PROFILE = "dagster"
"""Execution profile key used by the Dagster runtime integration."""

DAGSTER_PROFILE_VERSION = "0.3.0-draft"
"""Version of the SDK-owned Dagster execution-profile binding."""

OCLP_DAGSTER_RESOURCE_KEY = "oclp"
"""Conventional Dagster resource key used by :func:`dagster_adapter`."""


class DagsterExecutionProfile(OclpModel):
    """Durable scheduler facts attached to one observed OCLP Execution."""

    version: Literal["0.3.0-draft"] = DAGSTER_PROFILE_VERSION
    dagster_run_id: str = Field(min_length=1)
    asset_key: str = Field(min_length=1)
    asset_keys: tuple[str, ...] = ()
    step_key: str = Field(min_length=1)
    partition_key: str | None = None
    retry_number: int = Field(ge=0)


@dataclass(frozen=True)
class OclpDagsterResource:
    """Project-level runtime configuration for canonical OCLP declarations."""

    publisher: PublisherFactory
    source: ImplementationSource | SourceFactory
    strict: bool = False
    adapters: tuple[RunAdapter, ...] = ()
    lifecycle: LifecycleDeclaration | None = None
    profiles: ApplicationProfiles | None = None

    def __post_init__(self) -> None:
        _validate_resource_configuration(publisher=self.publisher, source=self.source)
        if not isinstance(self.strict, bool):
            raise TypeError("OclpDagsterResource strict must be a boolean")
        if not isinstance(self.adapters, tuple):
            raise TypeError("OclpDagsterResource adapters must be a tuple")

    def source_for(self, context: DagsterContext) -> ImplementationSource:
        """Resolve the implementation source for one asset attempt."""

        return _resolve_source(self.source, context)


def oclp_dagster_resource(
    *,
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    strict: bool = False,
    adapters: tuple[RunAdapter, ...] = (),
    lifecycle: LifecycleDeclaration | None = None,
    profiles: ApplicationProfiles | None = None,
) -> object:
    """Create the ``"oclp"`` resource required by :func:`dagster_adapter`."""

    dagster = _require_dagster()
    resource = OclpDagsterResource(
        publisher=publisher,
        source=source,
        strict=strict,
        adapters=adapters,
        lifecycle=lifecycle,
        profiles=profiles,
    )
    return dagster.ResourceDefinition.hardcoded_resource(resource)


@dataclass(frozen=True)
class DagsterRunContext:
    """Rehydratable OCLP facts for one Dagster asset execution."""

    run_id: UUID
    profile: DagsterExecutionProfile

    @property
    def profiles(self) -> ProfileBindings:
        return {DAGSTER_PROFILE: self.profile.model_dump(mode="json")}


@dataclass
class DagsterAdapter:
    """Attach the exact observed OCLP records to a Dagster materialization."""

    context: DagsterContext
    context_fields: tuple[str, ...]
    strict: bool = False
    artifact: ArtifactHandle | None = None
    execution_id: str | None = None
    outputs: dict[str, ArtifactHandle] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)
    artifact_set: ArtifactSetHandle | None = None

    def on_artifact(self, _observed: OclpRun, artifact: ArtifactHandle) -> None:
        """Remember the real Artifact acquired by one native asset."""

        self.artifact = artifact

    def on_execution(
        self,
        _observed: OclpRun,
        *,
        execution: object,
        computation: object,
        outputs: dict[str, ArtifactHandle],
        evidence: tuple[object, ...],
    ) -> None:
        """Remember a Computation's Execution and named output Artifacts."""

        execution_id = getattr(execution, "id", None)
        if not isinstance(execution_id, str) or not execution_id:
            raise TypeError("Dagster adapter received an Execution without an ID")
        self.execution_id = execution_id
        self.outputs = dict(outputs)

    def on_artifact_set(
        self,
        _observed: OclpRun,
        artifact_set: ArtifactSetHandle,
    ) -> None:
        """Remember a terminal cross-run ArtifactSet assembly."""

        self.artifact_set = artifact_set

    def on_run_end(self, observed: OclpRun, error: BaseException | None) -> None:
        metadata: dict[str, object] = {
            "oclp.run.id": str(observed.run_id),
            "oclp.run.name": observed.run_name or "",
            "oclp.record_root": str(observed.publisher.record_root),
            "oclp.status": "failed" if error is not None else "succeeded",
        }
        if self.artifact is not None:
            metadata["oclp.artifact.id"] = self.artifact.reference.id
        if self.execution_id is not None:
            metadata["oclp.execution.id"] = self.execution_id
        if self.artifact_set is not None:
            metadata["oclp.artifact_set.id"] = self.artifact_set.reference.id
        for context_field in self.context_fields:
            value = _context_value(self.context, context_field)
            if value is not None:
                metadata[f"oclp.dagster.{context_field}"] = value
        keys_by_output = getattr(
            getattr(self.context, "assets_def", None),
            "keys_by_output_name",
            None,
        )
        if isinstance(keys_by_output, Mapping):
            for output_name, asset_key in keys_by_output.items():
                output_metadata = dict(metadata)
                port = self.output_bindings.get(output_name)
                artifact = self.outputs.get(port) if port is not None else None
                if artifact is not None:
                    output_metadata[f"oclp.output.{port}.id"] = artifact.reference.id
                self.context.add_asset_metadata(output_metadata, asset_key=asset_key)
            return
        for port, artifact in self.outputs.items():
            metadata[f"oclp.output.{port}.id"] = artifact.reference.id
        self.context.add_asset_metadata(metadata)


@dataclass(frozen=True)
class DagsterDeclarationAdapter:
    """Observe a canonical declaration when a native Dagster asset invokes it."""

    lifecycle: LifecycleDeclaration | None = None
    profiles: ApplicationProfiles | None = None
    run_name: DagsterRunName | None = None
    output_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.run_name is not None and not (
            isinstance(self.run_name, str) or callable(self.run_name)
        ):
            raise TypeError("dagster_adapter run_name must be a string or callable")
        if any(
            not isinstance(dagster_output, str)
            or not dagster_output
            or not isinstance(oclp_port, str)
            or not oclp_port
            for dagster_output, oclp_port in self.output_bindings
        ):
            raise ValueError(
                "dagster_adapter output_bindings must map non-empty Dagster "
                "output names to non-empty OCLP output ports"
            )
        if len({dagster_output for dagster_output, _ in self.output_bindings}) != len(
            self.output_bindings
        ):
            raise ValueError("dagster_adapter output_bindings keys must be unique")
        if len({oclp_port for _, oclp_port in self.output_bindings}) != len(
            self.output_bindings
        ):
            raise ValueError("dagster_adapter output_bindings values must be unique")

    def wrap(
        self,
        function: Callable[Parameters, Result],
        *,
        kind: DeclarationKind,
    ) -> Callable[Parameters, Result]:
        if kind != "computation" and self.output_bindings:
            raise ValueError(
                "dagster_adapter output_bindings apply only to @computation"
            )
        signature = inspect.signature(function)
        public_signature = _dagster_asset_signature(
            signature,
            _optional_dagster(),
            function=function,
            kind=kind,
        )

        @wraps(function)
        def observed(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            context, invocation_args, invocation_kwargs = _dagster_invocation(
                signature=signature,
                public_signature=public_signature,
                args=args,
                kwargs=kwargs,
            )
            if context is None or active_run() is not None:
                return function(*invocation_args, **invocation_kwargs)

            resource = _oclp_resource(context)
            step = _dagster_declaration_run_context(context)
            output_bindings = (
                _resolve_dagster_output_bindings(
                    context=context,
                    function=function,
                    configured_bindings=self.output_bindings,
                )
                if kind == "computation"
                else {}
            )
            record_profiles = _resolve_application_profiles(
                self.profiles if self.profiles is not None else resource.profiles,
                context,
            )
            runtime_adapter = DagsterAdapter(
                context=context,
                context_fields=(
                    "run_id",
                    "asset_key",
                    "step_key",
                    "partition_key",
                    "retry_number",
                ),
                strict=resource.strict,
                output_bindings=output_bindings,
            )
            with resource.publisher(context) as publisher:
                with observe_run(
                    _dagster_asset_workflow(
                        context,
                        function=function,
                        kind=kind,
                        run_name=self.run_name,
                    ),
                    publisher=publisher,
                    run_id=step.run_id,
                    source=resource.source_for(context),
                    profiles=_execution_profiles(step, record_profiles),
                    record_profiles=record_profiles,
                    lifecycle=_resolve_lifecycle(
                        (
                            self.lifecycle
                            if self.lifecycle is not None
                            else resource.lifecycle
                        ),
                        context=context,
                    ),
                    adapters=(*resource.adapters, runtime_adapter),
                    finalize_decorated_artifact_sets=False,
                ) as observed_run:
                    result = function(*invocation_args, **invocation_kwargs)
                    if kind == "artifact_set":
                        return cast(Result, None)
                    if kind != "computation":
                        return cast(Result, result)
                    return cast(
                        Result,
                        _dagster_computation_result(
                            result=result,
                            function=function,
                            observed=observed_run,
                            context=context,
                            output_bindings=output_bindings,
                        ),
                    )

        observed.__signature__ = public_signature
        annotations = {
            parameter.name: parameter.annotation
            for parameter in public_signature.parameters.values()
            if parameter.annotation is not inspect.Parameter.empty
        }
        if public_signature.return_annotation is not inspect.Parameter.empty:
            annotations["return"] = public_signature.return_annotation
        observed.__annotations__ = annotations
        return observed


def dagster_adapter(
    *,
    lifecycle: LifecycleDeclaration | None = None,
    profiles: ApplicationProfiles | None = None,
    run_name: DagsterRunName | None = None,
    output_bindings: DagsterOutputBindings | None = None,
) -> DagsterDeclarationAdapter:
    """Attach Dagster observation inside a canonical OCLP decorator.

    Use it as ``adapters=(dagster_adapter(),)`` inside an OCLP Artifact,
    Computation, or cross-run ArtifactSet declaration, with native
    ``@dagster.asset`` outside that declaration. By default its OCLP Run name
    is the canonical Artifact, Computation, or ArtifactSet declaration name.
    Supply ``run_name`` when an application needs a context-specific label,
    such as a dynamically partitioned temporal fold.

    Shared multi-asset outputs use the same local name by default. Supply
    ``output_bindings`` only when a native Dagster output needs a differently
    named OCLP output port: ``{"dagster_output": "oclp_port"}``. Unbound
    Dagster outputs remain ordinary native values; unbound OCLP outputs remain
    materialized OCLP Artifacts.
    """

    normalized_output_bindings = _normalize_output_bindings(output_bindings)

    return DagsterDeclarationAdapter(
        lifecycle=lifecycle,
        profiles=profiles,
        run_name=run_name,
        output_bindings=normalized_output_bindings,
    )


def _normalize_output_bindings(
    output_bindings: DagsterOutputBindings | None,
) -> tuple[tuple[str, str], ...]:
    if output_bindings is None:
        return ()
    if not isinstance(output_bindings, Mapping):
        raise TypeError("dagster_adapter output_bindings must be a mapping or None")
    return tuple(output_bindings.items())


def _resolve_dagster_output_bindings(
    *,
    context: DagsterContext,
    function: Callable[..., object],
    configured_bindings: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Resolve automatic and application-specified native/OCLP output bridges."""

    dagster_outputs = _dagster_output_names(context)
    if not dagster_outputs:
        return {}
    oclp_outputs = tuple(computation_template(function).output_artifacts)
    available_dagster_outputs = set(dagster_outputs)
    available_oclp_outputs = set(oclp_outputs)
    bindings = dict(configured_bindings)

    unknown_dagster_outputs = sorted(
        set(bindings).difference(available_dagster_outputs)
    )
    if unknown_dagster_outputs:
        raise ValueError(
            "dagster_adapter output_bindings names native Dagster outputs that "
            "are not declared: "
            + ", ".join(unknown_dagster_outputs)
        )
    unknown_oclp_outputs = sorted(
        set(bindings.values()).difference(available_oclp_outputs)
    )
    if unknown_oclp_outputs:
        raise ValueError(
            "dagster_adapter output_bindings names OCLP output ports that are "
            "not declared: "
            + ", ".join(unknown_oclp_outputs)
        )

    bound_oclp_outputs = set(bindings.values())
    for output_name in dagster_outputs:
        if (
            output_name not in bindings
            and output_name in available_oclp_outputs
            and output_name not in bound_oclp_outputs
        ):
            bindings[output_name] = output_name
            bound_oclp_outputs.add(output_name)

    # A standard @dg.asset has the native output name "result", while a
    # canonical single-output Computation uses its meaningful OCLP port name.
    # That one-to-one bridge is unambiguous without a configuration mapping.
    if not bindings and len(dagster_outputs) == 1 and len(oclp_outputs) == 1:
        bindings[dagster_outputs[0]] = oclp_outputs[0]

    return bindings


def _dagster_output_names(context: DagsterContext) -> tuple[str, ...]:
    keys_by_output = getattr(
        getattr(context, "assets_def", None),
        "keys_by_output_name",
        None,
    )
    if not isinstance(keys_by_output, Mapping):
        return ()
    output_names = tuple(keys_by_output)
    if not all(isinstance(name, str) and name for name in output_names):
        raise TypeError("Dagster asset output names must be non-empty strings")
    return output_names


def _dagster_asset_signature(
    signature: inspect.Signature,
    dagster: Any | None,
    *,
    function: Callable[..., object],
    kind: DeclarationKind,
) -> inspect.Signature:
    """Expose a Dagster context and required callable inputs to the scheduler."""

    context = inspect.Parameter(
        "context",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=dagster.AssetExecutionContext if dagster is not None else Any,
    )
    parameters = [
        context,
        *(
            parameter.replace(annotation=Any)
            for name, parameter in signature.parameters.items()
            if name != "context" and parameter.default is inspect.Parameter.empty
        ),
    ]
    if kind == "computation":
        # Native and OCLP outputs may only partially overlap. The native
        # decorator owns the eventual Dagster return shape, so OCLP output
        # count cannot truthfully describe this wrapper's return annotation.
        # Leave it absent so @dg.multi_asset infers every output from its own
        # explicit `outs` mapping.
        return_annotation = inspect.Parameter.empty
    elif kind in {"artifact", "artifact_set"}:
        return_annotation = Any
    else:  # pragma: no cover - DeclarationKind is closed.
        return_annotation = signature.return_annotation
    return signature.replace(parameters=parameters, return_annotation=return_annotation)


def _dagster_invocation(
    *,
    signature: inspect.Signature,
    public_signature: inspect.Signature,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> tuple[DagsterContext | None, tuple[object, ...], dict[str, object]]:
    """Separate injected Dagster context from canonical callable arguments."""

    try:
        public_bound = public_signature.bind(*args, **kwargs)
    except TypeError:
        return None, args, dict(kwargs)
    context = public_bound.arguments.get("context")
    if not _is_dagster_context(context):
        return None, args, dict(kwargs)

    values = dict(public_bound.arguments)
    values.pop("context")
    op_execution_context = getattr(context, "op_execution_context", None)
    op_config = getattr(op_execution_context, "op_config", {})
    if isinstance(op_config, Mapping):
        for parameter in signature.parameters.values():
            if (
                parameter.name not in values
                and parameter.name != "context"
                and parameter.default is not inspect.Parameter.empty
                and parameter.name in op_config
            ):
                values[parameter.name] = op_config[parameter.name]
    if "context" in signature.parameters:
        values["context"] = context

    positional: list[object] = []
    keyword: dict[str, object] = {}
    for parameter in signature.parameters.values():
        if parameter.name not in values:
            continue
        value = values[parameter.name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        elif parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            keyword[parameter.name] = value
    return context, tuple(positional), keyword


def _is_dagster_context(value: object) -> bool:
    return (
        value is not None
        and hasattr(value, "run")
        and callable(getattr(value, "get_step_execution_context", None))
        and callable(getattr(value, "add_asset_metadata", None))
    )


def _optional_dagster() -> Any | None:
    try:
        import dagster
    except ImportError:
        return None
    return dagster


def _oclp_resource(context: DagsterContext) -> OclpDagsterResource:
    resources = getattr(context, "resources", None)
    resource = getattr(resources, OCLP_DAGSTER_RESOURCE_KEY, None)
    if not isinstance(resource, OclpDagsterResource):
        raise RuntimeError(
            "dagster_adapter requires an OclpDagsterResource configured as "
            f"resources[{OCLP_DAGSTER_RESOURCE_KEY!r}] and declared through "
            f"required_resource_keys={{{OCLP_DAGSTER_RESOURCE_KEY!r}}}"
        )
    return resource


def _dagster_asset_workflow(
    context: DagsterContext,
    *,
    function: Callable[..., object],
    kind: DeclarationKind,
    run_name: DagsterRunName | None,
) -> Callable[[], None]:
    @run(name=_dagster_asset_run_name(context, function, kind, run_name))
    def workflow() -> None:
        return None

    return workflow


def _dagster_asset_run_name(
    context: DagsterContext,
    function: Callable[..., object],
    kind: DeclarationKind,
    configured_name: DagsterRunName | None,
) -> str:
    if configured_name is not None:
        resolved_name = (
            configured_name(context)
            if callable(configured_name)
            else configured_name
        )
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            raise ValueError(
                "dagster_adapter run_name must resolve to a non-empty string"
            )
        return resolved_name.strip()
    if kind == "artifact":
        return artifact_type(function).name
    if kind == "computation":
        return computation_template(function).name
    if kind == "artifact_set":
        return artifact_set_assembly_name(function)
    raise AssertionError(f"unsupported Dagster declaration kind {kind!r}")


def _dagster_computation_result(
    *,
    result: object,
    function: Callable[..., object],
    observed: OclpRun,
    context: DagsterContext,
    output_bindings: Mapping[str, str],
) -> object:
    """Bridge one named domain result to native Dagster output values.

    Shared outputs become exact OCLP Artifact handles. Native-only outputs stay
    raw domain values, while OCLP-only outputs have already been materialized
    but are intentionally omitted from the Dagster result.
    """

    template = computation_template(function)
    oclp_outputs = (
        observed.outputs_for(result) if template.output_artifacts else {}
    )
    dagster_outputs = _dagster_output_names(context)
    if not dagster_outputs:
        if not oclp_outputs:
            return result
        if len(oclp_outputs) == 1:
            return next(iter(oclp_outputs.values()))
        return tuple(oclp_outputs[port] for port in template.output_artifacts)

    if len(dagster_outputs) == 1:
        output_name = dagster_outputs[0]
        oclp_port = output_bindings.get(output_name)
        if oclp_port is not None:
            return oclp_outputs[oclp_port]
        return _dagster_named_output_value(
            result=result,
            output_name=output_name,
            allow_direct_value=True,
        )

    return tuple(
        (
            oclp_outputs[output_bindings[output_name]]
            if output_name in output_bindings
            else _dagster_named_output_value(
                result=result,
                output_name=output_name,
                allow_direct_value=False,
            )
        )
        for output_name in dagster_outputs
    )


def _dagster_named_output_value(
    *,
    result: object,
    output_name: str,
    allow_direct_value: bool,
) -> object:
    """Return one native-only value from an application-owned named result."""

    if isinstance(result, Mapping) and output_name in result:
        return result[output_name]
    try:
        return getattr(result, output_name)
    except AttributeError:
        if allow_direct_value:
            return result
    raise ValueError(
        "Dagster multi-asset output "
        f"{output_name!r} is not present on the named Computation result; "
        "return a mapping or object with same-named fields"
    )


def _context_value(context: DagsterContext, field: str) -> object | None:
    if field == "run_id":
        return str(context.run.run_id)
    if field == "asset_key":
        return _context_asset_keys(context)[0]
    if field == "step_key":
        return context.get_step_execution_context().step.key
    if field == "partition_key":
        return context.partition_key if context.has_partition_key else None
    if field == "retry_number":
        return int(context.retry_number)
    raise AssertionError(f"unsupported Dagster context field {field!r}")


def _context_asset_keys(context: DagsterContext) -> tuple[str, ...]:
    try:
        return (context.asset_key.to_user_string(),)
    except Exception:
        assets_def = getattr(context, "assets_def", None)
        keys_by_output_name = getattr(assets_def, "keys_by_output_name", None)
        if not isinstance(keys_by_output_name, Mapping) or not keys_by_output_name:
            raise
        return tuple(
            key.to_user_string() for _, key in sorted(keys_by_output_name.items())
        )


def dagster_run_context(context: DagsterContext) -> DagsterRunContext:
    """Derive portable scheduler facts and shared Dagster run identity."""

    dagster_run_id = str(_context_value(context, "run_id"))
    asset_keys = _context_asset_keys(context)
    partition_key = _context_value(context, "partition_key")
    step_key = str(_context_value(context, "step_key"))
    retry_number = int(_context_value(context, "retry_number"))
    try:
        run_id = UUID(dagster_run_id)
    except ValueError:
        run_id = uuid5(NAMESPACE_URL, f"oclp-dagster-run:{dagster_run_id}")
    return DagsterRunContext(
        run_id=run_id,
        profile=DagsterExecutionProfile(
            dagster_run_id=dagster_run_id,
            asset_key=asset_keys[0],
            asset_keys=asset_keys,
            step_key=step_key,
            partition_key=str(partition_key) if partition_key is not None else None,
            retry_number=retry_number,
        ),
    )


def _dagster_declaration_run_context(context: DagsterContext) -> DagsterRunContext:
    """Return the stable OCLP identity for one native asset attempt."""

    shared = dagster_run_context(context)
    run_id = uuid5(
        NAMESPACE_URL,
        "oclp-dagster-asset:"
        + ":".join(
            (
                shared.profile.dagster_run_id,
                ",".join(shared.profile.asset_keys),
                shared.profile.step_key,
                str(shared.profile.partition_key or ""),
                str(shared.profile.retry_number),
            )
        ),
    )
    return DagsterRunContext(run_id=run_id, profile=shared.profile)


def _validate_resource_configuration(
    *,
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
) -> None:
    if not callable(publisher):
        raise TypeError("Dagster resource publisher must be a context-specific factory")
    if not callable(source) and not isinstance(source, _SOURCE_TYPES):
        raise TypeError("Dagster resource source must be an OCLP implementation source")


def _resolve_source(
    source: ImplementationSource | SourceFactory,
    context: DagsterContext,
) -> ImplementationSource:
    resolved = source(context) if callable(source) else source
    if not isinstance(resolved, _SOURCE_TYPES):
        raise TypeError(
            "Dagster resource source factory must return an OCLP implementation source"
        )
    return resolved


def _resolve_application_profiles(
    declaration: ApplicationProfiles | None,
    context: DagsterContext,
) -> ProfileBindings | None:
    if declaration is None:
        return None
    profiles = declaration(context) if callable(declaration) else declaration
    if not isinstance(profiles, Mapping):
        raise TypeError("application profiles must resolve to a profile mapping")
    reserved = {RUN_PROFILE, DAGSTER_PROFILE}.intersection(profiles)
    if reserved:
        raise ValueError(
            "application profiles cannot supply SDK-owned profile keys: "
            + ", ".join(sorted(reserved))
        )
    resolved: ProfileBindings = {}
    for name, value in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("application profile names must be non-empty strings")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"application profile {name!r} must map field names to values"
            )
        resolved[name] = dict(value)
    return resolved or None


def _resolve_lifecycle(
    declaration: LifecycleDeclaration | None,
    *,
    context: DagsterContext,
) -> Lifecycle | None:
    if declaration is None:
        return None
    resolved = declaration(context) if callable(declaration) else declaration
    try:
        return coerce_lifecycle(resolved)
    except (TypeError, ValidationError) as error:
        raise ValueError(
            "dagster_adapter lifecycle must resolve to a Lifecycle, UUID, or UUID "
            "string"
        ) from error


def _execution_profiles(
    step: DagsterRunContext,
    application_profiles: ProfileBindings | None,
) -> ProfileBindings:
    profiles = dict(application_profiles or {})
    profiles.update(step.profiles)
    return profiles


def oclp_artifact_io_manager(
    *,
    catalog_path: Path | str,
    storage_root: Path | str,
) -> object:
    """Return a Dagster I/O resource that exchanges OCLP Artifact handles."""

    dagster = _require_dagster()
    resolved_catalog_path = Path(catalog_path)
    resolved_storage_root = Path(storage_root)

    class OclpArtifactIOManager(dagster.IOManager):
        def handle_output(self, context: Any, obj: object) -> None:
            if not isinstance(obj, ArtifactHandle):
                raise TypeError(
                    "oclp_artifact_io_manager can persist only ArtifactHandle "
                    f"outputs, got {type(obj).__name__}"
                )
            pointer_path = _artifact_pointer_path(
                resolved_storage_root,
                asset_key_path=context.asset_key.path,
                partition_key=(
                    str(context.asset_partition_key)
                    if context.has_asset_partitions
                    else None
                ),
            )
            pointer_path.parent.mkdir(parents=True, exist_ok=True)
            pointer_path.write_text(
                json.dumps({"artifact_id": obj.reference.id}, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        def load_input(
            self, context: Any
        ) -> ArtifactHandle | tuple[ArtifactHandle, ...]:
            partition_keys = (
                tuple(str(key) for key in context.asset_partition_keys)
                if context.has_asset_partitions
                else (None,)
            )
            handles = tuple(
                _load_artifact_pointer(
                    catalog_path=resolved_catalog_path,
                    pointer_path=_artifact_pointer_path(
                        resolved_storage_root,
                        asset_key_path=context.asset_key.path,
                        partition_key=partition_key,
                    ),
                )
                for partition_key in partition_keys
            )
            return handles[0] if len(handles) == 1 else handles

    return dagster.IOManagerDefinition.hardcoded_io_manager(OclpArtifactIOManager())


def load_artifact_handle(
    *,
    catalog_path: Path | str,
    artifact_id: str,
) -> ArtifactHandle:
    """Resolve one local OCLP Artifact handle from its durable record UUID."""

    from oclp.catalog.duckdb import DuckdbCatalog

    with DuckdbCatalog(Path(catalog_path)) as catalog:
        reference = RecordReference(id=artifact_id)
        artifact = catalog.resolve(reference)
        if not isinstance(artifact, Artifact):
            raise TypeError(f"record {artifact_id!r} is not an OCLP Artifact")
        locations = catalog.locations_for(reference)
    file_locations = tuple(
        location for location in locations if urlparse(location).scheme == "file"
    )
    if not file_locations:
        raise FileNotFoundError(
            f"Artifact {artifact_id!r} has no local file location in the catalog"
        )
    payload_path = Path(unquote(urlparse(file_locations[0]).path))
    if not payload_path.is_file():
        raise FileNotFoundError(
            f"Artifact {artifact_id!r} payload is unavailable at {payload_path}"
        )
    return ArtifactHandle(
        PublishedArtifact(
            artifact=artifact,
            path=payload_path,
            reference=reference,
        )
    )


def _artifact_pointer_path(
    storage_root: Path,
    *,
    asset_key_path: tuple[str, ...] | list[str],
    partition_key: str | None,
) -> Path:
    encoded_asset_key = tuple(
        _encode_pointer_component(part) for part in asset_key_path
    )
    if partition_key is None:
        return storage_root.joinpath(*encoded_asset_key, "handle.json")
    return storage_root.joinpath(
        *encoded_asset_key,
        _encode_pointer_component(partition_key),
        "handle.json",
    )


def _encode_pointer_component(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _load_artifact_pointer(*, catalog_path: Path, pointer_path: Path) -> ArtifactHandle:
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no OCLP Artifact pointer exists at {pointer_path}; materialize the "
            "upstream asset partition first"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid OCLP Artifact pointer at {pointer_path}") from error
    artifact_id = pointer.get("artifact_id") if isinstance(pointer, dict) else None
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(
            "OCLP Artifact pointer at "
            f"{pointer_path} must contain a non-empty artifact_id"
        )
    return load_artifact_handle(catalog_path=catalog_path, artifact_id=artifact_id)


def _require_dagster() -> Any:
    try:
        import dagster
    except ModuleNotFoundError as error:  # pragma: no cover - install dependent
        raise ModuleNotFoundError(
            "Dagster integration requires the optional dependency; install "
            "oclp[dagster]."
        ) from error
    return dagster

"""Optional Dagster projections for explicit OCLP boundaries.

``dg_workflow`` projects an application-owned workflow as one Dagster asset.
``dg_artifact`` and ``dg_computation`` project already-declared OCLP
acquisition and Computation boundaries as a granular Dagster asset graph.
``dagster_asset`` is retained as the original low-level observation adapter
for applications that need to compose their own Dagster decorators. None of
these APIs derive an OCLP Computation from a Dagster asset or create a
synthetic parent Execution.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeAlias, TypeVar
from urllib.parse import unquote, urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field

from oclp.artifacts import ArtifactHandle, artifact_type
from oclp.computations import (
    ManyArtifacts,
    computation_input_artifact_types,
    computation_template,
)
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
from oclp.profiles.run import RUN_PROFILE
from oclp.publishing import LocalArtifactPublisher, PublishedArtifact
from oclp.runtime import ArtifactSetHandle, OclpRun, observe_run, run_template

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
DagsterContext: TypeAlias = Any
PublisherFactory: TypeAlias = Callable[
    [DagsterContext], AbstractContextManager[LocalArtifactPublisher]
]
SourceFactory: TypeAlias = Callable[[DagsterContext], ImplementationSource]
ApplicationProfiles: TypeAlias = ProfileBindings | Callable[
    [DagsterContext], ProfileBindings
]

_CONTEXT_FIELDS = frozenset(
    {
        "run_id",
        "job_name",
        "asset_key",
        "step_key",
        "partition_key",
        "retry_number",
    }
)
_SOURCE_TYPES = (GitSource, ArtifactSource, ServiceSource, OpaqueSource)

DAGSTER_PROFILE = "dagster"
"""Execution profile key used by the Dagster projection decorators."""

DAGSTER_PROFILE_VERSION = "0.3.0-draft"
"""Version of the SDK-owned Dagster execution-profile binding."""


class DagsterExecutionProfile(OclpModel):
    """Durable scheduler facts attached to one projected OCLP Execution."""

    version: Literal["0.3.0-draft"] = DAGSTER_PROFILE_VERSION
    dagster_run_id: str = Field(min_length=1)
    job_name: str = Field(min_length=1)
    asset_key: str = Field(min_length=1)
    asset_keys: tuple[str, ...] = ()
    step_key: str = Field(min_length=1)
    partition_key: str | None = None
    retry_number: int = Field(ge=0)


@dataclass(frozen=True)
class DagsterRunContext:
    """Rehydratable OCLP facts for one Dagster asset execution.

    Dagster may execute different asset steps in distinct worker processes.  A
    process therefore constructs this small value from its own Dagster context
    and opens an independent :func:`observe_run`.  The shared UUID is the
    durable OCLP run identity; no Python ``ContextVar`` crosses workers.
    """

    run_id: UUID
    profile: DagsterExecutionProfile

    @property
    def profiles(self) -> ProfileBindings:
        """Return the SDK-owned profile binding for this asset attempt."""

        return {
            DAGSTER_PROFILE: self.profile.model_dump(mode="json"),
        }

    @property
    def run_name(self) -> str:
        """Return the readable OCLP title shared by one Dagster run.

        All projected assets in the same Dagster job share this label and UUID.
        Separate partition runs retain the job name but gain their exact
        partition key, making Explorer's OCLP Run nodes distinguishable.
        """

        if self.profile.partition_key is None:
            return self.profile.job_name
        return f"{self.profile.job_name} [{self.profile.partition_key}]"


@dataclass
class DagsterAdapter:
    """Attach selected OCLP references to one Dagster asset materialization.

    This is an outbound projection: the SDK still publishes the immutable OCLP
    records, while Dagster receives local navigation metadata after the
    observed workflow ends. A metadata failure follows the regular adapter
    strict/Diagnostic policy.
    """

    context: DagsterContext
    context_fields: tuple[str, ...]
    strict: bool = False

    def on_run_end(self, observed: OclpRun, error: BaseException | None) -> None:
        """Add the completed OCLP run identity to Dagster asset metadata."""

        metadata: dict[str, object] = {
            "oclp.run.id": str(observed.run_id),
            "oclp.run.name": observed.run_name or "",
            "oclp.record_root": str(observed.publisher.record_root),
            "oclp.status": "failed" if error is not None else "succeeded",
        }
        for field in self.context_fields:
            value = _context_value(self.context, field)
            if value is not None:
                metadata[f"oclp.dagster.{field}"] = value
        self.context.add_asset_metadata(metadata)


def dagster_asset(
    *,
    workflow: Callable[..., object],
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    context_fields: tuple[str, ...] = (
        "run_id",
        "asset_key",
        "partition_key",
        "retry_number",
    ),
    strict: bool = False,
    context_parameter: str = "context",
) -> Callable[[Callable[Parameters, Result]], Callable[Parameters, Result]]:
    """Observe one Dagster asset through an existing OCLP ``@run`` workflow.

    This compatibility-level adapter is applied directly inside
    :func:`dagster.asset`::

        @dg.asset
        @dagster_asset(workflow=train, publisher=publisher_for, source=source)
        def trained_model(context: dg.AssetExecutionContext):
            return train(...)

    ``workflow`` must already carry :func:`oclp.run`. The asset body calls that
    workflow normally; its existing ``@computation`` and Artifact declarations
    remain the only source of OCLP records. ``context_fields`` is an explicit
    allow-list of Dagster context values copied to Dagster metadata, never a
    reflection of arbitrary scheduler state.
    """

    _require_dagster()
    template = run_template(workflow)
    if not callable(publisher):
        raise TypeError("dagster_asset publisher must be a context-specific factory")
    if not isinstance(context_fields, tuple):
        raise TypeError("dagster_asset context_fields must be a tuple")
    unknown_fields = sorted(set(context_fields).difference(_CONTEXT_FIELDS))
    if unknown_fields:
        raise ValueError(
            "dagster_asset context_fields must select known values; unknown: "
            + ", ".join(unknown_fields)
        )
    if len(context_fields) != len(set(context_fields)):
        raise ValueError("dagster_asset context_fields must not contain duplicates")
    if not isinstance(strict, bool):
        raise TypeError("dagster_asset strict must be a boolean")

    def decorate(
        function: Callable[Parameters, Result],
    ) -> Callable[Parameters, Result]:
        signature = inspect.signature(function)
        if context_parameter not in signature.parameters:
            raise ValueError(
                "dagster_asset requires an asset function parameter named "
                f"{context_parameter!r}"
            )

        @wraps(function)
        def observed(*args: Parameters.args, **kwargs: Parameters.kwargs) -> Result:
            bound = signature.bind(*args, **kwargs)
            context = bound.arguments[context_parameter]
            resolved_source = source(context) if callable(source) else source
            if not isinstance(resolved_source, _SOURCE_TYPES):
                raise TypeError(
                    "dagster_asset source factory must return an OCLP "
                    "implementation source"
                )
            adapter = DagsterAdapter(
                context=context,
                context_fields=context_fields,
                strict=strict,
            )
            with publisher(context) as local_publisher:
                with observe_run(
                    workflow,
                    publisher=local_publisher,
                    source=resolved_source,
                    adapters=(*template.adapters, adapter),
                ):
                    return function(*args, **kwargs)

        # Dagster validates the first parameter's concrete context type while
        # this repository uses postponed annotations. Give Dagster a resolved
        # public signature without exposing ``functools.wraps``' unwrapped
        # callable, whose annotations may still be strings.
        annotations = inspect.get_annotations(function, eval_str=True)
        observed.__annotations__ = annotations
        observed.__signature__ = signature.replace(
            parameters=[
                parameter.replace(
                    annotation=annotations.get(name, parameter.annotation)
                )
                for name, parameter in signature.parameters.items()
            ],
            return_annotation=annotations.get("return", signature.return_annotation),
        )
        del observed.__wrapped__
        return observed

    return decorate


def dg_workflow(
    *,
    workflow: Callable[..., object],
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    asset_key: object,
    group_name: str | None = None,
    description: str | None = None,
    retry_policy: object | None = None,
    partitions_def: object | None = None,
    io_manager_key: str | None = None,
    config_schema: object | None = None,
    context_fields: tuple[str, ...] = (
        "run_id",
        "asset_key",
        "partition_key",
        "retry_number",
    ),
    strict: bool = False,
    context_parameter: str = "context",
) -> Callable[[Callable[Parameters, Result]], object]:
    """Project an application-owned ``@run`` workflow as one Dagster asset.

    Unlike :func:`dg_artifact` and :func:`dg_computation`, this is a workflow
    observation boundary: its decorated body remains responsible for calling
    the selected ``@run`` workflow with application-specific arguments.  The
    decorator creates the Dagster asset directly, so users do not stack a
    second ``@dagster.asset`` decorator around it.

    :func:`dagster_asset` remains the lower-level compatibility adapter for
    applications that must compose their own Dagster decorator stack.
    """

    dagster = _require_dagster()
    observe = dagster_asset(
        workflow=workflow,
        publisher=publisher,
        source=source,
        context_fields=context_fields,
        strict=strict,
        context_parameter=context_parameter,
    )

    def decorate(function: Callable[Parameters, Result]) -> object:
        return dagster.asset(
            **_asset_options(
                asset_key=asset_key,
                group_name=group_name,
                description=description,
                retry_policy=retry_policy,
                partitions_def=partitions_def,
                io_manager_key=io_manager_key,
                config_schema=config_schema,
            )
        )(observe(function))

    return decorate


def dg_artifact(
    *,
    workflow: Callable[..., object],
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    asset_key: object,
    group_name: str | None = None,
    description: str | None = None,
    retry_policy: object | None = None,
    partitions_def: object | None = None,
    io_manager_key: str | None = None,
    config_schema: object | None = None,
    deps: object | None = None,
    application_profiles: ApplicationProfiles | None = None,
    context_parameter: str = "context",
) -> Callable[[Callable[Parameters, ArtifactHandle]], object]:
    """Project one existing OCLP Artifact acquisition as a Dagster asset.

    Apply this outside an OCLP Artifact decorator.  The inner decorator remains
    responsible for the acquired Artifact record; this projection only opens a
    per-step OCLP run context, returns the exact :class:`ArtifactHandle` to
    Dagster, and records cross-navigation metadata.  Acquisition does not
    manufacture an OCLP Execution. A no-argument forwarding callable may also
    return an already acquired ``ArtifactHandle`` from an existing declaration.

    The first projection keeps run configuration explicit: acquisition
    callables may have only defaulted parameters. A forwarding callable may
    instead receive the Dagster context as ``context_parameter`` and use it to
    supply explicit application arguments to an existing acquisition. Dagster
    config mapping is a separate concern from durable Artifact acquisition and
    is intentionally not inferred here.
    """

    dagster = _require_dagster()
    run_template(workflow)
    _validate_projection_common(publisher=publisher, source=source)

    def decorate(function: Callable[Parameters, ArtifactHandle]) -> object:
        try:
            declared = artifact_type(function)
        except ValueError:
            declared = None
        signature = inspect.signature(function)
        _validate_defaulted_parameters(
            signature,
            decorator="dg_artifact",
            ignored_parameters=frozenset((context_parameter,)),
        )

        @wraps(function)
        def projected(context: DagsterContext) -> ArtifactHandle:
            step = dagster_run_context(context)
            record_profiles = _resolve_application_profiles(
                application_profiles,
                context,
            )
            resolved_source = _resolve_source(source, context)
            with publisher(context) as local_publisher:
                with observe_run(
                    workflow,
                    publisher=local_publisher,
                    run_id=step.run_id,
                    run_name=step.run_name,
                    source=resolved_source,
                    profiles=_execution_profiles(step, record_profiles),
                    record_profiles=record_profiles,
                ) as observed:
                    if context_parameter in signature.parameters:
                        handle = function(context)
                    else:
                        handle = function()
                    if not isinstance(handle, ArtifactHandle):
                        raise TypeError(
                            "dg_artifact requires an OCLP Artifact-decorated "
                            "callable or a forwarding callable that returns an "
                            "ArtifactHandle"
                        )
            _add_output_metadata(
                context,
                _metadata_for_artifact(
                    observed=observed,
                    step=step,
                    artifact=handle,
                    artifact_name=declared.name if declared is not None else None,
                ),
            )
            return handle

        _set_asset_signature(projected, ())
        return dagster.asset(
            **_asset_options(
                asset_key=asset_key,
                group_name=group_name,
                description=(
                    description
                    or (declared.description if declared is not None else None)
                ),
                retry_policy=retry_policy,
                partitions_def=partitions_def,
                io_manager_key=io_manager_key,
                config_schema=config_schema,
                deps=deps,
            )
        )(projected)

    return decorate


def dg_computation(
    *,
    workflow: Callable[..., object],
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    asset_key: object | None = None,
    inputs: Mapping[str, object] | None = None,
    outputs: Mapping[str, object] | None = None,
    group_name: str | None = None,
    description: str | None = None,
    retry_policy: object | None = None,
    partitions_def: object | None = None,
    io_manager_key: str | None = None,
    config_schema: object | None = None,
    deps: object | None = None,
    application_profiles: ApplicationProfiles | None = None,
    context_parameter: str | None = None,
    target: Callable[..., object] | None = None,
) -> Callable[[Callable[Parameters, Result]], object]:
    """Project one existing OCLP Computation as one or more Dagster assets.

    Apply this outside :func:`oclp.computation`.  The inner Computation
    declaration remains canonical: it specifies durable input and output
    representations, while ``inputs`` maps those named input ports to Dagster
    assets.  The generated asset receives exact Artifact handles, so the
    ordinary OCLP runtime records each Dagster dependency in the resulting
    Execution's ``inputs`` without copying an in-memory value or payload. For
    a single output, pass ``asset_key``. For multiple outputs, pass an
    ``outputs`` mapping from OCLP output port to ``dagster.AssetOut``; the
    resulting ``multi_asset`` is non-subsettable, so its real OCLP Execution
    always materializes every declared output atomically.

    A ``many(...)`` OCLP input maps to an explicitly ordered tuple of
    ``AssetIn`` declarations. ``target`` supports a declarative named proxy
    for an existing decorated callable imported from another module: the
    decorated proxy should call ``target`` in its body, while this decorator
    reads the canonical OCLP declaration from ``target``.
    """

    dagster = _require_dagster()
    run_template(workflow)
    _validate_projection_common(publisher=publisher, source=source)
    if inputs is not None and not isinstance(inputs, Mapping):
        raise TypeError("dg_computation inputs must be a mapping of port names")
    if outputs is not None and not isinstance(outputs, Mapping):
        raise TypeError("dg_computation outputs must be a mapping of port names")
    if target is not None and not callable(target):
        raise TypeError("dg_computation target must be an OCLP-decorated callable")
    asset_inputs = dict(inputs or {})
    asset_outputs = dict(outputs or {})

    def decorate(function: Callable[Parameters, Result]) -> object:
        declared_function = target or function
        computation = computation_template(declared_function)
        declared_inputs = computation_input_artifact_types(declared_function)
        _validate_projected_computation(
            function=function,
            input_names=frozenset(declared_inputs),
            asset_inputs=asset_inputs,
            context_parameter=context_parameter,
        )
        # A decorated Computation may normally assemble an ArtifactSet at the
        # end of an application-owned run. Projected steps deliberately defer
        # that assembly (see ``finalize_decorated_artifact_sets=False`` below)
        # so one explicit ``dg_artifact_set`` can collect artifacts that were
        # materialized in different Dagster workers and runs.
        output_ports = tuple(computation.output_artifacts)
        if not output_ports:
            raise ValueError(
                "dg_computation requires at least one declared OCLP Artifact output"
            )
        if asset_outputs:
            if asset_key is not None:
                raise ValueError(
                    "dg_computation accepts either asset_key or outputs, not both"
                )
            if set(asset_outputs) != set(output_ports):
                missing = sorted(set(output_ports).difference(asset_outputs))
                unknown = sorted(set(asset_outputs).difference(output_ports))
                details: list[str] = []
                if missing:
                    details.append("missing: " + ", ".join(missing))
                if unknown:
                    details.append("unknown: " + ", ".join(unknown))
                raise ValueError(
                    "dg_computation outputs must map every declared OCLP Artifact "
                    "output exactly once (" + "; ".join(details) + ")"
                )
        elif asset_key is None:
            raise ValueError(
                "dg_computation requires asset_key for one output or outputs for "
                "a multi-output Computation"
            )
        elif len(output_ports) != 1:
            raise ValueError(
                "dg_computation requires outputs for a multi-output Computation"
            )

        bindings = _projected_input_bindings(
            declared_inputs=declared_inputs,
            asset_inputs=asset_inputs,
        )
        signature = inspect.signature(function)

        def invoke(
            *args: Any, **kwargs: Any
        ) -> tuple[
            DagsterContext,
            OclpRun,
            DagsterRunContext,
            Mapping[str, ArtifactHandle],
            object,
            object,
        ]:
            asset_signature = _asset_signature(signature, bindings.dagster_inputs)
            bound = asset_signature.bind(*args, **kwargs)
            context = bound.arguments.pop("context")
            step = dagster_run_context(context)
            record_profiles = _resolve_application_profiles(
                application_profiles,
                context,
            )
            resolved_source = _resolve_source(source, context)
            invocation = _invocation_arguments(bound.arguments, bindings)
            with publisher(context) as local_publisher:
                with observe_run(
                    workflow,
                    publisher=local_publisher,
                    run_id=step.run_id,
                    run_name=step.run_name,
                    source=resolved_source,
                    profiles=_execution_profiles(step, record_profiles),
                    record_profiles=record_profiles,
                    finalize_decorated_artifact_sets=False,
                ) as observed:
                    if context_parameter is not None:
                        result = function(**invocation, **{context_parameter: context})
                    else:
                        result = function(**invocation)
                    produced_outputs = observed.outputs_for(result)
                    execution = observed.execution_for(result)
                    computation_ref = observed.computation_for(result)
            return (
                context,
                observed,
                step,
                produced_outputs,
                execution,
                computation_ref,
            )

        if asset_outputs:

            @wraps(function)
            def projected_multi(*args: Any, **kwargs: Any) -> object:
                (
                    context,
                    observed,
                    step,
                    produced_outputs,
                    execution,
                    computation_ref,
                ) = invoke(*args, **kwargs)
                del context
                for output_port in output_ports:
                    artifact = produced_outputs[output_port]
                    yield dagster.Output(
                        artifact,
                        output_name=output_port,
                        metadata=_metadata_for_computation(
                            observed=observed,
                            step=step,
                            artifact=artifact,
                            output_port=output_port,
                            execution_id=execution.id,
                            computation_id=computation_ref.id,
                        ),
                    )

            _set_asset_signature(
                projected_multi,
                tuple(bindings.dagster_inputs),
                return_annotation=Any,
            )
            return dagster.multi_asset(
                **_multi_asset_options(
                    outputs=asset_outputs,
                    inputs=bindings.dagster_inputs,
                    group_name=group_name,
                    description=description or computation.description,
                    retry_policy=retry_policy,
                    partitions_def=partitions_def,
                    config_schema=config_schema,
                    deps=deps,
                )
            )(projected_multi)

        output_port = output_ports[0]

        @wraps(function)
        def projected(*args: Any, **kwargs: Any) -> ArtifactHandle:
            (
                context,
                observed,
                step,
                produced_outputs,
                execution,
                computation_ref,
            ) = invoke(*args, **kwargs)
            output = produced_outputs[output_port]
            _add_output_metadata(
                context,
                _metadata_for_computation(
                    observed=observed,
                    step=step,
                    artifact=output,
                    output_port=output_port,
                    execution_id=execution.id,
                    computation_id=computation_ref.id,
                ),
            )
            return output

        _set_asset_signature(projected, tuple(bindings.dagster_inputs))
        return dagster.asset(
            **_asset_options(
                asset_key=asset_key,
                inputs=bindings.dagster_inputs,
                group_name=group_name,
                description=description or computation.description,
                retry_policy=retry_policy,
                partitions_def=partitions_def,
                io_manager_key=io_manager_key,
                config_schema=config_schema,
                deps=deps,
            )
        )(projected)

    return decorate


def dg_artifact_set(
    *,
    workflow: Callable[..., object],
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
    asset_key: object,
    name: str,
    inputs: Mapping[str, object],
    members: Mapping[str, tuple[str, str | None]],
    group_name: str | None = None,
    description: str | None = None,
    retry_policy: object | None = None,
    partitions_def: object | None = None,
    io_manager_key: str | None = None,
    config_schema: object | None = None,
    deps: object | None = None,
    application_profiles: ApplicationProfiles | None = None,
) -> Callable[[Callable[Parameters, object]], object]:
    """Project an explicit OCLP ArtifactSet assembly as a Dagster asset.

    A run-wide set cannot be assembled opportunistically in individual worker
    processes: each projected Computation has its own short-lived OCLP context.
    This decorator makes collection assembly a visible terminal Dagster node.
    ``inputs`` maps local names to ``AssetIn`` declarations; ``members`` maps
    ArtifactSet member names to ``(input_name, optional_role)`` pairs. The
    decorated marker function supplies the asset's Python name and docstring;
    ArtifactSet assembly itself has no user-code body.
    """

    dagster = _require_dagster()
    run_template(workflow)
    _validate_projection_common(publisher=publisher, source=source)
    if not isinstance(inputs, Mapping) or not inputs:
        raise TypeError("dg_artifact_set inputs must be a non-empty mapping")
    if not isinstance(members, Mapping) or not members:
        raise TypeError("dg_artifact_set members must be a non-empty mapping")
    asset_inputs = dict(inputs)
    artifact_members = dict(members)
    _validate_artifact_set_members(
        asset_inputs=asset_inputs,
        artifact_members=artifact_members,
    )

    def decorate(function: Callable[Parameters, object]) -> object:
        signature = inspect.signature(function)

        @wraps(function)
        def projected(*args: Any, **kwargs: Any) -> ArtifactSetHandle:
            asset_signature = _asset_signature(signature, asset_inputs)
            bound = asset_signature.bind(*args, **kwargs)
            context = bound.arguments.pop("context")
            step = dagster_run_context(context)
            record_profiles = _resolve_application_profiles(
                application_profiles,
                context,
            )
            resolved_source = _resolve_source(source, context)
            input_handles = dict(bound.arguments)
            invalid = sorted(
                input_name
                for input_name, value in input_handles.items()
                if not isinstance(value, ArtifactHandle)
            )
            if invalid:
                raise TypeError(
                    "dg_artifact_set inputs must receive ArtifactHandle values; "
                    "invalid: " + ", ".join(invalid)
                )
            with publisher(context) as local_publisher:
                with observe_run(
                    workflow,
                    publisher=local_publisher,
                    run_id=step.run_id,
                    run_name=step.run_name,
                    source=resolved_source,
                    profiles=_execution_profiles(step, record_profiles),
                    record_profiles=record_profiles,
                    finalize_decorated_artifact_sets=False,
                ) as observed:
                    artifact_set = observed.publish_artifact_set(
                        name=name,
                        members={
                            member_name: (
                                input_handles[input_name],
                                role,
                            )
                            for member_name, (input_name, role) in (
                                artifact_members.items()
                            )
                        },
                        materialize_manifest=True,
                        manifest_name=name,
                    )
            _add_output_metadata(
                context,
                _metadata_for_artifact_set(
                    observed=observed,
                    step=step,
                    artifact_set=artifact_set,
                ),
            )
            return artifact_set

        _set_asset_signature(
            projected,
            tuple(asset_inputs),
            return_annotation=ArtifactSetHandle,
        )
        return dagster.asset(
            **_asset_options(
                asset_key=asset_key,
                inputs=asset_inputs,
                group_name=group_name,
                description=description or inspect.getdoc(function),
                retry_policy=retry_policy,
                partitions_def=partitions_def,
                io_manager_key=io_manager_key,
                config_schema=config_schema,
                deps=deps,
            )
        )(projected)

    return decorate


def _context_value(context: DagsterContext, field: str) -> object | None:
    """Return one intentionally supported Dagster context value."""

    if field == "run_id":
        return str(context.run.run_id)
    if field == "job_name":
        job_name = getattr(context, "job_name", None)
        if not isinstance(job_name, str) or not job_name:
            job_name = getattr(context.run, "job_name", None)
        if not isinstance(job_name, str) or not job_name:
            raise ValueError("Dagster context must expose a non-empty job_name")
        return job_name
    if field == "asset_key":
        return _context_asset_keys(context)[0]
    if field == "step_key":
        return context.get_step_execution_context().step.key
    if field == "partition_key":
        if not context.has_partition_key:
            return None
        return context.partition_key
    if field == "retry_number":
        return int(context.retry_number)
    raise AssertionError(f"unsupported Dagster context field {field!r}")


def _context_asset_keys(context: DagsterContext) -> tuple[str, ...]:
    """Return one key for an asset or every key for an atomic multi-asset."""

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
    """Derive portable OCLP run and step facts from one Dagster context.

    Dagster normally uses a UUID run ID directly.  The UUIDv5 fallback makes
    the mapping stable for compatible schedulers that expose another non-empty
    run identifier, while retaining OCLP's UUID run-profile requirement.
    """

    dagster_run_id = str(_context_value(context, "run_id"))
    try:
        run_id = UUID(dagster_run_id)
    except ValueError:
        run_id = uuid5(NAMESPACE_URL, f"oclp-dagster-run:{dagster_run_id}")
    asset_keys = _context_asset_keys(context)
    partition_key = _context_value(context, "partition_key")
    return DagsterRunContext(
        run_id=run_id,
        profile=DagsterExecutionProfile(
            dagster_run_id=dagster_run_id,
            job_name=str(_context_value(context, "job_name")),
            asset_key=asset_keys[0],
            asset_keys=asset_keys,
            step_key=str(_context_value(context, "step_key")),
            partition_key=str(partition_key) if partition_key is not None else None,
            retry_number=int(_context_value(context, "retry_number")),
        ),
    )


def _validate_projection_common(
    *,
    publisher: PublisherFactory,
    source: ImplementationSource | SourceFactory,
) -> None:
    """Validate shared declaration-time projection configuration."""

    if not callable(publisher):
        raise TypeError(
            "Dagster projection publisher must be a context-specific factory"
        )
    if not callable(source) and not isinstance(source, _SOURCE_TYPES):
        raise TypeError(
            "Dagster projection source must be an OCLP implementation source"
        )


def _resolve_source(
    source: ImplementationSource | SourceFactory,
    context: DagsterContext,
) -> ImplementationSource:
    """Resolve and validate the source basis for one asset attempt."""

    resolved = source(context) if callable(source) else source
    if not isinstance(resolved, _SOURCE_TYPES):
        raise TypeError(
            "Dagster projection source factory must return an OCLP "
            "implementation source"
        )
    return resolved


def _resolve_application_profiles(
    declaration: ApplicationProfiles | None,
    context: DagsterContext,
) -> ProfileBindings | None:
    """Resolve app-owned durable record profiles for one asset attempt.

    Scheduler/run profile keys are SDK-owned.  An application may attach its
    own profile either statically or from the Dagster context, which is useful
    for a project-owned release or business-cycle identity.
    """

    if declaration is None:
        return None
    profiles = declaration(context) if callable(declaration) else declaration
    if not isinstance(profiles, Mapping):
        raise TypeError("application_profiles must resolve to a profile mapping")
    reserved = {RUN_PROFILE, DAGSTER_PROFILE}.intersection(profiles)
    if reserved:
        raise ValueError(
            "application_profiles cannot supply SDK-owned profile keys: "
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


def _execution_profiles(
    step: DagsterRunContext,
    application_profiles: ProfileBindings | None,
) -> ProfileBindings:
    """Combine app-owned facts with the SDK-owned Dagster execution profile."""

    profiles = dict(application_profiles or {})
    profiles.update(step.profiles)
    return profiles


def _validate_defaulted_parameters(
    signature: inspect.Signature,
    *,
    decorator: str,
    ignored_parameters: frozenset[str] = frozenset(),
) -> None:
    """Reject required parameters until Dagster config has an explicit mapping."""

    required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in ignored_parameters
        and parameter.default is inspect.Parameter.empty
    ]
    if required:
        raise ValueError(
            f"{decorator} currently supports only defaulted acquisition "
            "parameters; required: " + ", ".join(required)
        )


def _validate_projected_computation(
    *,
    function: Callable[..., object],
    input_names: frozenset[str],
    asset_inputs: Mapping[str, object],
    context_parameter: str | None,
) -> None:
    """Keep Dagster edges explicit and OCLP parameters unambiguous."""

    if set(asset_inputs) != input_names:
        missing = sorted(input_names.difference(asset_inputs))
        unknown = sorted(set(asset_inputs).difference(input_names))
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError(
            "dg_computation inputs must map every declared OCLP Artifact "
            "input exactly once (" + "; ".join(details) + ")"
        )
    signature = inspect.signature(function)
    if context_parameter is not None and context_parameter not in signature.parameters:
        raise ValueError(
            "dg_computation context_parameter must name a proxy parameter; "
            f"missing: {context_parameter}"
        )
    undeclared_required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in input_names
        and parameter.name != context_parameter
        and parameter.default is inspect.Parameter.empty
    ]
    if undeclared_required:
        raise ValueError(
            "dg_computation requires non-Artifact parameters to have defaults "
            "until Dagster config mapping is explicit; required: "
            + ", ".join(undeclared_required)
        )
    positional_only = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name in input_names
        and parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        raise ValueError(
            "dg_computation does not support positional-only Artifact inputs: "
            + ", ".join(positional_only)
        )


@dataclass(frozen=True)
class _ProjectedInputBindings:
    """Dagster argument names and their OCLP invocation-port grouping."""

    dagster_inputs: Mapping[str, object]
    invocation_inputs: Mapping[str, tuple[str, ...]]
    many_inputs: frozenset[str]


def _projected_input_bindings(
    *,
    declared_inputs: Mapping[str, object],
    asset_inputs: Mapping[str, object],
) -> _ProjectedInputBindings:
    """Expand explicit ``many(...)`` edges into ordered Dagster inputs."""

    dagster_inputs: dict[str, object] = {}
    invocation_inputs: dict[str, tuple[str, ...]] = {}
    many_inputs: set[str] = set()
    for port, declaration in declared_inputs.items():
        declared_asset_input = asset_inputs[port]
        is_collection = isinstance(declared_asset_input, (tuple, list)) and not hasattr(
            declared_asset_input,
            "key",
        )
        if isinstance(declaration, ManyArtifacts):
            if not is_collection:
                # A single partition-mapped AssetIn represents a dynamically
                # sized collection. Its I/O manager returns the handles for the
                # mapped partitions in partition-key order.
                dagster_inputs[port] = declared_asset_input
                invocation_inputs[port] = (port,)
                many_inputs.add(port)
                continue
            if not declared_asset_input:
                raise ValueError(
                    f"dg_computation input {port!r} declares many(...), so it "
                    "requires at least one Dagster AssetIn"
                )
            argument_names: list[str] = []
            for index, asset_input in enumerate(declared_asset_input):
                argument_name = f"{port}__{index}"
                dagster_inputs[argument_name] = asset_input
                argument_names.append(argument_name)
            invocation_inputs[port] = tuple(argument_names)
            many_inputs.add(port)
            continue
        if is_collection:
            raise TypeError(
                f"dg_computation input {port!r} accepts one Artifact, so its "
                "Dagster input must be one AssetIn"
            )
        dagster_inputs[port] = declared_asset_input
        invocation_inputs[port] = (port,)
    return _ProjectedInputBindings(
        dagster_inputs=dagster_inputs,
        invocation_inputs=invocation_inputs,
        many_inputs=frozenset(many_inputs),
    )


def _invocation_arguments(
    bound_arguments: Mapping[str, object],
    bindings: _ProjectedInputBindings,
) -> dict[str, object]:
    """Restore one exact Artifact handle or ordered tuple per OCLP port."""

    invocation: dict[str, object] = {}
    for port, argument_names in bindings.invocation_inputs.items():
        values = tuple(bound_arguments[name] for name in argument_names)
        if port not in bindings.many_inputs:
            invocation[port] = values[0]
        elif len(values) == 1 and isinstance(values[0], (tuple, list)):
            invocation[port] = tuple(values[0])
        else:
            invocation[port] = values
    return invocation


def _validate_artifact_set_members(
    *,
    asset_inputs: Mapping[str, object],
    artifact_members: Mapping[str, tuple[str, str | None]],
) -> None:
    """Require every named set member to select one declared asset input."""

    referenced_inputs: set[str] = set()
    for member_name, declaration in artifact_members.items():
        if not isinstance(member_name, str) or not member_name:
            raise ValueError("dg_artifact_set member names must be non-empty strings")
        if not isinstance(declaration, tuple) or len(declaration) != 2:
            raise TypeError(
                "dg_artifact_set members must map names to "
                "(input_name, optional_role) pairs"
            )
        input_name, role = declaration
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("dg_artifact_set member input names must be non-empty")
        if input_name not in asset_inputs:
            raise ValueError(
                f"dg_artifact_set member {member_name!r} names unknown input "
                f"{input_name!r}"
            )
        if role is not None and (not isinstance(role, str) or not role):
            raise ValueError("dg_artifact_set member roles must be non-empty strings")
        referenced_inputs.add(input_name)
    unused = sorted(set(asset_inputs).difference(referenced_inputs))
    if unused:
        raise ValueError(
            "dg_artifact_set inputs must each become one member; unused: "
            + ", ".join(unused)
        )


def _asset_signature(
    signature: inspect.Signature,
    input_names: Mapping[str, object] | tuple[str, ...] | frozenset[str],
    *,
    return_annotation: object = ArtifactHandle,
) -> inspect.Signature:
    """Give Dagster only the context and durable Artifact input parameters.

    OCLP converts Artifact handles to domain values *inside* the projected
    call.  Dagster must therefore see ``Any`` at every edge rather than the
    inner computation's domain-value annotations.
    """

    parameters = [
        inspect.Parameter(
            "context",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=inspect.Parameter.empty,
        )
    ]
    for input_name in input_names:
        parameter = signature.parameters.get(input_name)
        if parameter is None:
            parameter = inspect.Parameter(
                input_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        parameters.append(
            parameter.replace(annotation=Any, default=inspect.Parameter.empty)
        )
    return inspect.Signature(parameters=parameters, return_annotation=return_annotation)


def _set_asset_signature(
    function: Callable[..., object],
    input_names: tuple[str, ...],
    *,
    return_annotation: object = ArtifactHandle,
) -> None:
    """Set Dagster-visible annotations without exposing the wrapped callable."""

    original = inspect.signature(function)
    if input_names:
        original = inspect.signature(getattr(function, "__wrapped__", function))
    signature = _asset_signature(
        original,
        input_names,
        return_annotation=return_annotation,
    )
    function.__annotations__ = {
        parameter.name: parameter.annotation
        for parameter in signature.parameters.values()
        if parameter.annotation is not inspect.Parameter.empty
    }
    function.__annotations__["return"] = return_annotation
    function.__signature__ = signature
    try:
        del function.__wrapped__
    except AttributeError:  # pragma: no cover - plain callable edge case.
        pass


def _metadata_for_artifact(
    *,
    observed: OclpRun,
    step: DagsterRunContext,
    artifact: ArtifactHandle,
    artifact_name: str | None,
) -> dict[str, object]:
    """Return materialization metadata for an acquired Artifact asset."""

    metadata = _metadata_for_step(observed=observed, step=step)
    metadata["oclp.artifact.id"] = artifact.reference.id
    if artifact_name is not None:
        metadata["oclp.artifact.name"] = artifact_name
    return metadata


def _metadata_for_computation(
    *,
    observed: OclpRun,
    step: DagsterRunContext,
    artifact: ArtifactHandle,
    output_port: str,
    execution_id: str,
    computation_id: str,
) -> dict[str, object]:
    """Return materialization metadata for a real projected Execution."""

    metadata = _metadata_for_step(observed=observed, step=step)
    metadata.update(
        {
            "oclp.computation.id": computation_id,
            "oclp.execution.id": execution_id,
            f"oclp.output.{output_port}.id": artifact.reference.id,
        }
    )
    return metadata


def _metadata_for_artifact_set(
    *,
    observed: OclpRun,
    step: DagsterRunContext,
    artifact_set: ArtifactSetHandle,
) -> dict[str, object]:
    """Return materialization metadata for an explicit collection boundary."""

    metadata = _metadata_for_step(observed=observed, step=step)
    metadata["oclp.artifact_set.id"] = artifact_set.reference.id
    if artifact_set.manifest is not None:
        metadata["oclp.artifact_set.manifest.id"] = artifact_set.manifest.reference.id
    return metadata


def _metadata_for_step(
    *,
    observed: OclpRun,
    step: DagsterRunContext,
) -> dict[str, object]:
    """Return navigation metadata shared by every projected asset."""

    metadata: dict[str, object] = {
        "oclp.run.id": str(step.run_id),
        "oclp.run.name": observed.run_name or "",
        "oclp.record_root": str(observed.publisher.record_root),
        "oclp.status": "succeeded",
    }
    for field in _CONTEXT_FIELDS:
        value = _context_value_from_profile(step.profile, field)
        if value is not None:
            metadata[f"oclp.dagster.{field}"] = value
    if len(step.profile.asset_keys) > 1:
        metadata["oclp.dagster.asset_keys"] = list(step.profile.asset_keys)
    return metadata


def _context_value_from_profile(
    profile: DagsterExecutionProfile,
    field: str,
) -> object | None:
    """Read the metadata allow-list from a durable profile rather than context."""

    if field == "run_id":
        return profile.dagster_run_id
    if field == "job_name":
        return profile.job_name
    if field == "asset_key":
        return profile.asset_key
    if field == "step_key":
        return profile.step_key
    if field == "partition_key":
        return profile.partition_key
    if field == "retry_number":
        return profile.retry_number
    raise AssertionError(f"unsupported Dagster context field {field!r}")


def _add_output_metadata(
    context: DagsterContext,
    metadata: Mapping[str, object],
) -> None:
    """Attach immutable OCLP navigation facts to the current materialization."""

    context.add_output_metadata(dict(metadata))


def oclp_artifact_io_manager(
    *,
    catalog_path: Path | str,
    storage_root: Path | str,
) -> object:
    """Return a Dagster I/O resource that hands off OCLP Artifact identities.

    Dagster workers do not share Python objects. This resource persists only a
    small pointer containing an Artifact record UUID; a downstream worker
    resolves the immutable record and local payload location from OCLP's
    DuckDB catalog. A partition-mapped input returns a tuple of handles in
    Dagster's partition-key order, which directly serves a ``many(...)`` OCLP
    computation input.

    ``storage_root`` is Dagster-owned state rather than an Artifact payload
    directory. It may be deleted and rebuilt from materializations; the OCLP
    catalog and immutable payload locations remain authoritative.
    """

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
            if len(handles) == 1:
                return handles[0]
            return handles

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
    """Return a filesystem-safe address for one materialized asset partition."""

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
    """Keep dynamic partition keys confined to one filesystem path segment."""

    from urllib.parse import quote

    return quote(value, safe="")


def _load_artifact_pointer(*, catalog_path: Path, pointer_path: Path) -> ArtifactHandle:
    """Read one I/O pointer and resolve the referenced immutable Artifact."""

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


def _asset_options(
    *,
    asset_key: object,
    inputs: Mapping[str, object] | None = None,
    group_name: str | None,
    description: str | None,
    retry_policy: object | None,
    partitions_def: object | None = None,
    io_manager_key: str | None = None,
    config_schema: object | None = None,
    deps: object | None = None,
) -> dict[str, object]:
    """Build Dagster decorator options without passing unsupported ``None``s."""

    options: dict[str, object] = {"key": asset_key}
    if inputs is not None:
        options["ins"] = dict(inputs)
    if group_name is not None:
        options["group_name"] = group_name
    if description is not None:
        options["description"] = description
    if retry_policy is not None:
        options["retry_policy"] = retry_policy
    if partitions_def is not None:
        options["partitions_def"] = partitions_def
    if io_manager_key is not None:
        options["io_manager_key"] = io_manager_key
    if config_schema is not None:
        options["config_schema"] = config_schema
    if deps is not None:
        options["deps"] = deps
    return options


def _multi_asset_options(
    *,
    outputs: Mapping[str, object],
    inputs: Mapping[str, object],
    group_name: str | None,
    description: str | None,
    retry_policy: object | None,
    partitions_def: object | None = None,
    config_schema: object | None = None,
    deps: object | None = None,
) -> dict[str, object]:
    """Build ``multi_asset`` options with explicit OCLP-to-Dagster ports."""

    options: dict[str, object] = {
        "outs": dict(outputs),
        "ins": dict(inputs),
    }
    if group_name is not None:
        options["group_name"] = group_name
    if description is not None:
        options["description"] = description
    if retry_policy is not None:
        options["retry_policy"] = retry_policy
    if partitions_def is not None:
        options["partitions_def"] = partitions_def
    if config_schema is not None:
        options["config_schema"] = config_schema
    if deps is not None:
        options["deps"] = deps
    return options


def _require_dagster() -> Any:
    """Return Dagster or raise a focused optional-dependency error."""

    try:
        import dagster
    except ModuleNotFoundError as error:  # pragma: no cover - install dependent
        raise ModuleNotFoundError(
            "Dagster integration requires the optional dependency; "
            "install oclp[dagster]."
        ) from error
    return dagster

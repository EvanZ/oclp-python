"""Optional Dagster integration for observing explicit OCLP workflow boundaries.

The adapter never turns a Dagster asset into an OCLP Computation. Applications
keep their existing OCLP-decorated callables and invoke them inside an asset;
this module establishes the surrounding ``@run`` observation and adds selected
OCLP references to the Dagster materialization metadata.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeAlias, TypeVar

from oclp.models import (
    ArtifactSource,
    GitSource,
    ImplementationSource,
    OpaqueSource,
    ServiceSource,
)
from oclp.publishing import LocalArtifactPublisher
from oclp.runtime import OclpRun, observe_run, run_template

Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
DagsterContext: TypeAlias = Any
PublisherFactory: TypeAlias = Callable[
    [DagsterContext], AbstractContextManager[LocalArtifactPublisher]
]
SourceFactory: TypeAlias = Callable[[DagsterContext], ImplementationSource]

_CONTEXT_FIELDS = frozenset(
    {"run_id", "asset_key", "partition_key", "retry_number"}
)
_SOURCE_TYPES = (GitSource, ArtifactSource, ServiceSource, OpaqueSource)


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
) -> Callable[
    [Callable[Parameters, Result]], Callable[Parameters, Result]
]:
    """Observe one Dagster asset through an existing OCLP ``@run`` workflow.

    Apply this decorator directly inside :func:`dagster.asset`::

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


def _context_value(context: DagsterContext, field: str) -> object | None:
    """Return one intentionally supported Dagster context value."""

    if field == "run_id":
        return str(context.run.run_id)
    if field == "asset_key":
        return context.asset_key.to_user_string()
    if field == "partition_key":
        if not context.has_partition_key:
            return None
        return context.partition_key
    if field == "retry_number":
        return int(context.retry_number)
    raise AssertionError(f"unsupported Dagster context field {field!r}")


def _require_dagster() -> None:
    """Raise a focused error when the optional integration is not installed."""

    try:
        import dagster  # noqa: F401
    except ModuleNotFoundError as error:  # pragma: no cover - install dependent
        raise ModuleNotFoundError(
            "Dagster integration requires the optional dependency; "
            "install oclp[dagster]."
        ) from error

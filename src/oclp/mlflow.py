"""Optional MLflow mirror for SDK-observed OCLP runs.

OCLP remains the authoritative provenance system.  This module mirrors
published records into MLflow for experiment tracking convenience; it neither
reads from MLflow to construct lineage nor changes OCLP publication semantics.
The MLflow dependency is imported only when an adapter is activated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oclp.artifacts import ArtifactHandle
from oclp.computations import computation_template
from oclp.models import ArtifactSet, Computation, Evidence, Execution

if TYPE_CHECKING:
    from oclp.runtime import ArtifactSetHandle, OclpRun


_MLFLOW_DECLARATION_ATTRIBUTE = "__oclp_mlflow_declaration__"


@dataclass(frozen=True)
class MlflowModelRegistration:
    """Opt-in registration of one explicitly nominated OCLP model Artifact."""

    artifact_name: str
    registered_model_name: str

    def __post_init__(self) -> None:
        if not self.artifact_name:
            raise ValueError("MLflow model artifact_name must be non-empty")
        if not self.registered_model_name:
            raise ValueError("MLflow registered_model_name must be non-empty")


@dataclass(frozen=True)
class MlflowMetrics:
    """Project numeric fields from one declared JSON output into MLflow.

    ``output_port`` is local to the decorated Computation. ``dimensions``
    names scalar Execution parameters included in metric keys when that
    Computation can execute repeatedly in one observed run.
    """

    output_port: str
    prefix: str | None = None
    dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output_port, str) or not self.output_port:
            raise ValueError("metric output_port must be a non-empty string")
        if self.prefix is not None and (
            not isinstance(self.prefix, str) or not self.prefix
        ):
            raise ValueError("metric output prefix must be a non-empty string")
        if not isinstance(self.dimensions, tuple) or not all(
            isinstance(name, str) and name for name in self.dimensions
        ):
            raise TypeError("metric output dimensions must be a tuple of names")
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("metric output dimensions must be unique")


@dataclass(frozen=True)
class _MlflowDeclaration:
    """SDK-only MLflow projection metadata attached to one Computation."""

    metrics: tuple[MlflowMetrics, ...]
    payloads: frozenset[str]


def mlflow(
    *,
    metrics: tuple[MlflowMetrics, ...] = (),
    payloads: tuple[str, ...] = (),
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Declare optional MLflow projections for one ``@computation``.

    Apply this decorator outside ``@computation``. It is inert unless the
    active ``@run`` includes :class:`MlflowAdapter`; OCLP publication and
    ordinary calls remain unchanged. Model payloads are mirrored by default.
    ``payloads`` opts named additional output-port payloads into MLflow.
    """

    if not isinstance(metrics, tuple) or not all(
        isinstance(metric, MlflowMetrics) for metric in metrics
    ):
        raise TypeError("mlflow metrics must be a tuple of MlflowMetrics")
    if not isinstance(payloads, tuple) or not all(
        isinstance(port, str) and port for port in payloads
    ):
        raise TypeError("mlflow payloads must be a tuple of non-empty output ports")
    if len(payloads) != len(set(payloads)):
        raise ValueError("mlflow payload ports must be unique")
    metric_ports = [metric.output_port for metric in metrics]
    if len(metric_ports) != len(set(metric_ports)):
        raise ValueError("each MLflow metric output port may be selected only once")

    def decorate(function: Callable[..., object]) -> Callable[..., object]:
        template = computation_template(function)
        available_outputs = set(template.output_artifacts)
        unknown_payloads = sorted(set(payloads).difference(available_outputs))
        if unknown_payloads:
            raise ValueError(
                "MLflow payloads must name declared Computation output ports; "
                f"unknown: {', '.join(unknown_payloads)}"
            )
        available_parameters = {
            parameter.name for parameter in template.parameter_definitions
        }
        for metric in metrics:
            if metric.output_port not in template.output_artifacts:
                available = ", ".join(sorted(available_outputs)) or "none"
                raise ValueError(
                    "MLflow metric output_port must name a declared Computation "
                    f"output; available: {available}"
                )
            artifact_type = template.output_artifacts[metric.output_port]
            if artifact_type.media_type != "application/json":
                raise ValueError(
                    "MLflow metric output must select an application/json Artifact "
                    "output"
                )
            unknown_dimensions = sorted(
                set(metric.dimensions).difference(available_parameters)
            )
            if unknown_dimensions:
                raise ValueError(
                    "MLflow metric dimensions must name declared Computation "
                    f"parameters; unknown: {', '.join(unknown_dimensions)}"
                )
        declaration = _MlflowDeclaration(
            metrics=metrics,
            payloads=frozenset(payloads),
        )
        observed_function = getattr(function, "__oclp_observed_function__", function)
        if getattr(function, _MLFLOW_DECLARATION_ATTRIBUTE, None) is not None:
            raise ValueError("a Computation can have only one @mlflow declaration")
        setattr(function, _MLFLOW_DECLARATION_ATTRIBUTE, declaration)
        setattr(observed_function, _MLFLOW_DECLARATION_ATTRIBUTE, declaration)
        return function

    return decorate


def _mlflow_declaration(function: Callable[..., object]) -> _MlflowDeclaration | None:
    """Return a valid local MLflow declaration, if the callable has one."""

    declaration = getattr(function, _MLFLOW_DECLARATION_ATTRIBUTE, None)
    if declaration is None:
        return None
    if not isinstance(declaration, _MlflowDeclaration):  # pragma: no cover - guard.
        raise TypeError("invalid MLflow declaration attached to Computation")
    return declaration


@dataclass
class MlflowAdapter:
    """Mirror an observed OCLP run into one optional MLflow run.

    Canonical OCLP records are logged as JSON. Model payloads are copied by
    default; each ``@mlflow`` Computation declaration can opt its own
    additional output payloads and metric projections in. Set ``strict=True``
    when a mirror failure must fail the application workflow. The default lets
    OCLP publication finish and asks :class:`~oclp.runtime.OclpRun` to publish
    an ``adapter-failed`` Event with a Diagnostic on the next real Execution.
    """

    experiment_name: str
    tracking_uri: str | None = None
    artifact_location: str | None = None
    model_registration: MlflowModelRegistration | None = None
    strict: bool = False
    _mlflow: Any = field(default=None, init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)
    _run_id: str | None = field(default=None, init=False, repr=False)
    _resolved_tracking_uri: str | None = field(default=None, init=False, repr=False)
    _registered_artifact_ids: set[str] = field(
        default_factory=set, init=False, repr=False
    )
    _emitted_metric_names: set[str] = field(default_factory=set, init=False, repr=False)
    _declarations: dict[str, _MlflowDeclaration] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("MLflow experiment_name must be non-empty")

    def for_run(self) -> MlflowAdapter:
        """Return a fresh mirror session from this reusable run declaration."""

        return MlflowAdapter(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
            model_registration=self.model_registration,
            strict=self.strict,
        )

    def on_run_start(self, observed: OclpRun) -> None:
        """Start the MLflow run and write its OCLP run-level tags."""

        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri, artifact_location = self._resolve_destination(observed)
        self._resolved_tracking_uri = tracking_uri
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            client.create_experiment(
                self.experiment_name,
                artifact_location=artifact_location,
            )
        mlflow.set_experiment(self.experiment_name)
        active = mlflow.start_run(run_name=observed.run_name)
        self._mlflow = mlflow
        self._client = client
        self._run_id = active.info.run_id
        tags = {"oclp.sdk.adapter": "mlflow"}
        if observed.run_id is not None:
            tags["oclp.run.id"] = str(observed.run_id)
        if observed.run_name is not None:
            tags["oclp.run.name"] = observed.run_name
        mlflow.set_tags(tags)

    def _resolve_destination(self, observed: OclpRun) -> tuple[str | None, str | None]:
        """Resolve explicit settings or a local store beside OCLP records."""

        if self.tracking_uri is not None or self.artifact_location is not None:
            return self.tracking_uri, self.artifact_location
        root = observed.publisher.record_root.parent / "mlflow"
        artifact_root = root / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + (root / "mlflow.db").resolve().as_posix(), (
            artifact_root.resolve().as_uri()
        )

    def on_run_end(self, _observed: OclpRun, error: BaseException | None) -> None:
        """End the mirror run after the OCLP observation context closes."""

        if self._mlflow is None:
            return
        status = "FAILED" if error is not None else "FINISHED"
        self._mlflow.end_run(status=status)

    def on_artifact(self, _observed: OclpRun, artifact: ArtifactHandle) -> None:
        """Mirror one Artifact record and any selected payload bytes."""

        self._log_artifact(artifact)

    def on_computation(
        self,
        _observed: OclpRun,
        *,
        computation: Computation,
        function: Callable[..., object],
    ) -> None:
        """Remember optional local MLflow policy for one Core Computation."""

        declaration = _mlflow_declaration(function)
        if declaration is not None:
            self._declarations[computation.id] = declaration

    def on_execution(
        self,
        _observed: OclpRun,
        *,
        execution: Execution,
        computation: Computation,
        outputs: dict[str, ArtifactHandle],
        evidence: tuple[Evidence, ...],
    ) -> None:
        """Mirror a completed execution, its typed parameters, and Evidence."""

        self._require_started()
        self._log_record(execution)
        self._log_record(computation)
        self._mlflow.set_tags(
            {
                "oclp.execution.id": execution.id,
                "oclp.computation.id": computation.id,
                f"oclp.execution.{execution.id}": computation.id,
            }
        )
        # MLflow parameters are immutable within one MLflow run. One OCLP run
        # may execute the same Computation repeatedly with different values
        # (for example, temporal fold_number), so parameter keys must retain
        # the exact Execution identity rather than colliding by Computation.
        parameter_prefix = ".".join(
            (
                _mlflow_component(computation.name or computation.id),
                _mlflow_component(execution.id),
            )
        )
        self._mlflow.log_params(
            {
                f"{parameter_prefix}.{name}": _parameter_value(value)
                for name, value in execution.parameters.items()
            }
        )
        declaration = self._declarations.get(computation.id)
        selected_payloads = (
            declaration.payloads if declaration is not None else frozenset()
        )
        for port, artifact in outputs.items():
            self._log_artifact(artifact, selected_payload=port in selected_payloads)
        for record in evidence:
            self._log_record(record)
            metrics = _numeric_evidence_metrics(record)
            if metrics:
                prefix = _mlflow_component(record.name or record.id)
                self._mlflow.log_metrics(
                    {f"{prefix}.{name}": value for name, value in metrics.items()}
                )
        self._log_declared_output_metrics(
            computation=computation,
            execution=execution,
            outputs=outputs,
            declaration=declaration,
        )

    def on_artifact_set(
        self,
        _observed: OclpRun,
        artifact_set: ArtifactSetHandle,
    ) -> None:
        """Mirror one immutable collection record and any sidecar manifest."""

        self._require_started()
        self._log_record(artifact_set.artifact_set)
        self._mlflow.set_tags(
            {
                "oclp.artifact_set.id": artifact_set.reference.id,
                f"oclp.artifact_set.{artifact_set.reference.id}": (
                    artifact_set.artifact_set.name or artifact_set.reference.id
                ),
            }
        )
        if artifact_set.manifest is not None:
            self._log_artifact(artifact_set.manifest)

    def _log_artifact(
        self, artifact: ArtifactHandle, *, selected_payload: bool = False
    ) -> None:
        self._require_started()
        if artifact.reference.id in self._registered_artifact_ids:
            return
        self._registered_artifact_ids.add(artifact.reference.id)
        self._log_record(artifact.artifact)
        self._mlflow.set_tag(
            f"oclp.artifact.{artifact.reference.id}", artifact.artifact.digest.value
        )
        nominated_for_registration = (
            self.model_registration is not None
            and self.model_registration.artifact_name == artifact.artifact.name
        )
        if nominated_for_registration or _should_upload_payload(
            artifact, selected_payload=selected_payload
        ):
            artifact_name = artifact.artifact.name or artifact.reference.id
            component = _mlflow_component(artifact_name)
            # Artifact names describe the payload for people; they are not a
            # unique storage key. Repeated executions (such as temporal
            # folds) commonly publish same-named artifacts with the same file
            # name, so scope the MLflow directory by the immutable Artifact
            # UUID to avoid overwriting an earlier payload.
            destination = f"oclp/payloads/{component}/{artifact.reference.id}"
            self._mlflow.log_artifact(str(artifact.path), artifact_path=destination)
            self._register_model_if_requested(artifact, destination)

    def _log_record(
        self,
        record: ArtifactSet | Computation | Evidence | Execution | ArtifactHandle,
    ) -> None:
        if isinstance(record, ArtifactHandle):
            payload = record.artifact.model_dump(mode="json")
            kind = "artifact"
            record_id = record.reference.id
        else:
            payload = record.model_dump(mode="json")
            kind = record.kind
            record_id = record.id
        self._mlflow.log_dict(payload, f"oclp/records/{kind}/{record_id}.json")

    def _register_model_if_requested(
        self, artifact: ArtifactHandle, destination: str
    ) -> None:
        registration = self.model_registration
        if registration is None or registration.artifact_name != artifact.artifact.name:
            return
        assert self._client is not None
        try:
            self._client.get_registered_model(registration.registered_model_name)
        except Exception:
            self._client.create_registered_model(registration.registered_model_name)
        self._client.create_model_version(
            name=registration.registered_model_name,
            source=artifact.path.resolve().as_uri(),
            run_id=self._run_id,
            tags={
                "oclp.artifact.id": artifact.reference.id,
                "oclp.artifact.media_type": artifact.artifact.media_type,
                "oclp.mlflow.payload_path": f"{destination}/{artifact.path.name}",
            },
        )

    def _require_started(self) -> None:
        if self._mlflow is None:
            raise RuntimeError("MLflow adapter has no active MLflow run")

    def log_parameters(self, parameters: Mapping[str, object]) -> None:
        """Log application-selected run-level parameters to the active mirror."""

        self._require_started()
        self._mlflow.log_params(
            {name: _parameter_value(value) for name, value in parameters.items()}
        )

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        """Log application-selected domain metrics to the active mirror."""

        self._require_started()
        self._mlflow.log_metrics(
            {
                name: float(value)
                for name, value in metrics.items()
                if not isinstance(value, bool)
            }
        )

    def _log_declared_output_metrics(
        self,
        *,
        computation: Computation,
        execution: Execution,
        outputs: Mapping[str, ArtifactHandle],
        declaration: _MlflowDeclaration | None,
    ) -> None:
        """Extract opted-in JSON output scalars into collision-safe MLflow keys."""

        if declaration is None:
            return
        for metric in declaration.metrics:
            artifact = outputs.get(metric.output_port)
            if artifact is None:
                raise ValueError(
                    "declared MLflow metric output was not materialized: "
                    f"{computation.name or computation.id}.{metric.output_port}"
                )
            values = _numeric_json_artifact_fields(artifact)
            if not values:
                continue
            prefix = _metric_output_prefix(
                declaration=metric,
                computation=computation,
                execution=execution,
            )
            metrics = {f"{prefix}.{name}": value for name, value in values.items()}
            collisions = sorted(set(metrics).intersection(self._emitted_metric_names))
            if collisions:
                raise ValueError(
                    "MLflow metric output keys would collide in one run: "
                    f"{', '.join(collisions)}; add dimensions to "
                    "MlflowMetrics for repeated Computation calls"
                )
            self._mlflow.log_metrics(metrics)
            self._emitted_metric_names.update(metrics)


def _should_upload_payload(artifact: ArtifactHandle, *, selected_payload: bool) -> bool:
    """Return whether a payload is a model or an explicitly selected copy."""

    media_type = artifact.artifact.media_type.lower()
    return (
        selected_payload
        or "model" in media_type
        or media_type.startswith("application/x-catboost")
    )


def _numeric_evidence_metrics(record: Evidence) -> dict[str, float]:
    """Select top-level numeric Evidence details suitable for MLflow metrics."""

    return {
        name: float(value)
        for name, value in record.details.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _numeric_json_artifact_fields(artifact: ArtifactHandle) -> dict[str, float]:
    """Return top-level numeric scalar fields from one verified JSON Artifact."""

    try:
        value = json.loads(artifact.read_verified_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"selected MLflow metric Artifact {artifact.reference.id} is not valid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise ValueError(
            "selected MLflow metric Artifact "
            f"{artifact.reference.id} must contain a JSON object"
        )
    return {
        _mlflow_component(name): float(field)
        for name, field in value.items()
        if isinstance(name, str)
        and isinstance(field, (int, float))
        and not isinstance(field, bool)
    }


def _metric_output_prefix(
    *,
    declaration: MlflowMetrics,
    computation: Computation,
    execution: Execution,
) -> str:
    """Build one stable MLflow metric prefix from explicit selection metadata."""

    prefix = _mlflow_component(
        declaration.prefix or computation.name or computation.implementation.locator
    )
    components = [prefix]
    for name in declaration.dimensions:
        value = execution.parameters[name]
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError(
                "MLflow metric output dimensions must have scalar Execution values: "
                f"{name!r} on {execution.id}"
            )
        components.append(f"{_mlflow_component(name)}-{_mlflow_component(str(value))}")
    return ".".join(components)


def _parameter_value(value: object) -> str:
    """Return MLflow's scalar/string representation without losing OCLP JSON."""

    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)


def _mlflow_component(value: str) -> str:
    """Return one stable MLflow-safe path or metric component."""

    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return component or "oclp"

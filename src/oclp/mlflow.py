"""Optional MLflow mirror for SDK-observed OCLP runs.

OCLP remains the authoritative provenance system.  This module mirrors
published records into MLflow for experiment tracking convenience; it neither
reads from MLflow to construct lineage nor changes OCLP publication semantics.
The MLflow dependency is imported only when an adapter is activated.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oclp.artifacts import ArtifactHandle
from oclp.computations import ComputationOutput, computation_template
from oclp.models import ArtifactSet, Computation, Evidence, Execution

if TYPE_CHECKING:
    from oclp.runtime import ArtifactSetHandle, OclpRun


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
class MlflowMetricOutput:
    """Select numeric fields from one exact JSON Computation output.

    ``output`` identifies a decorated callable and one of its declared output
    ports. ``dimensions`` names scalar Execution parameters incorporated into
    MLflow metric keys, which is necessary when the selected Computation may
    run more than once in one observed run.
    """

    output: ComputationOutput
    prefix: str | None = None
    dimensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.output, ComputationOutput):
            raise TypeError("metric output must be a ComputationOutput declaration")
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
        template = computation_template(self.output.function)
        available = {parameter.name for parameter in template.parameter_definitions}
        unknown = sorted(set(self.dimensions).difference(available))
        if unknown:
            raise ValueError(
                "metric output dimensions must name declared Computation "
                f"parameters; unknown: {', '.join(unknown)}"
            )
        artifact_type = template.output_artifacts[self.output.port]
        if artifact_type.media_type != "application/json":
            raise ValueError(
                "metric output must select an application/json Artifact output"
            )


@dataclass
class MlflowAdapter:
    """Mirror an observed OCLP run into one optional MLflow run.

    Canonical OCLP records are logged as JSON. Model payloads are copied by
    default; other payloads are copied only when their semantic artifact name
    appears in ``payload_artifacts``.  Set ``strict=True`` when a mirror
    failure must fail the application workflow.  The default lets OCLP
    publication finish and asks :class:`~oclp.runtime.OclpRun` to publish an
    ``adapter-failed`` Event with a Diagnostic on the next real Execution.
    """

    experiment_name: str
    tracking_uri: str | None = None
    artifact_location: str | None = None
    payload_artifacts: frozenset[str] = frozenset()
    metric_outputs: tuple[MlflowMetricOutput, ...] = ()
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

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("MLflow experiment_name must be non-empty")
        if not isinstance(self.payload_artifacts, frozenset) or not all(
            isinstance(name, str) and name for name in self.payload_artifacts
        ):
            raise TypeError("payload_artifacts must be a frozenset of non-empty names")
        if not isinstance(self.metric_outputs, tuple) or not all(
            isinstance(output, MlflowMetricOutput) for output in self.metric_outputs
        ):
            raise TypeError("metric_outputs must be a tuple of MlflowMetricOutput")
        selected_outputs = [
            (output.output.locator, output.output.port)
            for output in self.metric_outputs
        ]
        if len(selected_outputs) != len(set(selected_outputs)):
            raise ValueError("each MLflow metric output may be selected only once")

    def for_run(self) -> MlflowAdapter:
        """Return a fresh mirror session from this reusable run declaration."""

        return MlflowAdapter(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
            payload_artifacts=self.payload_artifacts,
            metric_outputs=self.metric_outputs,
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
        for artifact in outputs.values():
            self._log_artifact(artifact)
        for record in evidence:
            self._log_record(record)
            metrics = _numeric_evidence_metrics(record)
            if metrics:
                prefix = _mlflow_component(record.name or record.id)
                self._mlflow.log_metrics(
                    {f"{prefix}.{name}": value for name, value in metrics.items()}
                )
        self._log_selected_output_metrics(
            computation=computation,
            execution=execution,
            outputs=outputs,
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

    def _log_artifact(self, artifact: ArtifactHandle) -> None:
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
            artifact, self.payload_artifacts
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

    def _log_selected_output_metrics(
        self,
        *,
        computation: Computation,
        execution: Execution,
        outputs: Mapping[str, ArtifactHandle],
    ) -> None:
        """Extract opted-in JSON output scalars into collision-safe MLflow keys."""

        for declaration in self.metric_outputs:
            if declaration.output.locator != computation.implementation.locator:
                continue
            artifact = outputs.get(declaration.output.port)
            if artifact is None:
                raise ValueError(
                    "selected MLflow metric output was not materialized: "
                    f"{declaration.output.locator}.{declaration.output.port}"
                )
            values = _numeric_json_artifact_fields(artifact)
            if not values:
                continue
            prefix = _metric_output_prefix(
                declaration=declaration,
                computation=computation,
                execution=execution,
            )
            metrics = {f"{prefix}.{name}": value for name, value in values.items()}
            collisions = sorted(set(metrics).intersection(self._emitted_metric_names))
            if collisions:
                raise ValueError(
                    "MLflow metric output keys would collide in one run: "
                    f"{', '.join(collisions)}; add dimensions to "
                    "MlflowMetricOutput for repeated Computation calls"
                )
            self._mlflow.log_metrics(metrics)
            self._emitted_metric_names.update(metrics)


def _should_upload_payload(
    artifact: ArtifactHandle, selected_names: frozenset[str]
) -> bool:
    """Return whether a payload is a model or an explicitly selected copy."""

    media_type = artifact.artifact.media_type.lower()
    return (
        artifact.artifact.name in selected_names
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
    declaration: MlflowMetricOutput,
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

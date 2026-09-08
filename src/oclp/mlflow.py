"""Optional MLflow mirror for SDK-observed OCLP runs.

OCLP remains the authoritative provenance system.  This module mirrors
published records into MLflow for experiment tracking convenience; it neither
reads from MLflow to construct lineage nor changes OCLP publication semantics.
The MLflow dependency is imported only when an adapter is activated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oclp.artifacts import ArtifactHandle
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
    model_registration: MlflowModelRegistration | None = None
    strict: bool = False
    _mlflow: Any = field(default=None, init=False, repr=False)
    _client: Any = field(default=None, init=False, repr=False)
    _run_id: str | None = field(default=None, init=False, repr=False)
    _resolved_tracking_uri: str | None = field(default=None, init=False, repr=False)
    _registered_artifact_ids: set[str] = field(
        default_factory=set, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("MLflow experiment_name must be non-empty")
        if not isinstance(self.payload_artifacts, frozenset) or not all(
            isinstance(name, str) and name for name in self.payload_artifacts
        ):
            raise TypeError("payload_artifacts must be a frozenset of non-empty names")

    def for_run(self) -> MlflowAdapter:
        """Return a fresh mirror session from this reusable run declaration."""

        return MlflowAdapter(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            artifact_location=self.artifact_location,
            payload_artifacts=self.payload_artifacts,
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
        parameter_prefix = _mlflow_component(computation.name or computation.id)
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
            destination = f"oclp/payloads/{component}"
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


def _parameter_value(value: object) -> str:
    """Return MLflow's scalar/string representation without losing OCLP JSON."""

    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)


def _mlflow_component(value: str) -> str:
    """Return one stable MLflow-safe path or metric component."""

    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return component or "oclp"

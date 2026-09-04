"""The bike-demand application's explicit MLflow integration boundary."""

from __future__ import annotations

import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oclp import ArtifactHandle
from oclp.models import RecordReference

DEFAULT_EXPERIMENT_NAME = "oclp-bike-demand-service"


@dataclass(frozen=True)
class MLflowSettings:
    """Locations and identifiers for the demo's local MLflow tracking data."""

    root: Path
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    @property
    def database_path(self) -> Path:
        """Return the SQLite database that holds local MLflow run metadata."""

        return self.root / "mlflow.db"

    @property
    def artifact_root(self) -> Path:
        """Return the local artifact area for MLflow copies and manifests."""

        return self.root / "artifacts"

    @property
    def tracking_uri(self) -> str:
        """Return the absolute SQLite tracking URI expected by MLflow."""

        return "sqlite:///" + self.database_path.resolve().as_posix()

    def prepare(self) -> None:
        """Create local directories before MLflow is configured."""

        self.artifact_root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MLflowTracker:
    """Small application-facing façade around the optional MLflow dependency."""

    settings: MLflowSettings
    _mlflow: Any

    @property
    def tracking_uri(self) -> str:
        """Return the configured MLflow tracking URI."""

        return self.settings.tracking_uri

    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Generator[None, None, None]:
        """Open one named MLflow run without exposing the MLflow module."""

        with self._mlflow.start_run(run_name=name, nested=nested):
            yield

    def active_run_id(self) -> str:
        """Return the ID of the active MLflow run."""

        active = self._mlflow.active_run()
        if active is None:
            raise RuntimeError("MLflow has no active run")
        return active.info.run_id

    def log_parameters(self, parameters: Mapping[str, Any]) -> None:
        """Log application-selected scalar or stringified MLflow parameters."""

        self._mlflow.log_params(dict(parameters))

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        """Log application-selected scalar metrics to the active MLflow run."""

        self._mlflow.log_metrics(dict(metrics))

    def attach_execution(
        self,
        *,
        execution: RecordReference,
        computation: RecordReference,
        inputs: Mapping[str, tuple[RecordReference, ...]],
        outputs: Mapping[str, tuple[RecordReference, ...]] | None,
        artifacts: Mapping[str, ArtifactHandle] | None = None,
    ) -> None:
        """Link one OCLP Execution and mirror its output payloads."""

        self._mlflow.set_tags(
            {
                "oclp.execution.id": execution.id,
                "oclp.computation.id": computation.id,
            }
        )
        self._mlflow.log_dict(
            {
                "execution": execution.model_dump(mode="json"),
                "computation": computation.model_dump(mode="json"),
                "inputs": _references_json(inputs),
                "outputs": _references_json(outputs or {}),
                "note": (
                    "OCLP records are linked by immutable UUID; output payloads "
                    "are mirrored below."
                ),
            },
            "oclp/record-links.json",
        )
        if artifacts:
            self.mirror_artifacts(
                artifacts=artifacts,
                artifact_path="oclp/outputs",
            )

    def attach_artifact_set(
        self,
        *,
        artifact_set: RecordReference,
        artifacts: Mapping[str, ArtifactHandle] | None = None,
    ) -> None:
        """Link a published ArtifactSet and mirror its release payloads."""

        self._mlflow.set_tags(
            {
                "oclp.artifact_set.id": artifact_set.id,
            }
        )
        self._mlflow.log_dict(
            {
                "artifact_set": artifact_set.model_dump(mode="json"),
                "note": (
                    "OCLP ArtifactSet members are linked by immutable UUID and "
                    "their payloads are mirrored below. Publication is not a "
                    "Computation or Execution."
                ),
            },
            "oclp/artifact-set-link.json",
        )
        if artifacts:
            self.mirror_artifacts(
                artifacts=artifacts,
                artifact_path="oclp/release",
            )

    def mirror_artifacts(
        self,
        *,
        artifacts: Mapping[str, ArtifactHandle],
        artifact_path: str,
    ) -> None:
        """Mirror exact OCLP payload bytes into the active MLflow run.

        OCLP remains the source of truth. MLflow receives convenient copies
        accompanied by OCLP record UUIDs and payload digests for experiment
        inspection.
        """

        entries: list[dict[str, object]] = []
        for name, artifact in artifacts.items():
            component = _artifact_path_component(name)
            destination = f"{artifact_path}/{component}"
            self._mlflow.log_artifact(str(artifact.path), artifact_path=destination)
            entries.append(
                {
                    "name": name,
                    "reference": artifact.reference.model_dump(mode="json"),
                    "payload_digest": artifact.artifact.digest.model_dump(mode="json"),
                    "media_type": artifact.artifact.media_type,
                    "size": artifact.artifact.size,
                    "mlflow_path": f"{destination}/{artifact.path.name}",
                }
            )
        self._mlflow.log_dict(
            {"artifacts": entries},
            f"{artifact_path}/artifact-manifest.json",
        )


def create_mlflow_tracker(settings: MLflowSettings) -> MLflowTracker:
    """Configure local MLflow and return the application's tracker façade.

    MLflow is a dependency of this example, not of the OCLP SDK, so importing
    it remains isolated to this application integration module.
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    settings.prepare()
    mlflow.set_tracking_uri(settings.tracking_uri)
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        client.create_experiment(
            settings.experiment_name,
            artifact_location=settings.artifact_root.resolve().as_uri(),
        )
    mlflow.set_experiment(settings.experiment_name)
    return MLflowTracker(settings=settings, _mlflow=mlflow)


def _references_json(
    values: Mapping[str, tuple[RecordReference, ...]],
) -> dict[str, list[dict[str, object]]]:
    return {
        port: [reference.model_dump(mode="json") for reference in references]
        for port, references in values.items()
    }


def _artifact_path_component(name: str) -> str:
    """Map a semantic Artifact name to one safe MLflow path component."""

    component = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not component:
        raise ValueError("OCLP Artifact names need a usable MLflow path component")
    return component

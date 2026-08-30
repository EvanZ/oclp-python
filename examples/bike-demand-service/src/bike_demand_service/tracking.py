"""Local MLflow configuration and deliberately narrow OCLP correlation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        """Return the local artifact area for MLflow-only reports and manifests."""

        return self.root / "artifacts"

    @property
    def tracking_uri(self) -> str:
        """Return the absolute SQLite tracking URI expected by MLflow."""

        return "sqlite:///" + self.database_path.resolve().as_posix()

    def prepare(self) -> None:
        """Create local directories before MLflow is configured."""

        self.artifact_root.mkdir(parents=True, exist_ok=True)


def configure_mlflow(settings: MLflowSettings) -> Any:
    """Configure a local tracking backend and return the MLflow module.

    The import is intentionally local: MLflow is a dependency of this example,
    not of the OCLP SDK itself.
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
    return mlflow


def log_oclp_bridge(
    mlflow: Any,
    *,
    invocation: RecordReference,
    definition: RecordReference,
    inputs: dict[str, tuple[RecordReference, ...]],
    outputs: dict[str, tuple[RecordReference, ...]] | None,
) -> None:
    """Attach immutable OCLP links without copying OCLP payload bytes to MLflow."""

    mlflow.set_tags(
        {
            "oclp.invocation.id": invocation.id,
            "oclp.invocation.digest": _digest_value(invocation),
            "oclp.definition.id": definition.id,
            "oclp.definition.digest": _digest_value(definition),
        }
    )
    mlflow.log_dict(
        {
            "invocation": invocation.model_dump(mode="json"),
            "definition": definition.model_dump(mode="json"),
            "inputs": _references_json(inputs),
            "outputs": _references_json(outputs or {}),
            "note": (
                "OCLP artifacts are linked by digest; bytes are not duplicated "
                "in MLflow."
            ),
        },
        "oclp/record-links.json",
    )


def _digest_value(reference: RecordReference) -> str:
    if reference.digest is None:  # Defensive: all demo links are content-bound.
        raise ValueError(
            f"OCLP bridge requires a digest-bound reference: {reference.id}"
        )
    return reference.digest.value


def _references_json(
    values: dict[str, tuple[RecordReference, ...]],
) -> dict[str, list[dict[str, object]]]:
    return {
        port: [reference.model_dump(mode="json") for reference in references]
        for port, references in values.items()
    }

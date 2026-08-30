"""Explicit reusable OCLP Definitions for the bike-demand demo boundaries."""

from __future__ import annotations

import subprocess
from pathlib import Path

from oclp import ComputationDefinition, GitSource, OpaqueSource
from oclp.models import Implementation, PortDefinition


def _implementation_source(project_root: Path) -> GitSource | OpaqueSource:
    """Bind Definitions to this checkout when Git metadata is available."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return OpaqueSource(
            reason="Git source metadata was unavailable at observation time."
        )
    return GitSource(
        repository=repository or "https://github.com/EvanZ/oclp-python.git",
        commit=commit,
        path="examples/bike-demand-service/src/bike_demand_service",
    )


def definitions(project_root: Path) -> dict[str, ComputationDefinition]:
    """Return all computation boundaries used in the first batch milestone."""

    source = _implementation_source(project_root)

    def build(
        *,
        key: str,
        name: str,
        locator: str,
        inputs: tuple[PortDefinition, ...] = (),
        outputs: tuple[PortDefinition, ...] = (),
    ) -> ComputationDefinition:
        return ComputationDefinition(
            id=f"urn:oclp-bike-demand:definition:{key}",
            name=name,
            implementation=Implementation(
                kind="python-callable",
                locator=locator,
                source=source,
            ),
            input_ports=inputs,
            output_ports=outputs,
        )

    def json_port(name: str, **kwargs: object) -> PortDefinition:
        return PortDefinition(
            name=name,
            media_types=("application/json",),
            **kwargs,
        )

    def artifact_port(name: str, **kwargs: object) -> PortDefinition:
        return PortDefinition(name=name, **kwargs)

    return {
        "lifecycle": build(
            key="run-bike-demand-model-lifecycle",
            name="Run bike-demand model lifecycle",
            locator="bike_demand_service.runner.run_demo",
            outputs=(json_port("training_plan"),),
        ),
        "ingest": build(
            key="ingest-source-data",
            name="Ingest bike-demand source data",
            locator="bike_demand_service.data.download_source_data",
            outputs=(artifact_port("raw_dataset"),),
        ),
        "prepare": build(
            key="prepare-features",
            name="Prepare bike-demand features and folds",
            locator="bike_demand_service.data.prepare_features",
            inputs=(artifact_port("raw_dataset"),),
            outputs=(
                artifact_port("features"),
                json_port("dataset_snapshot"),
                json_port("fold_definition"),
                json_port("feature_contract"),
            ),
        ),
        "train_fold": build(
            key="train-fold-model",
            name="Train bike-demand fold model",
            locator="bike_demand_service.modeling.train_fold",
            inputs=(json_port("dataset_snapshot"), json_port("fold_definition")),
            outputs=(
                artifact_port("model"),
                artifact_port("validation_predictions"),
                json_port("metrics"),
            ),
        ),
        "evaluate": build(
            key="evaluate-candidate",
            name="Evaluate bike-demand candidate",
            locator="bike_demand_service.modeling.evaluate_folds",
            inputs=(
                artifact_port("fold_models", cardinality="many"),
                artifact_port("fold_predictions", cardinality="many"),
                json_port("fold_metrics", cardinality="many"),
            ),
            outputs=(json_port("evaluation"), json_port("training_config")),
        ),
        "train_final": build(
            key="train-final-model",
            name="Train final bike-demand model",
            locator="bike_demand_service.modeling.train_final_model",
            inputs=(json_port("dataset_snapshot"), json_port("training_config")),
            outputs=(artifact_port("model"),),
        ),
        "package": build(
            key="package-model-release",
            name="Package bike-demand model release",
            locator="bike_demand_service.runner.package_model_release",
            inputs=(
                artifact_port("model"),
                json_port("feature_contract"),
                json_port("evaluation"),
                json_port("training_config"),
            ),
            outputs=(artifact_port("model_release"),),
        ),
        "score": build(
            key="predict-bike-demand",
            name="Score bike-demand holdout set",
            locator="bike_demand_service.modeling.score_holdout",
            inputs=(artifact_port("model_release"), json_port("dataset_snapshot")),
            outputs=(artifact_port("predictions"), json_port("metrics")),
        ),
    }

"""Declared computation boundaries for the future end-to-end demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannedStage:
    """One planned OCLP Definition boundary in the demo."""

    name: str
    definition_id: str
    description: str


PLANNED_STAGES = (
    PlannedStage(
        name="Ingest bike-demand dataset",
        definition_id="urn:oclp-bike-demand:definition:ingest-source-data",
        description="Download and publish the raw UCI CSV as an immutable Artifact.",
    ),
    PlannedStage(
        name="Prepare features and folds",
        definition_id="urn:oclp-bike-demand:definition:prepare-features",
        description=(
            "Produce leakage-safe features, a DatasetSnapshot, and temporal folds."
        ),
    ),
    PlannedStage(
        name="Train fold model",
        definition_id="urn:oclp-bike-demand:definition:train-fold-model",
        description=(
            "Run one child Invocation and nested MLflow run per temporal fold."
        ),
    ),
    PlannedStage(
        name="Evaluate candidate",
        definition_id="urn:oclp-bike-demand:definition:evaluate-candidate",
        description=(
            "Publish holdout predictions, metrics Artifacts, and quality-gate Evidence."
        ),
    ),
    PlannedStage(
        name="Package model release",
        definition_id="urn:oclp-bike-demand:definition:package-model-release",
        description=(
            "Create an ArtifactSet for the selected model and its serving contract."
        ),
    ),
    PlannedStage(
        name="Predict bike demand",
        definition_id="urn:oclp-bike-demand:definition:predict-bike-demand",
        description="Bind a model release to an offline or FastAPI prediction request.",
    ),
)

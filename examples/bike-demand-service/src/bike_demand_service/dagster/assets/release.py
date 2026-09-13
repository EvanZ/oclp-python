"""The terminal ArtifactSet asset for one bike-demand release cycle."""

from __future__ import annotations

import dagster as dg
from oclp import ArtifactHandle, assemble_artifact_set
from oclp.dagster import dagster_adapter

from bike_demand_service.dagster.partitions import training_cycles


@dg.asset(
    key="bike_demand_model_release",
    ins={
        "release_cycle": dg.AssetIn(key=dg.AssetKey("bike_demand_release_cycle")),
        "features": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
        "feature_contract": dg.AssetIn(key=dg.AssetKey("bike_demand_feature_contract")),
        "evaluation": dg.AssetIn(key=dg.AssetKey("bike_demand_candidate_evaluation")),
        "training_config": dg.AssetIn(key=dg.AssetKey("bike_demand_training_config")),
        "model": dg.AssetIn(key=dg.AssetKey("bike_demand_final_model")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    required_resource_keys={"oclp"},
)
@assemble_artifact_set(
    name="Bike demand CatBoost release",
    members={
        "release-cycle": ("release_cycle", "release-cycle"),
        "features": ("features", "training-data"),
        "feature-contract": ("feature_contract", "serving-contract"),
        "temporal-evaluation": ("evaluation", "validation-report"),
        "training-config": ("training_config", "training-config"),
        "model": ("model", "model"),
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_model_release(
    release_cycle: ArtifactHandle,
    features: ArtifactHandle,
    feature_contract: ArtifactHandle,
    evaluation: ArtifactHandle,
    training_config: ArtifactHandle,
    model: ArtifactHandle,
) -> None:
    """Assemble the release from exact, visible Artifact asset partitions."""

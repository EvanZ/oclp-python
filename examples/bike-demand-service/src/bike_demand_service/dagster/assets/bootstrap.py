"""Assets that create a release cycle and its immutable training plan."""

from __future__ import annotations

import json

import dagster as dg
from oclp import ArtifactHandle, json_artifact
from oclp.dagster import dagster_adapter

from bike_demand_service.dagster.lifecycle import (
    bootstrap_mlflow_parent_profiles,
    release_cycle_id_for_start,
    start_mlflow_parent_profiles,
)
from bike_demand_service.dagster.partitions import (
    IO_MANAGER_KEY,
    TEMPORAL_VALIDATION_RMSE_MAX,
    training_cycle_id,
    training_cycles,
)
from bike_demand_service.dagster.resources import materialization_id
from bike_demand_service.mlflow_parent import MLFLOW_PARENT_PROFILE
from bike_demand_service.modeling import release_cycle_document, training_plan_document


@dg.asset(
    key="bike_demand_release_cycle_request",
    group_name="bike_demand",
    io_manager_key=IO_MANAGER_KEY,
    config_schema={
        "fold_count": dg.Field(int, default_value=3, is_required=False),
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    required_resource_keys={"oclp"},
)
@json_artifact(
    name="Bike demand release-cycle request",
    description_from_docstring=True,
    adapters=(dagster_adapter(),),
)
def bike_demand_release_cycle_request(
    context: dg.AssetExecutionContext,
) -> dict[str, object]:
    """Start a cycle without first selecting a dynamic Dagster partition."""

    release_cycle_id = release_cycle_id_for_start(context)
    parent = start_mlflow_parent_profiles(context)[MLFLOW_PARENT_PROFILE]
    return release_cycle_document(
        release_cycle_id=release_cycle_id,
        mlflow_parent_run_id=str(parent["mlflow_parent_run_id"]),
        fold_count=int(context.op_execution_context.op_config["fold_count"]),
        temporal_validation_rmse_max=float(
            context.op_execution_context.op_config["temporal_validation_rmse_max"]
        ),
    )


@dg.asset(
    key="bike_demand_release_cycle",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    config_schema={
        "fold_count": dg.Field(int, default_value=3, is_required=False),
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    required_resource_keys={"oclp"},
)
@json_artifact(
    name="Bike demand release cycle",
    description_from_docstring=True,
    adapters=(dagster_adapter(),),
)
def bike_demand_release_cycle(context: dg.AssetExecutionContext) -> dict[str, object]:
    """Persist one application-owned release cycle and its MLflow parent run."""

    release_cycle_id = training_cycle_id(context)
    parent = bootstrap_mlflow_parent_profiles(context)[MLFLOW_PARENT_PROFILE]
    return release_cycle_document(
        release_cycle_id=release_cycle_id,
        mlflow_parent_run_id=str(parent["mlflow_parent_run_id"]),
        fold_count=int(context.op_execution_context.op_config["fold_count"]),
        temporal_validation_rmse_max=float(
            context.op_execution_context.op_config["temporal_validation_rmse_max"]
        ),
    )


@dg.asset(
    key="bike_demand_training_plan",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    ins={"release_cycle": dg.AssetIn(key=dg.AssetKey("bike_demand_release_cycle"))},
    required_resource_keys={"oclp"},
)
@json_artifact(
    name="Bike demand training plan",
    description_from_docstring=True,
    adapters=(dagster_adapter(),),
)
def bike_demand_training_plan(
    release_cycle: ArtifactHandle,
    context: dg.AssetExecutionContext,
) -> dict[str, object]:
    """Persist the selected fold count as the cycle's immutable plan."""

    try:
        cycle = json.loads(release_cycle.read_verified_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("release-cycle Artifact must contain JSON") from error
    if not isinstance(cycle, dict) or not isinstance(cycle.get("fold_count"), int):
        raise ValueError("release-cycle Artifact fold_count must be an integer")
    return training_plan_document(
        materialization_id=materialization_id(context),
        fold_count=int(cycle["fold_count"]),
    )

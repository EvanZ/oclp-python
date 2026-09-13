"""OCLP runtime resources used by the native Dagster declarations."""

from __future__ import annotations

import dagster as dg
from oclp import Lifecycle, MlflowAdapter
from oclp.dagster import oclp_artifact_io_manager, oclp_dagster_resource
from oclp.publishing import LocalArtifactPublisher
from oclp.sources import source_from_git_checkout

from bike_demand_service.dagster.lifecycle import (
    bootstrap_lifecycle,
    bootstrap_mlflow_parent_profiles,
    release_cycle_lifecycle,
    release_cycle_mlflow_parent_profiles,
    start_lifecycle,
    start_mlflow_parent_profiles,
)
from bike_demand_service.dagster.partitions import IO_MANAGER_KEY, training_cycle_id
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.mlflow_parent import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_PARENT_PROFILE,
)


def materialization_id(context: dg.AssetExecutionContext) -> str:
    """Use the durable training-cycle key, not one transient Dagster run ID."""

    if not context.has_partition_key:
        return f"dagster-release-start-{context.run.run_id}"
    return f"dagster-cycle-{training_cycle_id(context)}"


def publisher_for(context: dg.AssetExecutionContext) -> LocalArtifactPublisher:
    """Create an isolated payload destination for one Dagster asset attempt."""

    environment = DemoEnvironment.default()
    environment.prepare()
    partition = str(context.partition_key) if context.has_partition_key else "none"
    return LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.materialization_root(
            "/".join(
                (
                    materialization_id(context),
                    "assets",
                    context.get_step_execution_context().step.key,
                    f"partition-{partition}",
                    f"attempt-{context.retry_number}",
                )
            )
        ),
    )


def source_for(_context: dg.AssetExecutionContext):
    """Use the real bike-demand checkout as the implementation source basis."""

    environment = DemoEnvironment.default()
    return source_from_git_checkout(
        environment.project_root,
        path="src/bike_demand_service",
    )


def dagster_asset_keys(context: dg.AssetExecutionContext) -> tuple[str, ...]:
    """Return every selected key without assuming a single-output asset."""

    try:
        return (context.asset_key.to_user_string(),)
    except Exception:
        keys_by_output = context.assets_def.keys_by_output_name
        return tuple(key.to_user_string() for _, key in sorted(keys_by_output.items()))


def dagster_lifecycle(context: dg.AssetExecutionContext) -> Lifecycle:
    """Resolve bike release lifecycle once from the native asset identity."""

    asset_keys = dagster_asset_keys(context)
    if asset_keys == ("bike_demand_release_cycle_request",):
        return start_lifecycle(context)
    if asset_keys == ("bike_demand_release_cycle",):
        return bootstrap_lifecycle(context)
    return release_cycle_lifecycle(context)


def dagster_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Resolve the optional MLflow parent from the native asset identity."""

    asset_keys = dagster_asset_keys(context)
    if asset_keys == ("bike_demand_release_cycle_request",):
        return start_mlflow_parent_profiles(context)
    if asset_keys == ("bike_demand_release_cycle",):
        return bootstrap_mlflow_parent_profiles(context)
    return release_cycle_mlflow_parent_profiles(context)


def asset_out(key: str) -> dg.AssetOut:
    """Give every Artifact-producing output the durable handle I/O manager."""

    return dg.AssetOut(key=key, io_manager_key=IO_MANAGER_KEY)


def resource_definitions() -> dict[str, object]:
    """Construct OCLP and Artifact-handle resources for the code location."""

    environment = DemoEnvironment.default()
    return {
        "oclp": oclp_dagster_resource(
            publisher=publisher_for,
            source=source_for,
            adapters=(
                MlflowAdapter(
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    parent_profile=MLFLOW_PARENT_PROFILE,
                ),
            ),
            lifecycle=dagster_lifecycle,
            profiles=dagster_profiles,
        ),
        IO_MANAGER_KEY: oclp_artifact_io_manager(
            catalog_path=environment.catalog_path,
            storage_root=environment.data_root / "dagster-artifact-handles",
        ),
    }

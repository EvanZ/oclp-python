"""Dagster UI dogfood for the existing OCLP bike-demand workflow.

The Dagster asset is deliberately a thin orchestration boundary. The imported
``run_bike_training`` workflow and every child Computation retain their OCLP
decorators; this module only supplies the local storage/source bootstrap and
adds resulting OCLP references to Dagster's materialization metadata.
"""

from __future__ import annotations

import dagster as dg
from oclp.dagster import dagster_asset
from oclp.publishing import LocalArtifactPublisher
from oclp.sources import source_from_git_checkout

from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.runner import run_bike_training


def _materialization_id(context: dg.AssetExecutionContext) -> str:
    """Derive one local immutable-payload namespace from the Dagster run."""

    return f"dagster-{context.run.run_id}"


def _publisher_for(context: dg.AssetExecutionContext) -> LocalArtifactPublisher:
    """Create the local OCLP publisher used by this one Dagster asset run."""

    environment = DemoEnvironment.default()
    environment.prepare()
    return LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.materialization_root(_materialization_id(context)),
    )


def _source_for(_context: dg.AssetExecutionContext):
    """Use the real bike-demand checkout as the implementation source basis."""

    environment = DemoEnvironment.default()
    return source_from_git_checkout(
        environment.project_root,
        path="src/bike_demand_service",
    )


@dg.asset(
    name="bike_demand_model_release",
    group_name="bike_demand",
    description=(
        "Run the existing OCLP and MLflow-instrumented bike-demand training "
        "workflow as one Dagster materialization."
    ),
)
@dagster_asset(
    workflow=run_bike_training,
    publisher=_publisher_for,
    source=_source_for,
)
def bike_demand_model_release(context: dg.AssetExecutionContext) -> None:
    """Materialize the existing release-producing OCLP workflow."""

    run_bike_training(
        materialization_id=_materialization_id(context),
        fold_count=3,
        temporal_validation_rmse_max=250,
    )


bike_demand_job = dg.define_asset_job(
    name="bike_demand_training",
    selection=dg.AssetSelection.assets(bike_demand_model_release),
    description="Materialize the bike-demand OCLP release through Dagster.",
    executor_def=dg.in_process_executor,
)

defs = dg.Definitions(
    assets=[bike_demand_model_release],
    jobs=[bike_demand_job],
)

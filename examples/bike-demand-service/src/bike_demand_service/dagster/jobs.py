"""Dagster jobs selecting the release-cycle stages."""

from __future__ import annotations

import dagster as dg

from bike_demand_service.dagster.partitions import fold_partitions, training_cycles

PREPARATION_ASSETS = (
    "bike_demand_release_cycle",
    "bike_demand_training_plan",
    "bike_demand_raw_source",
    "bike_demand_features",
    "bike_demand_fold_definition",
    "bike_demand_feature_contract",
    "bike_demand_data_metrics",
)
FOLD_ASSETS = (
    "bike_demand_fold_model",
    "bike_demand_fold_predictions",
    "bike_demand_fold_metrics",
)
AGGREGATE_ASSETS = (
    "bike_demand_candidate_evaluation",
    "bike_demand_training_config",
    "bike_demand_temporal_validation_chart",
    "bike_demand_final_model",
    "bike_demand_holdout_predictions",
    "bike_demand_holdout_metrics",
    "bike_demand_holdout_forecast_chart",
    "bike_demand_model_release",
)


bike_demand_start_release_cycle_job = dg.define_asset_job(
    name="bike_demand_start_release_cycle",
    selection=dg.AssetSelection.assets("bike_demand_release_cycle_request"),
    description=(
        "Create a release-cycle request; its sensor adds the UUID partition "
        "and launches preparation."
    ),
    executor_def=dg.in_process_executor,
)
bike_demand_prepare_job = dg.define_asset_job(
    name="bike_demand_prepare_cycle",
    selection=dg.AssetSelection.assets(*PREPARATION_ASSETS),
    partitions_def=training_cycles,
    description="Prepare one configured training cycle and persist its fold plan.",
    executor_def=dg.in_process_executor,
)
bike_demand_fold_job = dg.define_asset_job(
    name="bike_demand_train_fold_job",
    selection=dg.AssetSelection.assets(*FOLD_ASSETS),
    partitions_def=fold_partitions,
    description="Train one dynamically planned temporal fold for one cycle.",
    # DuckDB catalog operations are protected by the SDK's inter-process lock;
    # the sensor can therefore submit independent fold runs concurrently.
    executor_def=dg.in_process_executor,
)
bike_demand_aggregate_job = dg.define_asset_job(
    name="bike_demand_aggregate_cycle",
    selection=dg.AssetSelection.assets(*AGGREGATE_ASSETS),
    partitions_def=training_cycles,
    description="Fan in planned folds, evaluate, train, score, chart, and release.",
    executor_def=dg.in_process_executor,
)


ALL_JOBS = [
    bike_demand_start_release_cycle_job,
    bike_demand_prepare_job,
    bike_demand_fold_job,
    bike_demand_aggregate_job,
]

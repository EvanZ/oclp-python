"""Sensors that advance a bike-demand release cycle between Dagster jobs."""

from __future__ import annotations

import json

import dagster as dg
from oclp import ArtifactHandle, finish_mlflow_parent_run
from oclp.dagster import load_artifact_handle

from bike_demand_service.dagster.jobs import (
    bike_demand_aggregate_job,
    bike_demand_fold_job,
    bike_demand_prepare_job,
)
from bike_demand_service.dagster.lifecycle import (
    artifact_id,
    mlflow_tracking_uri,
    release_cycle_payload,
)
from bike_demand_service.dagster.partitions import (
    fold_partitions,
    temporal_folds,
    training_cycles,
)
from bike_demand_service.environment import DemoEnvironment


def planned_fold_keys(fold_definition: ArtifactHandle) -> tuple[str, ...]:
    """Turn the persisted plan into safe dynamic Dagster fold partition keys."""

    try:
        payload = json.loads(fold_definition.read_verified_bytes())
        folds = payload["folds"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            "fold-definition Artifact does not contain a folds list"
        ) from error
    if not isinstance(folds, list) or not folds:
        raise ValueError("fold-definition Artifact must contain at least one fold")
    keys: list[str] = []
    for fold in folds:
        if not isinstance(fold, dict) or not isinstance(fold.get("fold"), int):
            raise ValueError("fold-definition entries must include integer fold values")
        keys.append(f"fold-{fold['fold']}")
    if len(keys) != len(set(keys)):
        raise ValueError("fold-definition Artifact contains duplicate fold values")
    return tuple(sorted(keys, key=lambda key: int(key.removeprefix("fold-"))))


def event_partition(event: dg.EventLogEntry) -> str:
    """Require the partition key that identifies an observed asset event."""

    dagster_event = event.dagster_event
    partition = None if dagster_event is None else dagster_event.partition
    if not isinstance(partition, str) or not partition:
        raise ValueError("bike-demand orchestration sensors require partitioned events")
    return partition


def release_cycle_request(event: dg.EventLogEntry) -> dict[str, object]:
    """Resolve and validate the exact unpartitioned cycle-start Artifact."""

    environment = DemoEnvironment.default()
    handle = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=artifact_id(event, "oclp.artifact.id"),
    )
    try:
        payload = json.loads(handle.read_verified_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("release-cycle request Artifact must contain JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("release-cycle request Artifact must contain a JSON object")
    cycle_id = payload.get("release_cycle_id")
    parent_run_id = payload.get("mlflow_parent_run_id")
    fold_count = payload.get("fold_count")
    threshold = payload.get("temporal_validation_rmse_max")
    if not isinstance(cycle_id, str) or not cycle_id:
        raise ValueError("release-cycle request lacks release_cycle_id")
    if not isinstance(parent_run_id, str) or not parent_run_id:
        raise ValueError("release-cycle request lacks mlflow_parent_run_id")
    if not isinstance(fold_count, int) or fold_count <= 0:
        raise ValueError("release-cycle request fold_count must be positive")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            "release-cycle request temporal_validation_rmse_max must be numeric"
        )
    return payload


def latest_partition_event(
    context: dg.SensorEvaluationContext,
    *,
    asset_key: str,
    partition_key: str,
) -> dg.EventLogEntry | None:
    """Get the newest materialization event for one exact asset partition."""

    records = context.instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=dg.AssetKey(asset_key),
            asset_partitions=[partition_key],
        ),
        limit=1,
    ).records
    return records[0].event_log_entry if records else None


def cycle_fold_definition(
    context: dg.SensorEvaluationContext,
    cycle_id: str,
) -> ArtifactHandle:
    """Resolve this cycle's exact immutable fold-definition Artifact."""

    event = latest_partition_event(
        context,
        asset_key="bike_demand_fold_definition",
        partition_key=cycle_id,
    )
    if event is None:
        raise ValueError(f"training cycle {cycle_id!r} has no fold-definition asset")
    environment = DemoEnvironment.default()
    return load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=artifact_id(event, "oclp.output.fold_definition.id"),
    )


@dg.asset_sensor(
    asset_key=dg.AssetKey("bike_demand_release_cycle_request"),
    job=bike_demand_prepare_job,
    name="launch_bike_demand_preparation",
    description="Create the requested dynamic cycle partition and launch preparation.",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def launch_bike_demand_preparation(
    _context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.SensorResult:
    """Turn one unpartitioned start request into a configured cycle run."""

    request = release_cycle_request(asset_event)
    cycle_id = str(request["release_cycle_id"])
    return dg.SensorResult(
        dynamic_partitions_requests=[
            training_cycles.build_add_request([cycle_id]),
        ],
        run_requests=[
            dg.RunRequest(
                run_key=f"bike-demand-prepare:{cycle_id}",
                partition_key=cycle_id,
                run_config={
                    "ops": {
                        "bike_demand_release_cycle": {
                            "config": {
                                "fold_count": int(request["fold_count"]),
                                "temporal_validation_rmse_max": float(
                                    request["temporal_validation_rmse_max"]
                                ),
                            }
                        }
                    }
                },
            )
        ],
    )


@dg.asset_sensor(
    asset_key=dg.AssetKey("bike_demand_fold_definition"),
    job=bike_demand_fold_job,
    name="launch_bike_demand_fold_partitions",
    description="Create and materialize exactly the temporal folds in each plan.",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def launch_bike_demand_fold_partitions(
    _context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.SensorResult:
    """Fan out a prepared cycle into one run per persisted temporal fold."""

    cycle_id = event_partition(asset_event)
    environment = DemoEnvironment.default()
    definition = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=artifact_id(asset_event, "oclp.output.fold_definition.id"),
    )
    fold_keys = planned_fold_keys(definition)
    return dg.SensorResult(
        dynamic_partitions_requests=[
            temporal_folds.build_add_request(list(fold_keys)),
        ],
        run_requests=[
            dg.RunRequest(
                run_key=f"bike-demand-fold:{cycle_id}:{fold_key}",
                partition_key=dg.MultiPartitionKey(
                    {"cycle": cycle_id, "fold": fold_key}
                ),
            )
            for fold_key in fold_keys
        ],
    )


@dg.asset_sensor(
    asset_key=dg.AssetKey("bike_demand_fold_predictions"),
    job=bike_demand_aggregate_job,
    name="launch_bike_demand_cycle_fan_in",
    description="Launch evaluation and release only after all planned folds complete.",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def launch_bike_demand_cycle_fan_in(
    context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.RunRequest | dg.SkipReason:
    """Launch exactly one fan-in run after every cycle-fold prediction exists."""

    multi_partition = fold_partitions.get_partition_key_from_str(
        event_partition(asset_event)
    )
    cycle_id = multi_partition.keys_by_dimension["cycle"]
    definition = cycle_fold_definition(context, cycle_id)
    expected_partitions = tuple(
        str(dg.MultiPartitionKey({"cycle": cycle_id, "fold": fold_key}))
        for fold_key in planned_fold_keys(definition)
    )
    records = context.instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=dg.AssetKey("bike_demand_fold_predictions"),
            asset_partitions=expected_partitions,
        ),
        limit=len(expected_partitions),
    ).records
    completed_partitions = {
        event_partition(record.event_log_entry) for record in records
    }
    missing = sorted(set(expected_partitions).difference(completed_partitions))
    if missing:
        return dg.SkipReason(
            "waiting for planned temporal-fold predictions: " + ", ".join(missing)
        )
    threshold = release_cycle_payload(context, cycle_id).get(
        "temporal_validation_rmse_max"
    )
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            "release-cycle Artifact temporal_validation_rmse_max must be numeric"
        )
    return dg.RunRequest(
        run_key=f"bike-demand-aggregate:{cycle_id}",
        partition_key=cycle_id,
        run_config={
            "ops": {
                "bike_demand_evaluate_candidate": {
                    "config": {
                        "temporal_validation_rmse_max": float(threshold),
                    }
                },
                "bike_demand_temporal_validation_chart": {
                    "config": {
                        "temporal_validation_rmse_max": float(threshold),
                    }
                },
            }
        },
    )


@dg.asset_sensor(
    asset_key=dg.AssetKey("bike_demand_model_release"),
    name="finish_bike_demand_release_cycle_parent",
    description="Mark the cycle's persistent MLflow parent finished after release.",
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def finish_bike_demand_release_cycle_parent(
    context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.SkipReason:
    """Close the parent only after the immutable release ArtifactSet exists."""

    cycle_id = event_partition(asset_event)
    cycle = release_cycle_payload(context, cycle_id)
    parent_run_id = str(cycle["mlflow_parent_run_id"])
    environment = DemoEnvironment.default()
    finish_mlflow_parent_run(
        run_id=parent_run_id,
        tracking_uri=mlflow_tracking_uri(environment),
    )
    return dg.SkipReason(
        f"marked MLflow parent {parent_run_id} finished for cycle {cycle_id}"
    )


ALL_SENSORS = [
    launch_bike_demand_preparation,
    launch_bike_demand_fold_partitions,
    launch_bike_demand_cycle_fan_in,
    finish_bike_demand_release_cycle_parent,
]

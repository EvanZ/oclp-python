"""Partitioned Dagster orchestration for the complete bike-demand pipeline.

Every ordinary data and model boundary remains a visible Dagster asset. A
training cycle is a dynamic partition. Its preparation step persists the
actual temporal-fold plan, an asset sensor adds exactly those fold partitions,
and one multi-partitioned fold asset trains each requested fold. A second
sensor starts the fan-in assets only after every planned fold prediction is
materialized. This is deliberately a multi-run graph: OCLP record identities,
not live Python values, travel between Dagster workers.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import dagster as dg
from oclp import (
    ArtifactHandle,
    create_mlflow_parent_run,
    finish_mlflow_parent_run,
)
from oclp.dagster import (
    dg_artifact,
    dg_artifact_set,
    dg_computation,
    load_artifact_handle,
    oclp_artifact_io_manager,
)
from oclp.publishing import LocalArtifactPublisher
from oclp.sources import source_from_git_checkout

from bike_demand_service.data import download_source_csv, prepare_features
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.modeling import (
    chart_holdout_demand_forecast,
    chart_temporal_validation_quality,
    create_release_cycle,
    create_training_plan,
    evaluate_folds,
    request_release_cycle,
    score_holdout,
    train_final_model,
    train_fold,
)
from bike_demand_service.release_cycle import (
    MLFLOW_EXPERIMENT_NAME,
    release_cycle_profiles,
)
from bike_demand_service.runner import (
    run_bike_demand_aggregate_cycle,
    run_bike_demand_prepare_cycle,
    run_bike_demand_release_cycle_start,
    run_bike_demand_temporal_fold,
)

_TEMPORAL_VALIDATION_RMSE_MAX = 250
_IO_MANAGER_KEY = "oclp_artifact_io_manager"


def new_release_cycle_id() -> str:
    """Create the application-owned UUID used as one Dagster cycle partition."""

    return str(uuid4())


def _release_cycle_id_for_start(context: dg.AssetExecutionContext) -> str:
    """Derive one repeatable UUID while a Dagster start-job retry is in flight."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"bike-demand-release-cycle-start:{context.run.run_id}",
        )
    )

# A cycle key is selected before the preparation job runs. It is deliberately
# opaque: the persisted training-plan Artifact, rather than the key format,
# records the chosen training inputs. Fold keys become dynamic only after the
# plan's fold-definition Artifact exists.
training_cycles = dg.DynamicPartitionsDefinition(name="bike_demand_training_cycle")
temporal_folds = dg.DynamicPartitionsDefinition(name="bike_demand_temporal_fold")
fold_partitions = dg.MultiPartitionsDefinition(
    {
        "cycle": training_cycles,
        "fold": temporal_folds,
    }
)
_CYCLE_TO_FOLD = dg.MultiToSingleDimensionPartitionMapping("cycle")


def _training_cycle_id(context: dg.AssetExecutionContext) -> str:
    """Read the cycle component from either kind of pipeline partition."""

    if not context.has_partition_key:
        raise ValueError(
            "bike-demand Dagster assets require a training-cycle partition"
        )
    partition_key = context.partition_key
    if isinstance(partition_key, dg.MultiPartitionKey):
        return partition_key.keys_by_dimension["cycle"]
    return str(partition_key)


def _materialization_id(context: dg.AssetExecutionContext) -> str:
    """Use the durable training-cycle key, not one transient Dagster run ID."""

    if not context.has_partition_key:
        return f"dagster-release-start-{context.run.run_id}"
    return f"dagster-cycle-{_training_cycle_id(context)}"


def _publisher_for(context: dg.AssetExecutionContext) -> LocalArtifactPublisher:
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
                    _materialization_id(context),
                    "assets",
                    context.get_step_execution_context().step.key,
                    f"partition-{partition}",
                    f"attempt-{context.retry_number}",
                )
            )
        ),
    )


def _source_for(_context: dg.AssetExecutionContext):
    """Use the real bike-demand checkout as the implementation source basis."""

    environment = DemoEnvironment.default()
    return source_from_git_checkout(
        environment.project_root,
        path="src/bike_demand_service",
    )


def _mlflow_tracking_uri(environment: DemoEnvironment) -> str:
    """Return the project-local MLflow store shared by every Dagster worker."""

    return "sqlite:///" + (environment.mlflow_root / "mlflow.db").resolve().as_posix()


def _release_cycle_parent_run_id(release_cycle_id: str) -> str:
    """Create one retry-safe MLflow parent for an application release cycle."""

    environment = DemoEnvironment.default()
    environment.prepare()
    return create_mlflow_parent_run(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=f"bike-demand release cycle {release_cycle_id}",
        tracking_uri=_mlflow_tracking_uri(environment),
        artifact_location=(environment.mlflow_root / "artifacts").resolve().as_uri(),
        identity_tags={
            "oclp.profile.bike_demand.release_cycle_id": release_cycle_id,
        },
        tags={
            "bike_demand.mlflow_role": "release-cycle-parent",
        },
    )


def _profiles_for_release_cycle(
    release_cycle_id: str,
) -> dict[str, dict[str, object]]:
    """Resolve the profile and persistent parent for one cycle UUID."""

    return release_cycle_profiles(
        release_cycle_id=release_cycle_id,
        mlflow_parent_run_id=_release_cycle_parent_run_id(release_cycle_id),
    )


def _start_release_cycle_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Provide a stable UUID and parent before the start Artifact is observed."""

    return _profiles_for_release_cycle(_release_cycle_id_for_start(context))


def _bootstrap_release_cycle_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Build cycle facts before the bootstrap Artifact opens its OCLP run."""

    return _profiles_for_release_cycle(_training_cycle_id(context))


def _latest_cycle_asset_event(
    instance: Any,
    *,
    asset_key: str,
    cycle_id: str,
) -> dg.EventLogEntry | None:
    """Return the newest materialization for one single-dimensional cycle asset."""

    records = instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=dg.AssetKey(asset_key),
            asset_partitions=[cycle_id],
        ),
        limit=1,
    ).records
    return records[0].event_log_entry if records else None


def _release_cycle_payload(
    context: dg.AssetExecutionContext | dg.SensorEvaluationContext,
    cycle_id: str | None = None,
) -> dict[str, object]:
    """Load the exact cycle Artifact rather than reconstructing cycle settings."""

    resolved_cycle_id = cycle_id or _training_cycle_id(context)  # type: ignore[arg-type]
    event = _latest_cycle_asset_event(
        context.instance,
        asset_key="bike_demand_release_cycle",
        cycle_id=resolved_cycle_id,
    )
    if event is None:
        raise ValueError(f"release cycle {resolved_cycle_id!r} has not materialized")
    environment = DemoEnvironment.default()
    handle = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=_artifact_id(event, "oclp.artifact.id"),
    )
    try:
        payload = json.loads(handle.read_verified_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("release-cycle Artifact must contain a JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError("release-cycle Artifact must contain a JSON object")
    if payload.get("release_cycle_id") != resolved_cycle_id:
        raise ValueError("release-cycle Artifact does not match its partition key")
    parent_run_id = payload.get("mlflow_parent_run_id")
    if not isinstance(parent_run_id, str) or not parent_run_id:
        raise ValueError("release-cycle Artifact lacks mlflow_parent_run_id")
    return payload


def _release_cycle_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Use the bootstrap Artifact as the sole source of child-run context."""

    payload = _release_cycle_payload(context)
    return release_cycle_profiles(
        release_cycle_id=str(payload["release_cycle_id"]),
        mlflow_parent_run_id=str(payload["mlflow_parent_run_id"]),
    )


def _asset_out(key: str) -> dg.AssetOut:
    """Give every Artifact-producing output the durable handle I/O manager."""

    return dg.AssetOut(key=key, io_manager_key=_IO_MANAGER_KEY)


def _temporal_fold_number(context: dg.AssetExecutionContext) -> int:
    """Resolve the persisted temporal-fold number from one Dagster partition."""

    partition_key = context.partition_key
    if not isinstance(partition_key, dg.MultiPartitionKey):
        raise ValueError("bike_demand_train_fold requires a multi-partition key")
    fold_key = partition_key.keys_by_dimension["fold"]
    if not fold_key.startswith("fold-"):
        raise ValueError(f"unexpected temporal-fold partition key {fold_key!r}")
    try:
        return int(fold_key.removeprefix("fold-"))
    except ValueError as error:
        raise ValueError(
            f"unexpected temporal-fold partition key {fold_key!r}"
        ) from error


def _temporal_fold_run_name(context: dg.AssetExecutionContext) -> str:
    """Name one dynamic fold with its application-owned fold identity."""

    return f"Bike demand temporal validation fold {_temporal_fold_number(context)}"


@dg_artifact(
    workflow=run_bike_demand_release_cycle_start,
    publisher=_publisher_for,
    source=_source_for,
    asset_key="bike_demand_release_cycle_request",
    group_name="bike_demand",
    io_manager_key=_IO_MANAGER_KEY,
    config_schema={
        "fold_count": dg.Field(int, default_value=3, is_required=False),
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=_TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    application_profiles=_start_release_cycle_profiles,
)
def bike_demand_release_cycle_request(
    context: dg.AssetExecutionContext,
) -> ArtifactHandle:
    """Start a cycle without first selecting a dynamic Dagster partition."""

    profile = _start_release_cycle_profiles(context)["bike_demand"]
    return request_release_cycle(
        release_cycle_id=str(profile["release_cycle_id"]),
        mlflow_parent_run_id=str(profile["mlflow_parent_run_id"]),
        fold_count=int(context.op_config["fold_count"]),
        temporal_validation_rmse_max=float(
            context.op_config["temporal_validation_rmse_max"]
        ),
    )


@dg_artifact(
    workflow=run_bike_demand_prepare_cycle,
    publisher=_publisher_for,
    source=_source_for,
    asset_key="bike_demand_release_cycle",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    config_schema={
        "fold_count": dg.Field(int, default_value=3, is_required=False),
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=_TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    application_profiles=_bootstrap_release_cycle_profiles,
)
def bike_demand_release_cycle(context: dg.AssetExecutionContext) -> ArtifactHandle:
    """Persist one application-owned release cycle and its MLflow parent run."""

    profiles = _bootstrap_release_cycle_profiles(context)
    cycle = profiles["bike_demand"]
    return create_release_cycle(
        release_cycle_id=str(cycle["release_cycle_id"]),
        mlflow_parent_run_id=str(cycle["mlflow_parent_run_id"]),
        fold_count=int(context.op_config["fold_count"]),
        temporal_validation_rmse_max=float(
            context.op_config["temporal_validation_rmse_max"]
        ),
    )


@dg_artifact(
    workflow=run_bike_demand_prepare_cycle,
    publisher=_publisher_for,
    source=_source_for,
    asset_key="bike_demand_training_plan",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    deps=(dg.AssetKey("bike_demand_release_cycle"),),
    application_profiles=_release_cycle_profiles,
)
def bike_demand_training_plan(context: dg.AssetExecutionContext) -> ArtifactHandle:
    """Persist the selected fold count as the cycle's immutable plan."""

    cycle = _release_cycle_payload(context)
    fold_count = cycle.get("fold_count")
    if not isinstance(fold_count, int):
        raise ValueError("release-cycle Artifact fold_count must be an integer")
    return create_training_plan(
        materialization_id=_materialization_id(context),
        fold_count=fold_count,
    )


@dg_artifact(
    workflow=run_bike_demand_prepare_cycle,
    publisher=_publisher_for,
    source=_source_for,
    asset_key="bike_demand_raw_source",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    deps=(dg.AssetKey("bike_demand_release_cycle"),),
    application_profiles=_release_cycle_profiles,
)
def bike_demand_raw_source() -> ArtifactHandle:
    """Acquire the UCI Bike Sharing source as a transparent graph input."""

    return download_source_csv()


@dg_computation(
    workflow=run_bike_demand_prepare_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=prepare_features,
    outputs={
        "features": _asset_out("bike_demand_features"),
        "fold_definition": _asset_out("bike_demand_fold_definition"),
        "feature_contract": _asset_out("bike_demand_feature_contract"),
        "data_metrics": _asset_out("bike_demand_data_metrics"),
    },
    inputs={
        "source_snapshot": dg.AssetIn(key=dg.AssetKey("bike_demand_raw_source")),
        "training_plan": dg.AssetIn(key=dg.AssetKey("bike_demand_training_plan")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    application_profiles=_release_cycle_profiles,
)
def bike_demand_prepare_features(
    source_snapshot: ArtifactHandle,
    training_plan: ArtifactHandle,
) -> object:
    """Prepare the feature data and the durable temporal-fold definition."""

    return prepare_features(source_snapshot, training_plan)


@dg_computation(
    workflow=run_bike_demand_temporal_fold,
    publisher=_publisher_for,
    source=_source_for,
    target=train_fold,
    outputs={
        "model": _asset_out("bike_demand_fold_model"),
        "validation_predictions": _asset_out("bike_demand_fold_predictions"),
        "metrics": _asset_out("bike_demand_fold_metrics"),
    },
    inputs={
        "feature_table": dg.AssetIn(
            key=dg.AssetKey("bike_demand_features"),
            partition_mapping=_CYCLE_TO_FOLD,
        ),
        "fold_definition": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_definition"),
            partition_mapping=_CYCLE_TO_FOLD,
        ),
    },
    group_name="bike_demand",
    partitions_def=fold_partitions,
    context_parameter="context",
    application_profiles=_release_cycle_profiles,
    run_name=_temporal_fold_run_name,
)
def bike_demand_train_fold(
    feature_table: ArtifactHandle,
    fold_definition: ArtifactHandle,
    context: dg.AssetExecutionContext,
) -> object:
    """Train exactly the temporal fold named by this Dagster partition."""

    return train_fold(
        feature_table,
        fold_definition,
        fold_number=_temporal_fold_number(context),
    )


@dg_computation(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=evaluate_folds,
    outputs={
        "evaluation": _asset_out("bike_demand_candidate_evaluation"),
        "training_config": _asset_out("bike_demand_training_config"),
    },
    inputs={
        "fold_predictions": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_predictions"),
            partition_mapping=_CYCLE_TO_FOLD,
        ),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    context_parameter="context",
    application_profiles=_release_cycle_profiles,
)
def bike_demand_evaluate_candidate(
    fold_predictions: tuple[ArtifactHandle, ...],
    context: dg.AssetExecutionContext,
) -> object:
    """Aggregate all planned temporal-fold predictions for this one cycle."""

    cycle = _release_cycle_payload(context)
    threshold = cycle.get("temporal_validation_rmse_max")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            "release-cycle Artifact temporal_validation_rmse_max must be numeric"
        )
    return evaluate_folds(
        fold_predictions,
        temporal_validation_rmse_max=float(threshold),
    )


@dg_computation(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=chart_temporal_validation_quality,
    asset_key="bike_demand_temporal_validation_chart",
    inputs={
        "fold_predictions": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_predictions"),
            partition_mapping=_CYCLE_TO_FOLD,
        ),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    context_parameter="context",
    application_profiles=_release_cycle_profiles,
)
def bike_demand_chart_temporal_validation(
    fold_predictions: tuple[ArtifactHandle, ...],
    context: dg.AssetExecutionContext,
) -> object:
    """Render the temporal-validation chart from all visible fold partitions."""

    threshold = _release_cycle_payload(context).get("temporal_validation_rmse_max")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError(
            "release-cycle Artifact temporal_validation_rmse_max must be numeric"
        )
    return chart_temporal_validation_quality(
        fold_predictions,
        temporal_validation_rmse_max=float(threshold),
    )


@dg_computation(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=train_final_model,
    asset_key="bike_demand_final_model",
    inputs={
        "feature_table": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
        "training_config": dg.AssetIn(key=dg.AssetKey("bike_demand_training_config")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    application_profiles=_release_cycle_profiles,
)
def bike_demand_train_final_model(
    feature_table: ArtifactHandle,
    training_config: ArtifactHandle,
) -> object:
    """Train the final model only after the fan-in evaluation is available."""

    return train_final_model(
        feature_table,
        training_config,
        training_window="all-pre-holdout-rows",
    )


@dg_computation(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=score_holdout,
    outputs={
        "predictions": _asset_out("bike_demand_holdout_predictions"),
        "metrics": _asset_out("bike_demand_holdout_metrics"),
    },
    inputs={
        "model": dg.AssetIn(key=dg.AssetKey("bike_demand_final_model")),
        "feature_table": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    application_profiles=_release_cycle_profiles,
)
def bike_demand_score_holdout(
    model: ArtifactHandle,
    feature_table: ArtifactHandle,
) -> object:
    """Score the selected model against the transparent held-out rows."""

    return score_holdout(model, feature_table)


@dg_computation(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    target=chart_holdout_demand_forecast,
    asset_key="bike_demand_holdout_forecast_chart",
    inputs={
        "predictions": dg.AssetIn(key=dg.AssetKey("bike_demand_holdout_predictions")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=_IO_MANAGER_KEY,
    application_profiles=_release_cycle_profiles,
)
def bike_demand_chart_holdout_forecast(predictions: ArtifactHandle) -> object:
    """Render the final holdout forecast from the visible prediction asset."""

    return chart_holdout_demand_forecast(predictions)


@dg_artifact_set(
    workflow=run_bike_demand_aggregate_cycle,
    publisher=_publisher_for,
    source=_source_for,
    asset_key="bike_demand_model_release",
    name="Bike demand CatBoost release",
    inputs={
        "release_cycle": dg.AssetIn(key=dg.AssetKey("bike_demand_release_cycle")),
        "features": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
        "feature_contract": dg.AssetIn(key=dg.AssetKey("bike_demand_feature_contract")),
        "evaluation": dg.AssetIn(key=dg.AssetKey("bike_demand_candidate_evaluation")),
        "training_config": dg.AssetIn(key=dg.AssetKey("bike_demand_training_config")),
        "model": dg.AssetIn(key=dg.AssetKey("bike_demand_final_model")),
    },
    members={
        "release-cycle": ("release_cycle", "release-cycle"),
        "features": ("features", "training-data"),
        "feature-contract": ("feature_contract", "serving-contract"),
        "temporal-evaluation": ("evaluation", "validation-report"),
        "training-config": ("training_config", "training-config"),
        "model": ("model", "model"),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    application_profiles=_release_cycle_profiles,
)
def bike_demand_model_release() -> None:
    """Assemble the release from exact, visible Artifact asset partitions."""


_PREPARATION_ASSETS = (
    "bike_demand_release_cycle",
    "bike_demand_training_plan",
    "bike_demand_raw_source",
    "bike_demand_features",
    "bike_demand_fold_definition",
    "bike_demand_feature_contract",
    "bike_demand_data_metrics",
)
_FOLD_ASSETS = (
    "bike_demand_fold_model",
    "bike_demand_fold_predictions",
    "bike_demand_fold_metrics",
)
_AGGREGATE_ASSETS = (
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
    selection=dg.AssetSelection.assets(*_PREPARATION_ASSETS),
    partitions_def=training_cycles,
    description="Prepare one configured training cycle and persist its fold plan.",
    executor_def=dg.in_process_executor,
)
bike_demand_fold_job = dg.define_asset_job(
    name="bike_demand_train_fold_job",
    selection=dg.AssetSelection.assets(*_FOLD_ASSETS),
    partitions_def=fold_partitions,
    description="Train one dynamically planned temporal fold for one cycle.",
    # DuckDB catalog operations are protected by the SDK's inter-process lock;
    # the sensor can therefore submit independent fold runs concurrently.
    executor_def=dg.in_process_executor,
)
bike_demand_aggregate_job = dg.define_asset_job(
    name="bike_demand_aggregate_cycle",
    selection=dg.AssetSelection.assets(*_AGGREGATE_ASSETS),
    partitions_def=training_cycles,
    description="Fan in planned folds, evaluate, train, score, chart, and release.",
    executor_def=dg.in_process_executor,
)


def _materialization_metadata(event: dg.EventLogEntry) -> dict[str, object]:
    """Return one sensor event's Dagster materialization metadata."""

    dagster_event = event.dagster_event
    if dagster_event is None or dagster_event.event_specific_data is None:
        raise ValueError("asset sensor received an event without materialization data")
    materialization = dagster_event.event_specific_data.materialization
    return dict(materialization.metadata)


def _artifact_id(event: dg.EventLogEntry, metadata_key: str) -> str:
    """Read one OCLP Artifact reference written by a projection decorator."""

    value = _materialization_metadata(event).get(metadata_key)
    artifact_id = getattr(value, "value", value)
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(
            f"materialization metadata must contain non-empty {metadata_key!r}"
        )
    return artifact_id


def _planned_fold_keys(fold_definition: ArtifactHandle) -> tuple[str, ...]:
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


def _event_partition(event: dg.EventLogEntry) -> str:
    """Require the partition key that identifies an observed asset event."""

    dagster_event = event.dagster_event
    partition = None if dagster_event is None else dagster_event.partition
    if not isinstance(partition, str) or not partition:
        raise ValueError("bike-demand orchestration sensors require partitioned events")
    return partition


def _release_cycle_request(event: dg.EventLogEntry) -> dict[str, object]:
    """Resolve and validate the exact unpartitioned cycle-start Artifact."""

    environment = DemoEnvironment.default()
    handle = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=_artifact_id(event, "oclp.artifact.id"),
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


def _latest_partition_event(
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


def _cycle_fold_definition(
    context: dg.SensorEvaluationContext,
    cycle_id: str,
) -> ArtifactHandle:
    """Resolve this cycle's exact immutable fold-definition Artifact."""

    event = _latest_partition_event(
        context,
        asset_key="bike_demand_fold_definition",
        partition_key=cycle_id,
    )
    if event is None:
        raise ValueError(f"training cycle {cycle_id!r} has no fold-definition asset")
    environment = DemoEnvironment.default()
    return load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=_artifact_id(event, "oclp.output.fold_definition.id"),
    )


@dg.asset_sensor(
    asset_key=dg.AssetKey("bike_demand_release_cycle_request"),
    job=bike_demand_prepare_job,
    name="launch_bike_demand_preparation",
    description=(
        "Create the requested dynamic cycle partition and launch preparation."
    ),
    default_status=dg.DefaultSensorStatus.RUNNING,
)
def launch_bike_demand_preparation(
    _context: dg.SensorEvaluationContext,
    asset_event: dg.EventLogEntry,
) -> dg.SensorResult:
    """Turn one unpartitioned start request into a configured cycle run."""

    request = _release_cycle_request(asset_event)
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

    cycle_id = _event_partition(asset_event)
    environment = DemoEnvironment.default()
    definition = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=_artifact_id(asset_event, "oclp.output.fold_definition.id"),
    )
    fold_keys = _planned_fold_keys(definition)
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
        _event_partition(asset_event)
    )
    cycle_id = multi_partition.keys_by_dimension["cycle"]
    definition = _cycle_fold_definition(context, cycle_id)
    expected_partitions = tuple(
        str(dg.MultiPartitionKey({"cycle": cycle_id, "fold": fold_key}))
        for fold_key in _planned_fold_keys(definition)
    )
    records = context.instance.fetch_materializations(
        dg.AssetRecordsFilter(
            asset_key=dg.AssetKey("bike_demand_fold_predictions"),
            asset_partitions=expected_partitions,
        ),
        limit=len(expected_partitions),
    ).records
    completed_partitions = {
        _event_partition(record.event_log_entry) for record in records
    }
    missing = sorted(set(expected_partitions).difference(completed_partitions))
    if missing:
        return dg.SkipReason(
            "waiting for planned temporal-fold predictions: " + ", ".join(missing)
        )
    return dg.RunRequest(
        run_key=f"bike-demand-aggregate:{cycle_id}",
        partition_key=cycle_id,
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

    cycle_id = _event_partition(asset_event)
    cycle = _release_cycle_payload(context, cycle_id)
    parent_run_id = str(cycle["mlflow_parent_run_id"])
    environment = DemoEnvironment.default()
    finish_mlflow_parent_run(
        run_id=parent_run_id,
        tracking_uri=_mlflow_tracking_uri(environment),
    )
    return dg.SkipReason(
        f"marked MLflow parent {parent_run_id} finished for cycle {cycle_id}"
    )


_environment = DemoEnvironment.default()
defs = dg.Definitions(
    assets=[
        bike_demand_release_cycle_request,
        bike_demand_release_cycle,
        bike_demand_training_plan,
        bike_demand_raw_source,
        bike_demand_prepare_features,
        bike_demand_train_fold,
        bike_demand_evaluate_candidate,
        bike_demand_chart_temporal_validation,
        bike_demand_train_final_model,
        bike_demand_score_holdout,
        bike_demand_chart_holdout_forecast,
        bike_demand_model_release,
    ],
    jobs=[
        bike_demand_start_release_cycle_job,
        bike_demand_prepare_job,
        bike_demand_fold_job,
        bike_demand_aggregate_job,
    ],
    sensors=[
        launch_bike_demand_preparation,
        launch_bike_demand_fold_partitions,
        launch_bike_demand_cycle_fan_in,
        finish_bike_demand_release_cycle_parent,
    ],
    resources={
        _IO_MANAGER_KEY: oclp_artifact_io_manager(
            catalog_path=_environment.catalog_path,
            storage_root=_environment.data_root / "dagster-artifact-handles",
        ),
    },
)

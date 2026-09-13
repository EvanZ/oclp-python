"""Release-cycle identity, lifecycle, and MLflow-parent coordination."""

from __future__ import annotations

import json
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import dagster as dg
from oclp import (
    Lifecycle,
    create_mlflow_parent_run,
    lifecycle_from_id,
)
from oclp.dagster import load_artifact_handle

from bike_demand_service.dagster.partitions import training_cycle_id
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.mlflow_parent import (
    MLFLOW_EXPERIMENT_NAME,
    mlflow_parent_profiles,
)


def new_release_cycle_id() -> str:
    """Create the lifecycle UUID used as one Dagster cycle partition."""

    return str(uuid4())


def release_cycle_id_for_start(context: dg.AssetExecutionContext) -> str:
    """Use Dagster's run UUID as the lifecycle ID for one start-job retry set."""

    dagster_run_id = str(context.run.run_id)
    try:
        return str(UUID(dagster_run_id))
    except ValueError:
        # This preserves the lifecycle profile's UUID contract for a compatible
        # scheduler that exposes a non-UUID run identifier.
        return str(
            uuid5(
                NAMESPACE_URL,
                f"bike-demand-release-cycle-start:{dagster_run_id}",
            )
        )


def mlflow_tracking_uri(environment: DemoEnvironment) -> str:
    """Return the project-local MLflow store shared by every Dagster worker."""

    return "sqlite:///" + (environment.mlflow_root / "mlflow.db").resolve().as_posix()


def release_cycle_parent_run_id(release_cycle_id: str) -> str:
    """Create one retry-safe MLflow parent for an application release cycle."""

    environment = DemoEnvironment.default()
    environment.prepare()
    return create_mlflow_parent_run(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        run_name=f"bike-demand release cycle {release_cycle_id[:8]}",
        tracking_uri=mlflow_tracking_uri(environment),
        artifact_location=(environment.mlflow_root / "artifacts").resolve().as_uri(),
        identity_tags={
            "oclp.profile.lifecycle.lifecycle_id": release_cycle_id,
        },
        tags={
            "bike_demand.mlflow_role": "release-cycle-parent",
        },
    )


def mlflow_parent_profiles_for_release_cycle(
    release_cycle_id: str,
) -> dict[str, dict[str, object]]:
    """Resolve the optional MLflow parent for one lifecycle UUID."""

    return mlflow_parent_profiles(
        mlflow_parent_run_id=release_cycle_parent_run_id(release_cycle_id),
    )


def lifecycle_for_release_cycle(release_cycle_id: str) -> Lifecycle:
    """Return the portable lifecycle carried by all records for one cycle."""

    return lifecycle_from_id(release_cycle_id)


def start_lifecycle(context: dg.AssetExecutionContext) -> Lifecycle:
    """Provide a stable lifecycle before the unpartitioned start is observed."""

    return lifecycle_for_release_cycle(release_cycle_id_for_start(context))


def start_mlflow_parent_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Build MLflow parent context before the start Artifact opens its run."""

    return mlflow_parent_profiles_for_release_cycle(release_cycle_id_for_start(context))


def bootstrap_lifecycle(context: dg.AssetExecutionContext) -> Lifecycle:
    """Bind the preparation bootstrap Artifact to its selected lifecycle."""

    return lifecycle_for_release_cycle(training_cycle_id(context))


def bootstrap_mlflow_parent_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Build MLflow parent context before the partitioned bootstrap runs."""

    return mlflow_parent_profiles_for_release_cycle(training_cycle_id(context))


def materialization_metadata(event: dg.EventLogEntry) -> dict[str, object]:
    """Return one sensor event's Dagster materialization metadata."""

    dagster_event = event.dagster_event
    if dagster_event is None or dagster_event.event_specific_data is None:
        raise ValueError("asset sensor received an event without materialization data")
    materialization = dagster_event.event_specific_data.materialization
    return dict(materialization.metadata)


def artifact_id(event: dg.EventLogEntry, metadata_key: str) -> str:
    """Read one OCLP Artifact reference written by a projection decorator."""

    value = materialization_metadata(event).get(metadata_key)
    resolved_artifact_id = getattr(value, "value", value)
    if not isinstance(resolved_artifact_id, str) or not resolved_artifact_id:
        raise ValueError(
            f"materialization metadata must contain non-empty {metadata_key!r}"
        )
    return resolved_artifact_id


def latest_cycle_asset_event(
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


def release_cycle_payload(
    context: dg.AssetExecutionContext | dg.SensorEvaluationContext,
    cycle_id: str | None = None,
) -> dict[str, object]:
    """Load the exact cycle Artifact rather than reconstructing cycle settings."""

    resolved_cycle_id = cycle_id or training_cycle_id(context)  # type: ignore[arg-type]
    event = latest_cycle_asset_event(
        context.instance,
        asset_key="bike_demand_release_cycle",
        cycle_id=resolved_cycle_id,
    )
    if event is None:
        raise ValueError(f"release cycle {resolved_cycle_id!r} has not materialized")
    environment = DemoEnvironment.default()
    handle = load_artifact_handle(
        catalog_path=environment.catalog_path,
        artifact_id=artifact_id(event, "oclp.artifact.id"),
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


def release_cycle_lifecycle(context: dg.AssetExecutionContext) -> Lifecycle:
    """Read portable lifecycle identity from the persisted cycle Artifact."""

    return lifecycle_for_release_cycle(
        str(release_cycle_payload(context)["release_cycle_id"])
    )


def release_cycle_mlflow_parent_profiles(
    context: dg.AssetExecutionContext,
) -> dict[str, dict[str, object]]:
    """Use the bootstrap Artifact as the MLflow child-run context source."""

    payload = release_cycle_payload(context)
    return mlflow_parent_profiles(
        mlflow_parent_run_id=str(payload["mlflow_parent_run_id"]),
    )

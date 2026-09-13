"""Dagster-owned partition definitions for bike-demand release cycles."""

from __future__ import annotations

import dagster as dg

TEMPORAL_VALIDATION_RMSE_MAX = 250
IO_MANAGER_KEY = "oclp_artifact_io_manager"

# A cycle key is selected before preparation runs. It is deliberately opaque:
# the persisted training-plan Artifact, rather than the key format, records the
# chosen training inputs. Fold keys become dynamic only after that plan exists.
training_cycles = dg.DynamicPartitionsDefinition(name="bike_demand_training_cycle")
temporal_folds = dg.DynamicPartitionsDefinition(name="bike_demand_temporal_fold")
fold_partitions = dg.MultiPartitionsDefinition(
    {
        "cycle": training_cycles,
        "fold": temporal_folds,
    }
)
cycle_to_fold = dg.MultiToSingleDimensionPartitionMapping("cycle")


def training_cycle_id(context: dg.AssetExecutionContext) -> str:
    """Read the cycle component from either kind of pipeline partition."""

    if not context.has_partition_key:
        raise ValueError(
            "bike-demand Dagster assets require a training-cycle partition"
        )
    partition_key = context.partition_key
    if isinstance(partition_key, dg.MultiPartitionKey):
        return partition_key.keys_by_dimension["cycle"]
    return str(partition_key)


def temporal_fold_number(context: dg.AssetExecutionContext) -> int:
    """Read the Dagster-owned fold component for this partitioned asset."""

    if not context.has_partition_key:
        raise ValueError("bike-demand fold assets require a multi-partition key")
    partition_key = context.partition_key
    if not isinstance(partition_key, dg.MultiPartitionKey):
        raise ValueError("bike-demand fold assets require a multi-partition key")
    fold_key = partition_key.keys_by_dimension.get("fold")
    if not isinstance(fold_key, str) or not fold_key.startswith("fold-"):
        raise ValueError(f"unexpected temporal-fold partition key {fold_key!r}")
    try:
        return int(fold_key.removeprefix("fold-"))
    except ValueError as error:
        raise ValueError(
            f"unexpected temporal-fold partition key {fold_key!r}"
        ) from error


def temporal_fold_model_annotations(
    *,
    context: dg.AssetExecutionContext,
) -> dict[str, int]:
    """Derive this native asset's model annotation from its fold partition."""

    return {"fold_number": temporal_fold_number(context)}


def temporal_fold_run_name(context: dg.AssetExecutionContext) -> str:
    """Name each dynamically partitioned OCLP Run for its real fold."""

    return f"Bike demand temporal validation fold {temporal_fold_number(context)}"

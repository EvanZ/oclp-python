"""Native Dagster assets for source acquisition and feature preparation."""

from __future__ import annotations

import dagster as dg
import pandas as pd
from oclp import (
    CsvArtifact,
    JsonArtifact,
    MlflowMetrics,
    artifact_set,
    computation,
    csv_artifact,
    mlflow,
)
from oclp.dagster import dagster_adapter
from sklearn.model_selection import TimeSeriesSplit

from bike_demand_service.dagster.partitions import IO_MANAGER_KEY, training_cycles
from bike_demand_service.dagster.resources import asset_out
from bike_demand_service.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    HOLDOUT_START,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    UCI_BIKE_SHARING_DATASET_ID,
    _fetch_source_frame,
    _normalize_source_frame,
    _temporal_fold_count,
    _timestamp_text,
)


# This package is a separate Dagster implementation. Native Dagster decorators
# own orchestration declarations such as keys, edges, partitions, and I/O; the
# inner OCLP decorators own the real Artifact and Computation contracts.
# ``data.py`` and ``modeling.py`` remain the ordinary runner implementation;
# they never import Dagster or activate its adapter. This module declares its
# OCLP contracts directly with SDK decorators and shares only plain domain
# helpers where that is useful. The adapter is active only at this boundary.
def prepare_features_value(
    source_snapshot: pd.DataFrame,
    training_plan: dict[str, object],
) -> dict[str, object]:
    """Prepare this scheduler implementation's time-ordered model inputs."""

    fold_count = _temporal_fold_count(training_plan)
    normalized = _normalize_source_frame(source_snapshot)
    feature_frame = normalized.loc[
        :, (TIMESTAMP_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN)
    ]
    feature_frame = feature_frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    training = feature_frame.loc[feature_frame[TIMESTAMP_COLUMN] < HOLDOUT_START]
    if len(training) <= fold_count:
        raise ValueError("not enough pre-holdout rows to create temporal folds")

    folds: list[dict[str, str | int]] = []
    splitter = TimeSeriesSplit(n_splits=fold_count)
    for number, (train_indices, validation_indices) in enumerate(
        splitter.split(training), start=1
    ):
        folds.append(
            {
                "fold": number,
                "train_end": _timestamp_text(
                    training.iloc[train_indices[-1]][TIMESTAMP_COLUMN]
                ),
                "validation_start": _timestamp_text(
                    training.iloc[validation_indices[0]][TIMESTAMP_COLUMN]
                ),
                "validation_end": _timestamp_text(
                    training.iloc[validation_indices[-1]][TIMESTAMP_COLUMN]
                ),
            }
        )
    return {
        "features": feature_frame,
        "fold_definition": {"strategy": "TimeSeriesSplit", "folds": folds},
        "data_metrics": {
            "source_rows": int(len(source_snapshot)),
            "prepared_rows": int(len(feature_frame)),
            "training_rows": int(len(training)),
            "holdout_rows": int(len(feature_frame) - len(training)),
        },
        "feature_contract": {
            "version": 1,
            "dataset": "UCI Bike Sharing Dataset (hourly)",
            "target": TARGET_COLUMN,
            "timestamp": TIMESTAMP_COLUMN,
            "features": list(FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "excluded_source_columns": ["instant", "dteday", "casual", "registered"],
            "holdout_start": _timestamp_text(HOLDOUT_START),
        },
    }


@dg.asset(
    key="bike_demand_raw_source",
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    deps=(dg.AssetKey("bike_demand_release_cycle"),),
    required_resource_keys={"oclp"},
)
@csv_artifact(
    name="UCI Bike Sharing source (CSV)",
    description_from_docstring=True,
    index=False,
    lineterminator="\n",
    adapters=(dagster_adapter(),),
)
def bike_demand_raw_source() -> pd.DataFrame:
    """Acquire the source snapshot for one training cycle."""

    return _fetch_source_frame(UCI_BIKE_SHARING_DATASET_ID)


@dg.multi_asset(
    outs={
        "features": asset_out("bike_demand_features"),
        "fold_definition": asset_out("bike_demand_fold_definition"),
        "feature_contract": asset_out("bike_demand_feature_contract"),
        "data_metrics": asset_out("bike_demand_data_metrics"),
    },
    ins={
        "source_snapshot": dg.AssetIn(key=dg.AssetKey("bike_demand_raw_source")),
        "training_plan": dg.AssetIn(key=dg.AssetKey("bike_demand_training_plan")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Bike demand CatBoost release",
    members={
        "features": ("features", "training-data"),
        "feature-contract": ("feature_contract", "serving-contract"),
    },
)
@mlflow(metrics=(MlflowMetrics(output_port="data_metrics", prefix="data"),))
@computation(
    name="Prepare bike demand features",
    description_from_docstring=True,
    inputs={
        "source_snapshot": CsvArtifact,
        "training_plan": JsonArtifact,
    },
    outputs={
        "features": CsvArtifact(
            name="Bike demand features",
            description=(
                "Time-indexed features and demand target prepared for training and "
                "holdout scoring."
            ),
            path="prepared/features.csv",
        ),
        "fold_definition": JsonArtifact(
            name="Temporal fold definition",
            description=(
                "TimeSeriesSplit fold windows used for temporal model validation."
            ),
            path="prepared/temporal-folds.json",
        ),
        "feature_contract": JsonArtifact(
            name="Feature contract",
            description=(
                "Serving contract defining the feature columns, categorical columns, "
                "target, and holdout boundary."
            ),
            path="prepared/feature-contract.json",
        ),
        "data_metrics": JsonArtifact(
            name="Training data metrics",
            description=(
                "Counts describing the source, prepared, training, and holdout rows."
            ),
            path="prepared/data-metrics.json",
        ),
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_prepare_features(
    source_snapshot: pd.DataFrame,
    training_plan: dict[str, object],
) -> dict[str, object]:
    """Prepare the source features and temporal-fold definition for one cycle."""

    return prepare_features_value(source_snapshot, training_plan)

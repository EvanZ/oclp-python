"""UCI data access and deterministic, leakage-safe feature preparation."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from oclp import (
    ArtifactHandle,
    CsvArtifact,
    JsonArtifact,
    computation,
    csv_artifact,
    json_artifact,
    parquet_artifact,
)
from sklearn.model_selection import TimeSeriesSplit
from ucimlrepo import fetch_ucirepo

UCI_BIKE_SHARING_DATASET_ID = 275
TARGET_COLUMN = "cnt"
TIMESTAMP_COLUMN = "timestamp"
HOLDOUT_START = pd.Timestamp("2012-07-01")
FEATURE_COLUMNS = (
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)
CATEGORICAL_FEATURES = (
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
)
SourceArtifactFormat = Literal["csv", "parquet", "json"]


def _fetch_source_frame(dataset_id: int) -> pd.DataFrame:
    """Fetch UCI's hourly Bike Sharing table before choosing its representation."""

    dataset = fetch_ucirepo(id=dataset_id)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if TARGET_COLUMN not in targets:
        raise ValueError("UCI Bike Sharing data did not expose the cnt target column")
    frame = features.copy()
    frame[TARGET_COLUMN] = targets[TARGET_COLUMN]
    return frame


@csv_artifact(
    name="UCI Bike Sharing source (CSV)",
    index=False,
    lineterminator="\n",
)
def download_source_csv(
    dataset_id: int = UCI_BIKE_SHARING_DATASET_ID,
) -> pd.DataFrame:
    """Acquire UCI's hourly Bike Sharing data as a persisted CSV Artifact."""

    return _fetch_source_frame(dataset_id)


@parquet_artifact(
    name="UCI Bike Sharing source (Parquet)",
    index=False,
    compression="zstd",
)
def download_source_parquet(
    dataset_id: int = UCI_BIKE_SHARING_DATASET_ID,
) -> pd.DataFrame:
    """Acquire UCI's hourly Bike Sharing data as a persisted Parquet Artifact."""

    return _fetch_source_frame(dataset_id)


@json_artifact(
    name="UCI Bike Sharing source (JSON)",
    serialization="pandas-table",
)
def download_source_json(
    dataset_id: int = UCI_BIKE_SHARING_DATASET_ID,
) -> pd.DataFrame:
    """Acquire UCI's hourly Bike Sharing data as a table-JSON Artifact."""

    return _fetch_source_frame(dataset_id)


def download_source_artifact(
    storage_format: SourceArtifactFormat = "csv",
    *,
    dataset_id: int = UCI_BIKE_SHARING_DATASET_ID,
) -> ArtifactHandle:
    """Acquire the UCI source in one selected durable representation.

    This is an experiment harness, not a second pipeline. The default batch
    flow continues to call :func:`download_source_csv`, so it remains on CSV.
    Every representation originates from the same UCI table but has its own
    logical Artifact identity because it has distinct persisted bytes.
    """

    downloaders = {
        "csv": download_source_csv,
        "parquet": download_source_parquet,
        "json": download_source_json,
    }
    return downloaders[storage_format](dataset_id)


@computation(
    id="urn:oclp-bike-demand:computation:prepare-features",
    name="Prepare bike demand features",
    inputs={
        "source_snapshot": CsvArtifact,
        "training_plan": JsonArtifact,
    },
    outputs={
        "features": CsvArtifact(
            name="Bike demand features",
            path="prepared/features.csv",
        ),
        "fold_definition": JsonArtifact(
            name="Temporal fold definition",
            path="prepared/temporal-folds.json",
        ),
        "feature_contract": JsonArtifact(
            name="Feature contract",
            path="prepared/feature-contract.json",
        ),
    },
)
def prepare_features(
    source_snapshot: pd.DataFrame,
    training_plan: dict[str, object],
) -> dict[str, object]:
    """Create time-ordered folds while excluding target-derived leakage fields."""

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


def _temporal_fold_count(training_plan: dict[str, object]) -> int:
    """Read the declared fold count from the durable configuration Artifact."""

    fold_count = training_plan.get("temporal_fold_count")
    if (
        not isinstance(fold_count, int)
        or isinstance(fold_count, bool)
        or fold_count < 1
    ):
        raise ValueError("training_plan.temporal_fold_count must be a positive integer")
    return fold_count


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix with CatBoost's categorical values normalized."""

    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str)
    return result


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return source rows available to fitting and temporal validation."""

    normalized = _normalize_feature_timestamps(frame)
    return normalized.loc[normalized[TIMESTAMP_COLUMN] < HOLDOUT_START].copy()


def holdout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the final time-based holdout, never used in fold fitting."""

    normalized = _normalize_feature_timestamps(frame)
    return normalized.loc[normalized[TIMESTAMP_COLUMN] >= HOLDOUT_START].copy()


def _normalize_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"dteday", "hr", TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Bike Sharing source data is missing columns: {', '.join(missing)}"
        )
    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(result["dteday"]) + pd.to_timedelta(
        result["hr"], unit="h"
    )
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="raise")
    return result


def _normalize_feature_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore the timestamp runtime type after a durable table reload.

    A CSV Artifact preserves the timestamp value but not pandas' datetime
    dtype.  This is a domain interpretation step, deliberately kept separate
    from the generic ``CsvArtifact -> pandas.DataFrame`` adapter.
    """

    if TIMESTAMP_COLUMN not in frame:
        raise ValueError(
            "Bike-demand feature data is missing the timestamp column "
            f"{TIMESTAMP_COLUMN!r}"
        )
    result = frame.copy()
    result[TIMESTAMP_COLUMN] = pd.to_datetime(result[TIMESTAMP_COLUMN], errors="raise")
    return result


def _timestamp_text(value: object) -> str:
    return pd.Timestamp(value).isoformat()

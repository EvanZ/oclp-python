"""UCI data access and deterministic, leakage-safe feature preparation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
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


@dataclass(frozen=True)
class PreparedFeatures:
    """Feature rows and split policy produced by the preparation boundary."""

    frame: pd.DataFrame
    folds: tuple[dict[str, str | int], ...]
    feature_contract: dict[str, object]


def download_source_data() -> pd.DataFrame:
    """Retrieve UCI's hourly Bike Sharing data as a single ordered table."""

    dataset = fetch_ucirepo(id=UCI_BIKE_SHARING_DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    if TARGET_COLUMN not in targets:
        raise ValueError("UCI Bike Sharing data did not expose the cnt target column")
    frame = features.copy()
    frame[TARGET_COLUMN] = targets[TARGET_COLUMN]
    return frame


def prepare_features(source: pd.DataFrame, *, fold_count: int = 3) -> PreparedFeatures:
    """Create time-ordered folds while excluding target-derived leakage fields."""

    normalized = _normalize_source_frame(source)
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
    return PreparedFeatures(
        frame=feature_frame,
        folds=tuple(folds),
        feature_contract={
            "version": 1,
            "dataset": "UCI Bike Sharing Dataset (hourly)",
            "target": TARGET_COLUMN,
            "timestamp": TIMESTAMP_COLUMN,
            "features": list(FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "excluded_source_columns": ["instant", "dteday", "casual", "registered"],
            "holdout_start": _timestamp_text(HOLDOUT_START),
        },
    )


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the model matrix with CatBoost's categorical values normalized."""

    result = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str)
    return result


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return source rows available to fitting and temporal validation."""

    return frame.loc[frame[TIMESTAMP_COLUMN] < HOLDOUT_START].copy()


def holdout_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the final time-based holdout, never used in fold fitting."""

    return frame.loc[frame[TIMESTAMP_COLUMN] >= HOLDOUT_START].copy()


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


def _timestamp_text(value: object) -> str:
    return pd.Timestamp(value).isoformat()

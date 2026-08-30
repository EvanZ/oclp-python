"""CatBoost training and scoring functions used by the observed Invocations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from oclp import definition
from oclp.models import PortDefinition
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bike_demand_service.data import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    holdout_rows,
    model_features,
    training_rows,
)


@dataclass(frozen=True)
class FoldResult:
    """In-memory result of one temporal validation model."""

    fold: int
    model_path: Path
    predictions: pd.DataFrame
    metrics: dict[str, float | int]


@definition(
    id="urn:oclp-bike-demand:definition:train-fold-model",
    name="Train bike-demand fold model",
    input_ports=(
        PortDefinition(name="dataset_snapshot", media_types=("application/json",)),
        PortDefinition(name="fold_definition", media_types=("application/json",)),
    ),
    output_ports=(
        PortDefinition(name="model"),
        PortDefinition(name="validation_predictions"),
        PortDefinition(name="metrics", media_types=("application/json",)),
    ),
)
def train_fold(
    frame: pd.DataFrame,
    fold: dict[str, str | int],
    *,
    model_path: Path,
) -> FoldResult:
    """Fit a CatBoost model on one past-only fold and score its next window."""

    training = training_rows(frame)
    train_end = pd.Timestamp(str(fold["train_end"]))
    validation_start = pd.Timestamp(str(fold["validation_start"]))
    validation_end = pd.Timestamp(str(fold["validation_end"]))
    fit_rows = training.loc[training[TIMESTAMP_COLUMN] <= train_end]
    validation = training.loc[
        (training[TIMESTAMP_COLUMN] >= validation_start)
        & (training[TIMESTAMP_COLUMN] <= validation_end)
    ]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = _new_model(train_dir=model_path.parent / "catboost-info")
    model.fit(
        model_features(fit_rows),
        fit_rows[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    prediction = model.predict(model_features(validation))
    model.save_model(model_path)
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: validation[TIMESTAMP_COLUMN].to_numpy(),
            "actual": validation[TARGET_COLUMN].to_numpy(),
            "prediction": prediction,
            "fold": int(fold["fold"]),
        }
    )
    return FoldResult(
        fold=int(fold["fold"]),
        model_path=model_path,
        predictions=predictions,
        metrics=_metrics(predictions),
    )


@definition(
    id="urn:oclp-bike-demand:definition:evaluate-candidate",
    name="Evaluate bike-demand candidate",
    input_ports=(
        PortDefinition(name="fold_models", cardinality="many"),
        PortDefinition(name="fold_predictions", cardinality="many"),
        PortDefinition(
            name="fold_metrics",
            cardinality="many",
            media_types=("application/json",),
        ),
    ),
    output_ports=(
        PortDefinition(name="evaluation", media_types=("application/json",)),
        PortDefinition(name="training_config", media_types=("application/json",)),
    ),
)
def evaluate_folds(results: tuple[FoldResult, ...]) -> dict[str, float | int | str]:
    """Aggregate exactly the validation results used for candidate selection."""

    if not results:
        raise ValueError("at least one fold result is required")
    prediction_frame = pd.concat([result.predictions for result in results])
    metrics = _metrics(prediction_frame)
    metrics["fold_count"] = len(results)
    metrics["quality_gate"] = "pass" if metrics["rmse"] <= 250 else "fail"
    return metrics


@definition(
    id="urn:oclp-bike-demand:definition:train-final-model",
    name="Train final bike-demand model",
    input_ports=(
        PortDefinition(name="dataset_snapshot", media_types=("application/json",)),
        PortDefinition(name="training_config", media_types=("application/json",)),
    ),
    output_ports=(PortDefinition(name="model"),),
)
def train_final_model(frame: pd.DataFrame, *, model_path: Path) -> CatBoostRegressor:
    """Fit the release candidate on every row before the untouched holdout."""

    fitting_rows = training_rows(frame)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = _new_model(train_dir=model_path.parent / "catboost-info")
    model.fit(
        model_features(fitting_rows),
        fitting_rows[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    model.save_model(model_path)
    return model


@definition(
    id="urn:oclp-bike-demand:definition:predict-bike-demand",
    name="Score bike-demand holdout set",
    input_ports=(
        PortDefinition(name="model_release"),
        PortDefinition(name="dataset_snapshot", media_types=("application/json",)),
    ),
    output_ports=(
        PortDefinition(name="predictions"),
        PortDefinition(name="metrics", media_types=("application/json",)),
    ),
)
def score_holdout(
    model: CatBoostRegressor, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Score the final, unobserved temporal holdout for the offline demo."""

    holdout = holdout_rows(frame)
    if holdout.empty:
        raise ValueError("the prepared data has no post-cutoff holdout rows")
    prediction = model.predict(model_features(holdout))
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: holdout[TIMESTAMP_COLUMN].to_numpy(),
            "actual": holdout[TARGET_COLUMN].to_numpy(),
            "prediction": prediction,
        }
    )
    return predictions, _metrics(predictions)


def _new_model(*, train_dir: Path) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=17,
        train_dir=str(train_dir),
        verbose=False,
    )


def _metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    actual = predictions["actual"]
    predicted = predictions["prediction"]
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(sqrt(mean_squared_error(actual, predicted))),
        "rows": int(len(predictions)),
    }

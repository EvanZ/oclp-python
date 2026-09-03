"""CatBoost training and scoring functions used by the observed Executions."""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Literal

import pandas as pd
from catboost import CatBoostRegressor
from oclp import (
    CatBoostModelArtifact,
    CsvArtifact,
    JsonArtifact,
    computation,
    evidence,
    json_artifact,
    many,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bike_demand_service.data import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    UCI_BIKE_SHARING_DATASET_ID,
    holdout_rows,
    model_features,
    training_rows,
)


@evidence(
    name="Temporal validation quality",
)
def temporal_validation_quality(
    evaluation: dict[str, float | int],
) -> Literal["pass", "fail", "error"]:
    """Accept a candidate only when its temporal validation RMSE is usable."""

    try:
        maximum_rmse = float(evaluation["temporal_validation_rmse_max"])
        rmse = float(evaluation["rmse"])
    except (KeyError, TypeError, ValueError):
        return "error"
    if not isfinite(maximum_rmse) or maximum_rmse <= 0:
        return "error"
    return "pass" if rmse <= maximum_rmse else "fail"


@evidence(
    name="Holdout response validation",
)
def holdout_response(
    metrics: dict[str, float | int],
) -> Literal["pass", "fail", "error"]:
    """Verify the holdout scorer emitted finite numeric regression metrics."""

    return (
        "pass"
        if all(isfinite(float(metrics[name])) for name in ("mae", "rmse"))
        else "fail"
    )


@json_artifact(
    id=lambda run_id: f"urn:oclp-bike-demand:artifact:training-plan:{run_id}",
    name="Bike demand training plan",
)
def create_training_plan(*, run_id: str, fold_count: int) -> dict[str, object]:
    """Persist the model workflow's declared fold configuration.

    This is an Artifact boundary, not an orchestration Computation: it makes
    the configuration a durable input that feature preparation consumes.
    """

    return {
        "run_id": run_id,
        "dataset": "UCI Bike Sharing Dataset (hourly)",
        "dataset_id": UCI_BIKE_SHARING_DATASET_ID,
        "temporal_fold_count": fold_count,
        "model": "CatBoostRegressor",
    }


@computation(
    id="urn:oclp-bike-demand:computation:train-fold-model",
    name="Train bike demand fold",
    inputs={
        "feature_table": CsvArtifact,
        "fold_definition": JsonArtifact,
    },
    outputs={
        "model": CatBoostModelArtifact(
            name="Temporal fold model",
        ),
        "validation_predictions": CsvArtifact(
            name="Validation predictions",
        ),
        "metrics": JsonArtifact(
            name="Validation metrics",
        ),
    },
)
def train_fold(
    feature_table: pd.DataFrame,
    fold_definition: dict[str, object],
    *,
    fold_number: int,
) -> dict[str, object]:
    """Fit one materialized-data temporal fold and score its next window."""

    fold = _fold_for_number(fold_definition, fold_number)
    training = training_rows(feature_table)
    train_end = pd.Timestamp(str(fold["train_end"]))
    validation_start = pd.Timestamp(str(fold["validation_start"]))
    validation_end = pd.Timestamp(str(fold["validation_end"]))
    fit_rows = training.loc[training[TIMESTAMP_COLUMN] <= train_end]
    validation = training.loc[
        (training[TIMESTAMP_COLUMN] >= validation_start)
        & (training[TIMESTAMP_COLUMN] <= validation_end)
    ]
    model = _new_model()
    model.fit(
        model_features(fit_rows),
        fit_rows[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    prediction = model.predict(model_features(validation))
    predictions = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: validation[TIMESTAMP_COLUMN].to_numpy(),
            "actual": validation[TARGET_COLUMN].to_numpy(),
            "prediction": prediction,
            "fold": int(fold["fold"]),
        }
    )
    return {
        "model": model,
        "validation_predictions": predictions,
        "metrics": _metrics(predictions),
    }


def _fold_for_number(
    fold_definition: dict[str, object], fold_number: int
) -> dict[str, str | int]:
    """Select one named fold from the durable temporal-fold JSON document."""

    raw_folds = fold_definition.get("folds")
    if not isinstance(raw_folds, list):
        raise ValueError("fold definition must contain a list at 'folds'")
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, dict):
            raise ValueError("fold definition entries must be JSON objects")
        if raw_fold.get("fold") != fold_number:
            continue
        required = {"fold", "train_end", "validation_start", "validation_end"}
        if required.difference(raw_fold):
            raise ValueError(
                f"fold definition entry is missing one of {', '.join(sorted(required))}"
            )
        return {
            "fold": int(raw_fold["fold"]),
            "train_end": str(raw_fold["train_end"]),
            "validation_start": str(raw_fold["validation_start"]),
            "validation_end": str(raw_fold["validation_end"]),
        }
    raise ValueError(f"fold definition does not contain fold {fold_number}")


@computation(
    id="urn:oclp-bike-demand:computation:evaluate-candidate",
    name="Evaluate bike demand candidate",
    inputs={"fold_predictions": many(CsvArtifact)},
    outputs={
        "evaluation": JsonArtifact(name="Candidate evaluation"),
        "training_config": JsonArtifact(
            name="Final training configuration"
        ),
    },
    requires=(temporal_validation_quality,),
)
def evaluate_folds(
    fold_predictions: tuple[pd.DataFrame, ...],
    *,
    temporal_validation_rmse_max: float = 250,
) -> dict[str, object]:
    """Aggregate exactly the validation results used for candidate selection."""

    if not fold_predictions:
        raise ValueError("at least one fold prediction Artifact is required")
    evaluation = _metrics(pd.concat(fold_predictions))
    evaluation["fold_count"] = len(fold_predictions)
    # Persist the gate's concrete threshold with the metrics it evaluates.  The
    # Execution parameters capture the same value for the invocation record.
    evaluation["temporal_validation_rmse_max"] = temporal_validation_rmse_max
    return {
        "evaluation": evaluation,
        "training_config": _training_config(),
    }


@computation(
    id="urn:oclp-bike-demand:computation:train-final-model",
    name="Train final bike demand model",
    inputs={
        "feature_table": CsvArtifact,
        "training_config": JsonArtifact,
    },
    outputs={
        "model": CatBoostModelArtifact(
            name="Final CatBoost model",
        ),
    },
)
def train_final_model(
    feature_table: pd.DataFrame,
    training_config: dict[str, object],
    *,
    training_window: Literal["all-pre-holdout-rows"] = "all-pre-holdout-rows",
) -> CatBoostRegressor:
    """Fit the release candidate from materialized data and configuration."""

    if training_window != "all-pre-holdout-rows":  # pragma: no cover - type guard.
        raise ValueError(f"unsupported bike-demand training window: {training_window}")
    fitting_rows = training_rows(feature_table)
    model = _new_model(
        training_config=training_config,
    )
    model.fit(
        model_features(fitting_rows),
        fitting_rows[TARGET_COLUMN],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    return model


@computation(
    id="urn:oclp-bike-demand:computation:predict-bike-demand",
    name="Score bike demand holdout",
    inputs={
        "model": CatBoostModelArtifact,
        "feature_table": CsvArtifact,
    },
    outputs={
        "predictions": CsvArtifact(name="Holdout predictions"),
        "metrics": JsonArtifact(name="Holdout metrics"),
    },
    requires=(holdout_response,),
)
def score_holdout(
    model: CatBoostRegressor, feature_table: pd.DataFrame
) -> dict[str, object]:
    """Score the final, unobserved temporal holdout for the offline demo."""

    holdout = holdout_rows(feature_table)
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
    return {"predictions": predictions, "metrics": _metrics(predictions)}


def _training_config() -> dict[str, float | int | str]:
    """Return the explicit training configuration selected by this lifecycle."""

    return {
        "model": "CatBoostRegressor",
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 17,
        "selection_metric": "temporal_validation_rmse",
    }


def _new_model(
    *,
    training_config: dict[str, object] | None = None,
) -> CatBoostRegressor:
    """Build the CatBoost estimator from defaults or a durable config Artifact."""

    config = training_config or {
        "model": "CatBoostRegressor",
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 17,
    }
    if config.get("model") != "CatBoostRegressor":
        raise ValueError("bike-demand training config must select CatBoostRegressor")
    try:
        iterations = int(config["iterations"])
        depth = int(config["depth"])
        learning_rate = float(config["learning_rate"])
        random_seed = int(config["random_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "bike-demand training config requires numeric iterations, depth, "
            "learning_rate, and random_seed values"
        ) from error
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="RMSE",
        random_seed=random_seed,
        allow_writing_files=False,
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

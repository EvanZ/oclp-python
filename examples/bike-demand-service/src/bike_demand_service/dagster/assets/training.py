"""Native Dagster assets for temporal validation and final model release."""

from __future__ import annotations

import dagster as dg
import pandas as pd
from catboost import CatBoostRegressor
from oclp import (
    BytesArtifact,
    CatBoostModelArtifact,
    CsvArtifact,
    JsonArtifact,
    MlflowMetrics,
    artifact_set,
    computation,
    many,
    mlflow,
)
from oclp.dagster import dagster_adapter

from bike_demand_service.dagster.partitions import (
    IO_MANAGER_KEY,
    TEMPORAL_VALIDATION_RMSE_MAX,
    cycle_to_fold,
    fold_partitions,
    temporal_fold_model_annotations,
    temporal_fold_number,
    temporal_fold_run_name,
    training_cycles,
)
from bike_demand_service.dagster.resources import asset_out
from bike_demand_service.modeling import (
    chart_holdout_demand_forecast_value,
    chart_temporal_validation_quality_value,
    evaluate_folds_value,
    holdout_response,
    score_holdout_value,
    temporal_validation_quality,
    train_final_model_value,
    train_fold_value,
)


@dg.multi_asset(
    outs={
        "model": asset_out("bike_demand_fold_model"),
        "validation_predictions": asset_out("bike_demand_fold_predictions"),
        "metrics": asset_out("bike_demand_fold_metrics"),
    },
    ins={
        "feature_table": dg.AssetIn(
            key=dg.AssetKey("bike_demand_features"),
            partition_mapping=cycle_to_fold,
        ),
        "fold_definition": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_definition"),
            partition_mapping=cycle_to_fold,
        ),
    },
    group_name="bike_demand",
    partitions_def=fold_partitions,
    required_resource_keys={"oclp"},
)
@mlflow(metrics=(MlflowMetrics(output_port="metrics", prefix="temporal-fold"),))
@computation(
    name="Train bike demand fold",
    description_from_docstring=True,
    inputs={
        "feature_table": CsvArtifact,
        "fold_definition": JsonArtifact,
    },
    outputs={
        "model": CatBoostModelArtifact(
            name="Temporal fold model",
            description="CatBoost regressor fitted for one temporal validation fold.",
            annotation_factory=temporal_fold_model_annotations,
        ),
        "validation_predictions": CsvArtifact(
            name="Validation predictions",
            description=(
                "Predictions and observed demand for one temporal validation window."
            ),
        ),
        "metrics": JsonArtifact(
            name="Validation metrics",
            description="MAE, RMSE, and row count for one temporal validation fold.",
        ),
    },
    adapters=(dagster_adapter(run_name=temporal_fold_run_name),),
)
def bike_demand_train_fold(
    context: dg.AssetExecutionContext,
    feature_table: pd.DataFrame,
    fold_definition: dict[str, object],
) -> dict[str, object]:
    """Train one dynamically partitioned temporal fold."""

    return train_fold_value(
        feature_table,
        fold_definition,
        fold_number=temporal_fold_number(context),
    )


@dg.multi_asset(
    outs={
        "evaluation": asset_out("bike_demand_candidate_evaluation"),
        "training_config": asset_out("bike_demand_training_config"),
    },
    ins={
        "fold_predictions": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_predictions"),
            partition_mapping=cycle_to_fold,
        ),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    config_schema={
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Bike demand CatBoost release",
    members={
        "temporal-evaluation": ("evaluation", "validation-report"),
        "training-config": ("training_config", "training-config"),
    },
)
@mlflow(metrics=(MlflowMetrics(output_port="evaluation", prefix="candidate"),))
@computation(
    name="Evaluate bike demand candidate",
    description_from_docstring=True,
    inputs={"fold_predictions": many(CsvArtifact)},
    outputs={
        "evaluation": JsonArtifact(
            name="Candidate evaluation",
            description=(
                "Aggregated temporal validation metrics and the release quality "
                "threshold."
            ),
        ),
        "training_config": JsonArtifact(
            name="Final training configuration",
            description=(
                "CatBoost hyperparameters selected for the final release model."
            ),
        ),
    },
    requires=(temporal_validation_quality,),
    adapters=(dagster_adapter(),),
)
def bike_demand_evaluate_candidate(
    fold_predictions: tuple[pd.DataFrame, ...],
    *,
    temporal_validation_rmse_max: float = TEMPORAL_VALIDATION_RMSE_MAX,
) -> dict[str, object]:
    """Aggregate the fold predictions into the candidate evaluation."""

    return evaluate_folds_value(
        fold_predictions,
        temporal_validation_rmse_max=temporal_validation_rmse_max,
    )


@dg.asset(
    key="bike_demand_temporal_validation_chart",
    ins={
        "fold_predictions": dg.AssetIn(
            key=dg.AssetKey("bike_demand_fold_predictions"),
            partition_mapping=cycle_to_fold,
        ),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    config_schema={
        "temporal_validation_rmse_max": dg.Field(
            float,
            default_value=TEMPORAL_VALIDATION_RMSE_MAX,
            is_required=False,
        ),
    },
    required_resource_keys={"oclp"},
)
@mlflow(payloads=("chart",))
@computation(
    name="Chart temporal validation quality",
    description_from_docstring=True,
    inputs={"fold_predictions": many(CsvArtifact)},
    outputs={
        "chart": BytesArtifact(
            name="Temporal validation quality chart",
            description=(
                "Per-fold validation RMSE compared with the configured quality "
                "threshold."
            ),
            media_type="image/png",
            suffix="png",
        )
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_chart_temporal_validation(
    fold_predictions: tuple[pd.DataFrame, ...],
    *,
    temporal_validation_rmse_max: float = TEMPORAL_VALIDATION_RMSE_MAX,
) -> dict[str, bytes]:
    """Chart temporal-validation quality across the planned folds."""

    return chart_temporal_validation_quality_value(
        fold_predictions,
        temporal_validation_rmse_max=temporal_validation_rmse_max,
    )


@dg.asset(
    key="bike_demand_final_model",
    ins={
        "feature_table": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
        "training_config": dg.AssetIn(key=dg.AssetKey("bike_demand_training_config")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Bike demand CatBoost release",
    output_port="model",
    role="model",
)
@computation(
    name="Train final bike demand model",
    description_from_docstring=True,
    inputs={
        "feature_table": CsvArtifact,
        "training_config": JsonArtifact,
    },
    outputs={
        "model": CatBoostModelArtifact(
            name="Final CatBoost model",
            description=(
                "Release-candidate CatBoost model trained on all pre-holdout rows."
            ),
        ),
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_train_final_model(
    feature_table: pd.DataFrame,
    training_config: dict[str, object],
) -> CatBoostRegressor:
    """Train the final model from validated cycle inputs."""

    return train_final_model_value(feature_table, training_config)


@dg.multi_asset(
    outs={
        "predictions": asset_out("bike_demand_holdout_predictions"),
        "metrics": asset_out("bike_demand_holdout_metrics"),
    },
    ins={
        "model": dg.AssetIn(key=dg.AssetKey("bike_demand_final_model")),
        "feature_table": dg.AssetIn(key=dg.AssetKey("bike_demand_features")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    required_resource_keys={"oclp"},
)
@mlflow(metrics=(MlflowMetrics(output_port="metrics", prefix="holdout"),))
@computation(
    name="Score bike demand holdout",
    description_from_docstring=True,
    inputs={
        "model": CatBoostModelArtifact,
        "feature_table": CsvArtifact,
    },
    outputs={
        "predictions": CsvArtifact(
            name="Holdout predictions",
            description=(
                "Predictions and observed demand for the untouched temporal holdout."
            ),
        ),
        "metrics": JsonArtifact(
            name="Holdout metrics",
            description="MAE, RMSE, and row count for the untouched temporal holdout.",
        ),
    },
    requires=(holdout_response,),
    adapters=(dagster_adapter(),),
)
def bike_demand_score_holdout(
    model: CatBoostRegressor,
    feature_table: pd.DataFrame,
) -> dict[str, object]:
    """Score the final model against the holdout partition."""

    return score_holdout_value(model, feature_table)


@dg.asset(
    key="bike_demand_holdout_forecast_chart",
    ins={
        "predictions": dg.AssetIn(key=dg.AssetKey("bike_demand_holdout_predictions")),
    },
    group_name="bike_demand",
    partitions_def=training_cycles,
    io_manager_key=IO_MANAGER_KEY,
    required_resource_keys={"oclp"},
)
@mlflow(payloads=("chart",))
@computation(
    name="Chart holdout demand forecast",
    description_from_docstring=True,
    inputs={"predictions": CsvArtifact},
    outputs={
        "chart": BytesArtifact(
            name="Holdout demand forecast chart",
            description=(
                "Observed and predicted hourly demand across the untouched "
                "holdout window."
            ),
            media_type="image/png",
            suffix="png",
        )
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_chart_holdout_forecast(
    predictions: pd.DataFrame,
) -> dict[str, bytes]:
    """Chart the final model's holdout-demand forecast."""

    return chart_holdout_demand_forecast_value(predictions)

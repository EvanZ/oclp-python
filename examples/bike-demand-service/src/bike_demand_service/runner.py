"""Run the complete, locally inspectable OCLP bike-demand workflow."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from oclp import (
    MlflowAdapter,
    OclpRun,
    capture_git_source_overlay,
    load_release_manifest,
    observe_run,
    run,
    source_from_git_checkout,
    validate_derivation_graph,
    validate_execution_acceptance,
    validate_execution_hierarchy,
)
from oclp.models import GitSource, RecordReference
from oclp.publishing import LocalArtifactPublisher

from bike_demand_service.data import (
    HOLDOUT_START,
    UCI_BIKE_SHARING_DATASET_ID,
    download_source_csv,
    prepare_features,
)
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.modeling import (
    create_training_plan,
    evaluate_folds,
    score_holdout,
    train_final_model,
    train_fold,
)
from bike_demand_service.service import (
    persist_prediction_request,
    predict_bike_demand,
)

_RELEASE_SMOKE_REQUEST: dict[str, int | float] = {
    "season": 1,
    "yr": 1,
    "mnth": 7,
    "hr": 12,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.65,
    "atemp": 0.62,
    "hum": 0.55,
    "windspeed": 0.18,
}
_MLFLOW_EXPERIMENT_NAME = "oclp-bike-demand-service"


@dataclass(frozen=True)
class DemoRunResult:
    """Useful local destinations from one observed model-training run."""

    materialization_id: str
    training_run_id: str
    model_release: RecordReference
    model_release_manifest: RecordReference
    model_release_manifest_path: str
    release_smoke_run_id: str
    release_smoke_execution: RecordReference
    release_smoke_response: RecordReference
    oclp_root: str
    mlflow_tracking_uri: str


@dataclass(frozen=True)
class _ReleaseSmokeTestResult:
    """Exact records produced by one release-backed inference check."""

    execution: RecordReference
    response: RecordReference


@run(
    name="Bike demand model training",
    adapters=(
        MlflowAdapter(experiment_name=_MLFLOW_EXPERIMENT_NAME),
    ),
    required_evidence_policy="raise",
)
def run_bike_training(
    *,
    observed: OclpRun,
    materialization_id: str,
    fold_count: int,
    temporal_validation_rmse_max: float,
) -> None:
    """Execute the application's real data and model flow once.

    ``@run`` gives every real decorated Computation the same SDK-owned run
    profile. The active SDK context carries exact Artifact bindings between
    decorated calls automatically. ``observed`` is used only for the optional
    application-selected MLflow run context; the SDK-owned adapter mirrors
    records, model payloads, and declared output metrics automatically.
    """

    # Acquisition is an Artifact boundary, not a derived Computation. The
    # decorated fetcher returns a CsvArtifact handle. The runtime adapts its
    # verified CSV bytes to prepare_features' pandas input.
    training_plan = create_training_plan(
        materialization_id=materialization_id,
        fold_count=fold_count,
    )
    mlflow = observed.adapter(MlflowAdapter)
    mlflow.log_parameters(
        {
            "materialization_id": materialization_id,
            "temporal_fold_count": fold_count,
            "temporal_validation_rmse_max": temporal_validation_rmse_max,
            "uci_dataset_id": UCI_BIKE_SHARING_DATASET_ID,
        }
    )
    source_snapshot = download_source_csv()

    prepared = prepare_features(source_snapshot, training_plan)
    mlflow.log_metrics({"source_rows": len(prepared["features"])})

    # Passing exact raw values between decorated calls is sufficient within
    # this OclpRun: the SDK reuses their materialized Artifact bindings when it
    # records the next Execution.
    feature_table = prepared["features"]
    folds = prepared["fold_definition"]

    fold_prediction_artifacts = []
    for fold in prepared["fold_definition"]["folds"]:
        fold_number = int(fold["fold"])
        result = train_fold(
            feature_table,
            folds,
            fold_number=fold_number,
        )
        fold_prediction_artifacts.append(result["validation_predictions"])

    evaluation_result = evaluate_folds(
        tuple(fold_prediction_artifacts),
        temporal_validation_rmse_max=temporal_validation_rmse_max,
    )
    training_config_value = evaluation_result["training_config"]

    final_model = train_final_model(
        feature_table,
        training_config_value,
        training_window="all-pre-holdout-rows",
    )
    mlflow.log_metrics(
        {
            "training_rows": int(
                (prepared["features"]["timestamp"] < HOLDOUT_START).sum()
            )
        }
    )

    score_holdout(final_model, feature_table)


@run(
    name="Release inference smoke test",
    required_evidence_policy="raise",
)
def run_release_smoke_test(
    *,
    observed: OclpRun,
    release_manifest_path: Path,
    materialization_id: str,
) -> _ReleaseSmokeTestResult:
    """Score one fixed request with the model selected by a release manifest.

    This is intentionally a separate run from training. Its prediction
    Execution consumes the exact ArtifactSet selected by the preceding release
    manifest, so a lineage explorer shows a sibling inference branch connected
    through the released bundle rather than treating service work as part of
    the training Execution tree.
    """

    release = load_release_manifest(release_manifest_path)
    request_artifact = persist_prediction_request(
        payload=_RELEASE_SMOKE_REQUEST,
    )
    result = predict_bike_demand(
        release,
        request_artifact,
    )
    return _ReleaseSmokeTestResult(
        execution=observed.execution_for(result),
        response=observed.outputs_for(result)["prediction_response"].reference,
    )


def run_demo(
    *,
    materialization_id: str,
    fold_count: int = 3,
    temporal_validation_rmse_max: float = 250,
    environment: DemoEnvironment | None = None,
) -> DemoRunResult:
    """Bootstrap and execute the decorated bike-demand model-training run.

    This narrow application entry point chooses local filesystem destinations,
    a Git source basis, and the optional MLflow observer. The SDK owns the
    active run and UUID-based run profile for the real workflow above.
    """

    _validate_materialization_id(materialization_id)
    _validate_temporal_validation_rmse_max(temporal_validation_rmse_max)
    environment = environment or DemoEnvironment.default()
    environment.prepare()
    (environment.mlflow_root / "artifacts").mkdir(parents=True, exist_ok=True)

    with LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.materialization_root(materialization_id),
    ) as publisher:
        source = source_from_git_checkout(
            environment.project_root,
            path="examples/bike-demand-service/src/bike_demand_service",
        )
        if isinstance(source, GitSource) and source.dirty:
            source = capture_git_source_overlay(
                environment.project_root,
                source=source,
                publisher=publisher,
                name="Bike-demand training source overlay",
                relative_path=f"source-overlays/{materialization_id}",
            )
        with observe_run(
            run_bike_training,
            publisher=publisher,
            source=source,
        ) as observed:
            assert observed.run_id is not None
            training_run_id = str(observed.run_id)
            run_bike_training(
                observed=observed,
                materialization_id=materialization_id,
                fold_count=fold_count,
                temporal_validation_rmse_max=temporal_validation_rmse_max,
            )
        model_release = observed.artifact_set("Bike demand CatBoost release")
        assert model_release.manifest is not None
    release_smoke_materialization_id = f"{materialization_id}-release-smoke"
    with LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.materialization_root(release_smoke_materialization_id),
    ) as smoke_publisher:
        with observe_run(
            run_release_smoke_test,
            publisher=smoke_publisher,
            source=source,
        ) as observed:
            assert observed.run_id is not None
            release_smoke_run_id = str(observed.run_id)
            smoke_result = run_release_smoke_test(
                observed=observed,
                release_manifest_path=model_release.manifest.path,
                materialization_id=release_smoke_materialization_id,
            )
        records = smoke_publisher.records()
        validate_derivation_graph(records)
        validate_execution_acceptance(records)
        validate_execution_hierarchy(records)

    return DemoRunResult(
        materialization_id=materialization_id,
        training_run_id=training_run_id,
        model_release=model_release.reference,
        model_release_manifest=model_release.manifest.reference,
        model_release_manifest_path=str(model_release.manifest.path),
        release_smoke_run_id=release_smoke_run_id,
        release_smoke_execution=smoke_result.execution,
        release_smoke_response=smoke_result.response,
        oclp_root=str(environment.oclp_root),
        mlflow_tracking_uri=_mlflow_tracking_uri(environment),
    )


def _validate_materialization_id(materialization_id: str) -> None:
    if not materialization_id or any(
        character.isspace() for character in materialization_id
    ):
        raise ValueError(
            "materialization_id must be a non-empty value without whitespace"
        )


def _mlflow_tracking_uri(environment: DemoEnvironment) -> str:
    """Return the local URI derived by the SDK adapter for this environment."""

    return "sqlite:///" + (environment.mlflow_root / "mlflow.db").resolve().as_posix()


def _validate_temporal_validation_rmse_max(value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(
            "temporal_validation_rmse_max must be a finite positive number"
        )

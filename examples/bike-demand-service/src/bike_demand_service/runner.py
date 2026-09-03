"""Run the complete, locally inspectable OCLP bike-demand lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from oclp import (
    OclpRun,
    lifecycle,
    load_release_manifest,
    observe_lifecycle,
    source_from_git_checkout,
    validate_derivation_graph,
    validate_execution_acceptance,
    validate_execution_hierarchy,
)
from oclp.models import RecordReference
from oclp.publishing import LocalArtifactPublisher

from bike_demand_service.data import (
    HOLDOUT_START,
    UCI_BIKE_SHARING_DATASET_ID,
    download_source_csv,
    prepare_features,
)
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.mlflow import (
    MLflowSettings,
    MLflowTracker,
    create_mlflow_tracker,
)
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


@dataclass(frozen=True)
class DemoRunResult:
    """Useful local destinations from an observed model lifecycle run."""

    run_id: str
    model_release: RecordReference
    model_release_manifest: RecordReference
    model_release_manifest_path: str
    release_smoke_run_id: str
    release_smoke_execution: RecordReference
    release_smoke_response: RecordReference
    oclp_root: str
    mlflow_tracking_uri: str


@dataclass(frozen=True)
class _ObservedLifecycleResult:
    """The OCLP-owned outcome returned by the decorated workflow body."""

    model_release: RecordReference
    model_release_manifest: RecordReference
    model_release_manifest_path: str


@dataclass(frozen=True)
class _ReleaseSmokeTestResult:
    """Exact records produced by one release-backed inference check."""

    execution: RecordReference
    response: RecordReference


@lifecycle(
    namespace="urn:oclp-bike-demand",
    name="Bike demand model lifecycle",
)
def run_bike_lifecycle(
    *,
    observed: OclpRun,
    tracker: MLflowTracker,
    run_id: str,
    fold_count: int,
    temporal_validation_rmse_max: float,
) -> _ObservedLifecycleResult:
    """Execute the application's real data and model flow once.

    ``@lifecycle`` gives every real decorated Computation the same SDK-owned
    lifecycle profile. ``observed`` is used here only to retrieve exact OCLP
    references for the existing, optional MLflow mirror; it does not publish
    Executions, Events, or Artifacts itself.
    """

    # Acquisition is an Artifact boundary, not a derived Computation. The
    # decorated fetcher returns a CsvArtifact handle. The runtime adapts its
    # verified CSV bytes to prepare_features' pandas input.
    training_plan = create_training_plan(
        run_id=run_id,
        fold_count=fold_count,
    )
    source_snapshot = download_source_csv()
    tracker.mirror_artifacts(
        artifacts={
            "source-snapshot": source_snapshot,
            "training-plan": training_plan,
        },
        artifact_path="oclp/acquired",
    )

    prepared = prepare_features(source_snapshot, training_plan)
    prepare_artifacts = observed.outputs_for(prepared)
    prepare_ref = observed.execution_for(prepared)
    prepare_computation = observed.computation_for(prepared)
    tracker.log_parameters({"uci_dataset_id": UCI_BIKE_SHARING_DATASET_ID})
    tracker.log_metrics({"source_rows": float(len(prepared["features"]))})

    feature_table = prepare_artifacts["features"]
    folds = prepare_artifacts["fold_definition"]
    feature_contract = prepare_artifacts["feature_contract"]
    with tracker.run("Prepare Features", nested=True):
        tracker.attach_execution(
            execution=prepare_ref,
            computation=prepare_computation,
            inputs={
                "source_snapshot": (source_snapshot.reference,),
                "training_plan": (training_plan.reference,),
            },
            outputs={
                port: (artifact.reference,)
                for port, artifact in prepare_artifacts.items()
            },
            artifacts=prepare_artifacts,
        )
        tracker.log_parameters(
            {
                "fold_count": len(prepared["fold_definition"]["folds"]),
                "holdout_start": HOLDOUT_START.isoformat(),
            }
        )
        tracker.log_metrics({"prepared_rows": float(len(prepared["features"]))})

    fold_prediction_artifacts = []
    for fold in prepared["fold_definition"]["folds"]:
        fold_number = int(fold["fold"])
        with tracker.run(f"Train temporal fold {fold_number}", nested=True):
            tracker.log_parameters(
                {
                    "fold": fold_number,
                    "train_end": str(fold["train_end"]),
                    "validation_start": str(fold["validation_start"]),
                    "validation_end": str(fold["validation_end"]),
                    "model": "CatBoostRegressor",
                }
            )
            result = train_fold(
                feature_table,
                folds,
                fold_number=fold_number,
            )
            train_artifacts = observed.outputs_for(result)
            train_ref = observed.execution_for(result)
            train_computation = observed.computation_for(result)
            model_artifact = train_artifacts["model"]
            predictions = train_artifacts["validation_predictions"]
            metrics = train_artifacts["metrics"]
            tracker.attach_execution(
                execution=train_ref,
                computation=train_computation,
                inputs={
                    "feature_table": (feature_table.reference,),
                    "fold_definition": (folds.reference,),
                },
                outputs={
                    "model": (model_artifact.reference,),
                    "validation_predictions": (predictions.reference,),
                    "metrics": (metrics.reference,),
                },
                artifacts=train_artifacts,
            )
            tracker.log_metrics(result["metrics"])
        fold_prediction_artifacts.append(predictions)

    with tracker.run("Evaluate bike-demand candidate", nested=True):
        evaluation_result = evaluate_folds(
            tuple(fold_prediction_artifacts),
            temporal_validation_rmse_max=temporal_validation_rmse_max,
        )
        evaluation_artifacts = observed.outputs_for(evaluation_result)
        evaluation_ref = observed.execution_for(evaluation_result)
        evaluation_computation = observed.computation_for(evaluation_result)
        quality_evidence = observed.evidence_for(evaluation_result)
        evaluation_artifact = evaluation_artifacts["evaluation"]
        training_config = evaluation_artifacts["training_config"]
        tracker.attach_execution(
            execution=evaluation_ref,
            computation=evaluation_computation,
            inputs={
                "fold_predictions": tuple(
                    item.reference for item in fold_prediction_artifacts
                ),
            },
            outputs={
                port: (artifact.reference,)
                for port, artifact in evaluation_artifacts.items()
            },
            artifacts=evaluation_artifacts,
        )
        tracker.log_metrics(evaluation_result["evaluation"])
        if any(item.outcome != "pass" for item in quality_evidence):
            raise RuntimeError("bike-demand temporal quality gate failed")

    with tracker.run("Train final bike-demand model", nested=True):
        tracker.log_parameters(
            {
                "model": "CatBoostRegressor",
                "training_window": "all pre-holdout rows",
            }
        )
        final_result = train_final_model(
            feature_table,
            training_config,
            training_window="all-pre-holdout-rows",
        )
        final_artifacts = observed.outputs_for(final_result)
        final_train_ref = observed.execution_for(final_result)
        final_train_computation = observed.computation_for(final_result)
        final_model_artifact = final_artifacts["model"]
        tracker.attach_execution(
            execution=final_train_ref,
            computation=final_train_computation,
            inputs={
                "feature_table": (feature_table.reference,),
                "training_config": (training_config.reference,),
            },
            outputs={"model": (final_model_artifact.reference,)},
            artifacts=final_artifacts,
        )
        tracker.log_metrics(
            {
                "training_rows": int(
                    (prepared["features"]["timestamp"] < HOLDOUT_START).sum()
                )
            }
        )

    with tracker.run("Publish bike-demand model release", nested=True):
        model_release = observed.publish_artifact_set(
            key="model-release",
            name="Bike demand CatBoost release",
            members={
                "model": (final_model_artifact, "model"),
                "feature-contract": (feature_contract, "serving-contract"),
                "temporal-evaluation": (
                    evaluation_artifact,
                    "validation-report",
                ),
                "training-config": (training_config, "training-config"),
                "feature-table": (feature_table, "training-data"),
            },
            materialize_manifest=True,
            manifest_name="Bike demand release manifest",
        )
        assert model_release.manifest is not None
        tracker.attach_artifact_set(
            artifact_set=model_release.reference,
            artifacts={
                "model": final_model_artifact,
                "feature-contract": feature_contract,
                "temporal-evaluation": evaluation_artifact,
                "training-config": training_config,
                "feature-table": feature_table,
                # The SDK-created sidecar identifies this exact ArtifactSet;
                # it is mirrored beside the release rather than being a set
                # member (which would make its own digest cyclic).
                "release-manifest-sidecar": model_release.manifest,
            },
        )
        tracker.log_metrics(
            {"release_members": len(model_release.artifact_set.members)}
        )

    with tracker.run("Score bike-demand holdout", nested=True):
        score_result = score_holdout(final_model_artifact, feature_table)
        score_artifacts = observed.outputs_for(score_result)
        score_ref = observed.execution_for(score_result)
        score_computation = observed.computation_for(score_result)
        holdout_evidence = observed.evidence_for(score_result)
        if any(item.outcome != "pass" for item in holdout_evidence):
            raise RuntimeError("bike-demand holdout response contract failed")
        tracker.attach_execution(
            execution=score_ref,
            computation=score_computation,
            inputs={
                "model": (final_model_artifact.reference,),
                "feature_table": (feature_table.reference,),
            },
            outputs={
                port: (artifact.reference,)
                for port, artifact in score_artifacts.items()
            },
            artifacts=score_artifacts,
        )
        tracker.log_metrics(score_result["metrics"])

    return _ObservedLifecycleResult(
        model_release=model_release.reference,
        model_release_manifest=model_release.manifest.reference,
        model_release_manifest_path=str(model_release.manifest.path),
    )


@lifecycle(
    namespace="urn:oclp-bike-demand",
    name="Release inference smoke test",
)
def run_release_smoke_test(
    *,
    observed: OclpRun,
    release_manifest_path: Path,
    run_id: str,
) -> _ReleaseSmokeTestResult:
    """Score one fixed request with the model selected by a release manifest.

    This is intentionally a separate lifecycle from training. Its prediction
    Execution consumes the exact ArtifactSet selected by the preceding release
    manifest, so a lineage explorer shows a sibling inference branch connected
    through the released bundle rather than treating service work as part of
    the training Execution tree.
    """

    release = load_release_manifest(release_manifest_path)
    request_artifact = persist_prediction_request(
        request_id=f"{run_id}-request",
        payload=_RELEASE_SMOKE_REQUEST,
    )
    result = predict_bike_demand(
        release,
        request_artifact,
        request_id=f"{run_id}-request",
    )
    evidence = observed.evidence_for(result)
    if any(record.outcome != "pass" for record in evidence):
        raise RuntimeError("release inference smoke test failed")
    return _ReleaseSmokeTestResult(
        execution=observed.execution_for(result),
        response=observed.outputs_for(result)["prediction_response"].reference,
    )


def run_demo(
    *,
    run_id: str,
    fold_count: int = 3,
    temporal_validation_rmse_max: float = 250,
    environment: DemoEnvironment | None = None,
) -> DemoRunResult:
    """Bootstrap and execute the decorated bike-demand lifecycle.

    This narrow application entry point chooses local filesystem destinations,
    a Git source basis, and the optional MLflow observer. The SDK owns the
    active run and lifecycle profile for the real workflow above.
    """

    _validate_run_id(run_id)
    _validate_temporal_validation_rmse_max(temporal_validation_rmse_max)
    environment = environment or DemoEnvironment.default()
    environment.prepare()
    source = source_from_git_checkout(
        environment.project_root,
        path="examples/bike-demand-service/src/bike_demand_service",
    )
    tracker = create_mlflow_tracker(MLflowSettings(environment.mlflow_root))

    with LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.run_root(run_id),
    ) as publisher:
        with tracker.run(f"Bike demand lifecycle — {run_id}"):
            tracker.log_parameters(
                {
                    "run_id": run_id,
                    "temporal_fold_count": fold_count,
                    "temporal_validation_rmse_max": temporal_validation_rmse_max,
                }
            )
            with observe_lifecycle(
                run_bike_lifecycle,
                publisher=publisher,
                run_id=run_id,
                source=source,
            ) as observed:
                lifecycle_result = run_bike_lifecycle(
                    observed=observed,
                    tracker=tracker,
                    run_id=run_id,
                    fold_count=fold_count,
                    temporal_validation_rmse_max=temporal_validation_rmse_max,
                )
    release_smoke_run_id = f"{run_id}-release-smoke"
    with LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.run_root(release_smoke_run_id),
    ) as smoke_publisher:
        with observe_lifecycle(
            run_release_smoke_test,
            publisher=smoke_publisher,
            run_id=release_smoke_run_id,
            source=source,
        ) as observed:
            smoke_result = run_release_smoke_test(
                observed=observed,
                release_manifest_path=Path(lifecycle_result.model_release_manifest_path),
                run_id=release_smoke_run_id,
            )
        records = smoke_publisher.records()
        validate_derivation_graph(records)
        validate_execution_acceptance(records)
        validate_execution_hierarchy(records)

    return DemoRunResult(
        run_id=run_id,
        model_release=lifecycle_result.model_release,
        model_release_manifest=lifecycle_result.model_release_manifest,
        model_release_manifest_path=lifecycle_result.model_release_manifest_path,
        release_smoke_run_id=release_smoke_run_id,
        release_smoke_execution=smoke_result.execution,
        release_smoke_response=smoke_result.response,
        oclp_root=str(environment.oclp_root),
        mlflow_tracking_uri=tracker.tracking_uri,
    )


def _validate_run_id(run_id: str) -> None:
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a non-empty value without whitespace")


def _validate_temporal_validation_rmse_max(value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(
            "temporal_validation_rmse_max must be a finite positive number"
        )

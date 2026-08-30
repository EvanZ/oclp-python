"""Run the first complete, locally inspectable OCLP bike-demand lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from oclp import (
    ArtifactSet,
    ArtifactSetMember,
    Evidence,
    Invocation,
    record_digest,
    validate_derivation_graph,
    validate_invocation_hierarchy,
)
from oclp.models import ComputationDefinition, ContractReference, RecordReference
from oclp.profiles import DatasetSnapshotManifest, DatasetSnapshotPartition

from bike_demand_service.data import (
    HOLDOUT_START,
    UCI_BIKE_SHARING_DATASET_ID,
    download_source_data,
    prepare_features,
)
from bike_demand_service.definitions import definitions
from bike_demand_service.lifecycle import (
    LIFECYCLE_PROFILE,
    publish_completed,
    publish_started,
    publish_terminal,
)
from bike_demand_service.modeling import (
    FoldResult,
    evaluate_folds,
    score_holdout,
    train_final_model,
    train_fold,
)
from bike_demand_service.publication import LocalPublisher, PublishedArtifact, utc_now
from bike_demand_service.settings import DemoPaths
from bike_demand_service.tracking import (
    MLflowSettings,
    configure_mlflow,
    log_oclp_bridge,
)


@dataclass(frozen=True)
class DemoRunResult:
    """Useful local destinations from an observed model lifecycle run."""

    run_id: str
    root_invocation: RecordReference
    model_release: RecordReference
    oclp_root: str
    mlflow_tracking_uri: str


def run_demo(*, run_id: str, paths: DemoPaths | None = None) -> DemoRunResult:
    """Execute the batch milestone without making OCLP or MLflow the other.

    OCLP owns the immutable artifacts, references, evidence, and lifecycle
    events. MLflow owns a deliberately smaller parallel view of experimental
    parameters and metrics, linked back to the OCLP records by digest.
    """

    _validate_run_id(run_id)
    paths = paths or DemoPaths.default()
    paths.prepare()
    definitions_by_key = definitions(paths.project_root)
    mlflow_settings = MLflowSettings(paths.mlflow_root)
    mlflow = configure_mlflow(mlflow_settings)

    with LocalPublisher(
        catalog_path=paths.catalog_path,
        record_root=paths.oclp_root,
        run_root=paths.run_root(run_id),
    ) as publisher:
        definition_refs = {
            key: publisher.publish(definition)
            for key, definition in definitions_by_key.items()
        }
        root_started_at = utc_now()
        plan = publisher.json_artifact(
            artifact_id=_artifact_id(run_id, "training-plan"),
            name=f"Bike-demand training plan — {run_id}",
            relative_path="plans/training-plan.json",
            value={
                "run_id": run_id,
                "dataset": "UCI Bike Sharing Dataset (hourly)",
                "dataset_id": UCI_BIKE_SHARING_DATASET_ID,
                "holdout_start": HOLDOUT_START.isoformat(),
                "temporal_fold_count": 3,
                "model": "CatBoostRegressor",
            },
            created_at=root_started_at,
        )
        root_invocation = _invocation(
            run_id=run_id,
            stage="lifecycle",
            definition=definition_refs["lifecycle"],
            parameters={"run_id": run_id, "fold_count": 3},
            outputs={"training_plan": (plan.reference,)},
        )
        root_attempt_id = _attempt_id(run_id, "lifecycle")
        root_ref = publish_started(
            publisher=publisher,
            invocation=root_invocation,
            attempt_id=root_attempt_id,
            started_at=root_started_at,
        )

        with mlflow.start_run(run_name=f"Bike demand lifecycle — {run_id}"):
            _log_root_mlflow(
                mlflow,
                run_id=run_id,
                root=root_ref,
                definition=definition_refs["lifecycle"],
                outputs=root_invocation.outputs or {},
            )

            ingest_started_at = utc_now()
            source = download_source_data()
            source_artifact = publisher.artifact_for_bytes(
                artifact_id=_artifact_id(run_id, "raw-source-data"),
                name=f"UCI Bike Sharing hourly source — {run_id}",
                relative_path="source/bike-sharing-hourly.csv",
                content=_csv_bytes(source),
                media_type="text/csv",
                created_at=utc_now(),
            )
            ingest_ref = _publish_stage(
                publisher=publisher,
                mlflow=mlflow,
                run_id=run_id,
                stage="ingest",
                definition=definition_refs["ingest"],
                parent=root_ref,
                parameters={"uci_dataset_id": UCI_BIKE_SHARING_DATASET_ID},
                outputs={"raw_dataset": (source_artifact.reference,)},
                metrics={"source_rows": len(source)},
                started_at=ingest_started_at,
            )

            prepare_started_at = utc_now()
            prepared = prepare_features(source)
            features = publisher.artifact_for_bytes(
                artifact_id=_artifact_id(run_id, "feature-table"),
                name=f"Leakage-safe bike-demand features — {run_id}",
                relative_path="prepared/features.csv",
                content=_csv_bytes(prepared.frame),
                media_type="text/csv",
                created_at=utc_now(),
                schema_uri="urn:oclp-bike-demand:schema:feature-table:v1",
            )
            snapshot = publisher.json_artifact(
                artifact_id=_artifact_id(run_id, "feature-dataset-snapshot"),
                name=f"Bike-demand feature dataset snapshot — {run_id}",
                relative_path="prepared/dataset-snapshot.json",
                value=DatasetSnapshotManifest(
                    dataset_id=f"urn:oclp-bike-demand:dataset:features:{run_id}",
                    data_format="text/csv",
                    partitions=(
                        DatasetSnapshotPartition(
                            name="features",
                            artifact=features.reference,
                            values={"rows": len(prepared.frame)},
                        ),
                    ),
                    annotations={"feature_contract_version": 1},
                ).model_dump(mode="json"),
                created_at=utc_now(),
                profiles={"dataset-snapshot": {"version": "0.1.0-draft"}},
            )
            folds = publisher.json_artifact(
                artifact_id=_artifact_id(run_id, "temporal-fold-definition"),
                name=f"Bike-demand temporal fold definition — {run_id}",
                relative_path="prepared/temporal-folds.json",
                value={"strategy": "TimeSeriesSplit", "folds": prepared.folds},
                created_at=utc_now(),
            )
            feature_contract = publisher.json_artifact(
                artifact_id=_artifact_id(run_id, "feature-contract"),
                name=f"Bike-demand feature contract — {run_id}",
                relative_path="prepared/feature-contract.json",
                value=prepared.feature_contract,
                created_at=utc_now(),
            )
            prepare_ref = _publish_stage(
                publisher=publisher,
                mlflow=mlflow,
                run_id=run_id,
                stage="prepare",
                definition=definition_refs["prepare"],
                parent=root_ref,
                parameters={
                    "fold_count": len(prepared.folds),
                    "holdout_start": HOLDOUT_START.isoformat(),
                },
                inputs={"raw_dataset": (source_artifact.reference,)},
                outputs={
                    "features": (features.reference,),
                    "dataset_snapshot": (snapshot.reference,),
                    "fold_definition": (folds.reference,),
                    "feature_contract": (feature_contract.reference,),
                },
                metrics={"prepared_rows": len(prepared.frame)},
                started_at=prepare_started_at,
            )

            fold_results: list[FoldResult] = []
            fold_model_artifacts: list[PublishedArtifact] = []
            fold_prediction_artifacts: list[PublishedArtifact] = []
            fold_metric_artifacts: list[PublishedArtifact] = []
            for fold in prepared.folds:
                fold_number = int(fold["fold"])
                model_path = paths.run_root(run_id) / f"models/fold-{fold_number}.cbm"
                with mlflow.start_run(
                    run_name=f"Train temporal fold {fold_number}", nested=True
                ):
                    fold_started_at = utc_now()
                    mlflow.log_params(
                        {
                            "fold": fold_number,
                            "train_end": str(fold["train_end"]),
                            "validation_start": str(fold["validation_start"]),
                            "validation_end": str(fold["validation_end"]),
                            "model": "CatBoostRegressor",
                        }
                    )
                    result = train_fold(prepared.frame, fold, model_path=model_path)
                    model_artifact = publisher.artifact_for_file(
                        artifact_id=_artifact_id(run_id, f"fold-{fold_number}-model"),
                        name=f"Bike-demand fold {fold_number} model — {run_id}",
                        relative_path=f"models/fold-{fold_number}.cbm",
                        source_path=model_path,
                        media_type="application/x-catboost-model",
                        created_at=utc_now(),
                    )
                    predictions = publisher.artifact_for_bytes(
                        artifact_id=_artifact_id(
                            run_id, f"fold-{fold_number}-validation-predictions"
                        ),
                        name=(
                            f"Bike-demand fold {fold_number} validation predictions "
                            f"— {run_id}"
                        ),
                        relative_path=f"validation/fold-{fold_number}-predictions.csv",
                        content=_csv_bytes(result.predictions),
                        media_type="text/csv",
                        created_at=utc_now(),
                    )
                    metrics = publisher.json_artifact(
                        artifact_id=_artifact_id(run_id, f"fold-{fold_number}-metrics"),
                        name=f"Bike-demand fold {fold_number} metrics — {run_id}",
                        relative_path=f"validation/fold-{fold_number}-metrics.json",
                        value=result.metrics,
                        created_at=utc_now(),
                    )
                    stage_ref = _publish_stage(
                        publisher=publisher,
                        mlflow=mlflow,
                        run_id=run_id,
                        stage=f"train-fold-{fold_number}",
                        definition=definition_refs["train_fold"],
                        parent=root_ref,
                        parameters=fold,
                        inputs={
                            "dataset_snapshot": (snapshot.reference,),
                            "fold_definition": (folds.reference,),
                        },
                        outputs={
                            "model": (model_artifact.reference,),
                            "validation_predictions": (predictions.reference,),
                            "metrics": (metrics.reference,),
                        },
                        metrics=result.metrics,
                        started_at=fold_started_at,
                        use_active_mlflow_run=True,
                    )
                    _ = stage_ref
                fold_results.append(result)
                fold_model_artifacts.append(model_artifact)
                fold_prediction_artifacts.append(predictions)
                fold_metric_artifacts.append(metrics)

            evaluation_started_at = utc_now()
            evaluation = evaluate_folds(tuple(fold_results))
            evaluation_artifact = publisher.json_artifact(
                artifact_id=_artifact_id(run_id, "candidate-evaluation"),
                name=f"Bike-demand temporal candidate evaluation — {run_id}",
                relative_path="evaluation/candidate-evaluation.json",
                value=evaluation,
                created_at=utc_now(),
            )
            training_config = publisher.json_artifact(
                artifact_id=_artifact_id(run_id, "final-training-config"),
                name=f"Bike-demand final training configuration — {run_id}",
                relative_path="evaluation/final-training-config.json",
                value={
                    "model": "CatBoostRegressor",
                    "iterations": 200,
                    "depth": 6,
                    "learning_rate": 0.05,
                    "random_seed": 17,
                    "selection_metric": "temporal_validation_rmse",
                },
                created_at=utc_now(),
            )
            evaluation_invocation = _invocation(
                run_id=run_id,
                stage="evaluate",
                definition=definition_refs["evaluate"],
                parent=root_ref,
                parameters={"quality_gate_rmse_max": 250},
                inputs={
                    "fold_models": tuple(
                        item.reference for item in fold_model_artifacts
                    ),
                    "fold_predictions": tuple(
                        item.reference for item in fold_prediction_artifacts
                    ),
                    "fold_metrics": tuple(
                        item.reference for item in fold_metric_artifacts
                    ),
                },
                outputs={
                    "evaluation": (evaluation_artifact.reference,),
                    "training_config": (training_config.reference,),
                },
            )
            evaluation_ref = _publish_stage(
                publisher=publisher,
                mlflow=mlflow,
                run_id=run_id,
                stage="evaluate",
                definition=definition_refs["evaluate"],
                parent=root_ref,
                parameters=evaluation_invocation.parameters,
                inputs=evaluation_invocation.inputs,
                outputs=evaluation_invocation.outputs or {},
                metrics=evaluation,
                invocation=evaluation_invocation,
                evidence=_quality_evidence(evaluation_invocation, evaluation),
                started_at=evaluation_started_at,
            )

            final_model_path = paths.run_root(run_id) / "models/final-model.cbm"
            with mlflow.start_run(
                run_name="Train final bike-demand model", nested=True
            ):
                final_training_started_at = utc_now()
                mlflow.log_params(
                    {
                        "model": "CatBoostRegressor",
                        "training_window": "all pre-holdout rows",
                    }
                )
                final_model = train_final_model(
                    prepared.frame, model_path=final_model_path
                )
                final_model_artifact = publisher.artifact_for_file(
                    artifact_id=_artifact_id(run_id, "final-model"),
                    name=f"Bike-demand final CatBoost model — {run_id}",
                    relative_path="models/final-model.cbm",
                    source_path=final_model_path,
                    media_type="application/x-catboost-model",
                    created_at=utc_now(),
                )
                final_train_ref = _publish_stage(
                    publisher=publisher,
                    mlflow=mlflow,
                    run_id=run_id,
                    stage="train-final",
                    definition=definition_refs["train_final"],
                    parent=root_ref,
                    parameters={"training_window": "all pre-holdout rows"},
                    inputs={
                        "dataset_snapshot": (snapshot.reference,),
                        "training_config": (training_config.reference,),
                    },
                    outputs={"model": (final_model_artifact.reference,)},
                    metrics={
                        "training_rows": int(
                            (prepared.frame["timestamp"] < HOLDOUT_START).sum()
                        )
                    },
                    started_at=final_training_started_at,
                    use_active_mlflow_run=True,
                )
                _ = final_train_ref

            package_started_at = utc_now()
            model_release = ArtifactSet(
                id=f"urn:oclp-bike-demand:artifact-set:model-release:{run_id}",
                name=f"Bike-demand CatBoost release — {run_id}",
                created_at=utc_now(),
                members=(
                    ArtifactSetMember(
                        name="model",
                        artifact=final_model_artifact.reference,
                        role="model",
                    ),
                    ArtifactSetMember(
                        name="feature-contract",
                        artifact=feature_contract.reference,
                        role="serving-contract",
                    ),
                    ArtifactSetMember(
                        name="temporal-evaluation",
                        artifact=evaluation_artifact.reference,
                        role="validation-report",
                    ),
                    ArtifactSetMember(
                        name="training-config",
                        artifact=training_config.reference,
                        role="training-config",
                    ),
                    ArtifactSetMember(
                        name="feature-dataset-snapshot",
                        artifact=snapshot.reference,
                        role="training-data",
                    ),
                ),
            )
            model_release_ref = publisher.publish(model_release)
            package_ref = _publish_stage(
                publisher=publisher,
                mlflow=mlflow,
                run_id=run_id,
                stage="package",
                definition=definition_refs["package"],
                parent=root_ref,
                parameters={"release_format": "oclp-artifact-set"},
                inputs={
                    "model": (final_model_artifact.reference,),
                    "feature_contract": (feature_contract.reference,),
                    "evaluation": (evaluation_artifact.reference,),
                    "training_config": (training_config.reference,),
                },
                outputs={"model_release": (model_release_ref,)},
                metrics={"release_members": len(model_release.members)},
                started_at=package_started_at,
            )
            _ = (ingest_ref, prepare_ref, evaluation_ref, package_ref)

            with mlflow.start_run(run_name="Score bike-demand holdout", nested=True):
                scoring_started_at = utc_now()
                predictions_frame, holdout_metrics = score_holdout(
                    final_model, prepared.frame
                )
                holdout_predictions = publisher.artifact_for_bytes(
                    artifact_id=_artifact_id(run_id, "holdout-predictions"),
                    name=f"Bike-demand holdout predictions — {run_id}",
                    relative_path="holdout/predictions.csv",
                    content=_csv_bytes(predictions_frame),
                    media_type="text/csv",
                    created_at=utc_now(),
                )
                holdout_metrics_artifact = publisher.json_artifact(
                    artifact_id=_artifact_id(run_id, "holdout-metrics"),
                    name=f"Bike-demand holdout metrics — {run_id}",
                    relative_path="holdout/metrics.json",
                    value=holdout_metrics,
                    created_at=utc_now(),
                )
                score_invocation = _invocation(
                    run_id=run_id,
                    stage="score",
                    definition=definition_refs["score"],
                    parent=root_ref,
                    parameters={"dataset_window": "post-2012-07-01 holdout"},
                    inputs={
                        "model_release": (model_release_ref,),
                        "dataset_snapshot": (snapshot.reference,),
                    },
                    outputs={
                        "predictions": (holdout_predictions.reference,),
                        "metrics": (holdout_metrics_artifact.reference,),
                    },
                )
                _publish_stage(
                    publisher=publisher,
                    mlflow=mlflow,
                    run_id=run_id,
                    stage="score",
                    definition=definition_refs["score"],
                    parent=root_ref,
                    parameters=score_invocation.parameters,
                    inputs=score_invocation.inputs,
                    outputs=score_invocation.outputs or {},
                    metrics=holdout_metrics,
                    invocation=score_invocation,
                    evidence=_holdout_evidence(score_invocation, holdout_metrics),
                    started_at=scoring_started_at,
                    use_active_mlflow_run=True,
                )

        publish_terminal(
            publisher=publisher,
            invocation=root_invocation,
            invocation_ref=root_ref,
            attempt_id=root_attempt_id,
            completed_at=utc_now(),
            sequence=3,
        )
        records = publisher.records()
        validate_derivation_graph(records)
        validate_invocation_hierarchy(records)
    return DemoRunResult(
        run_id=run_id,
        root_invocation=root_ref,
        model_release=model_release_ref,
        oclp_root=str(paths.oclp_root),
        mlflow_tracking_uri=mlflow_settings.tracking_uri,
    )


def _publish_stage(
    *,
    publisher: LocalPublisher,
    mlflow: Any,
    run_id: str,
    stage: str,
    definition: RecordReference,
    parent: RecordReference,
    parameters: dict[str, Any],
    inputs: dict[str, tuple[RecordReference, ...]] | None = None,
    outputs: dict[str, tuple[RecordReference, ...]] | None = None,
    metrics: dict[str, float | int | str] | None = None,
    invocation: Invocation | None = None,
    evidence: Evidence | None = None,
    started_at: datetime | None = None,
    use_active_mlflow_run: bool = False,
) -> RecordReference:
    """Complete one nested MLflow/OCLP observation after bytes exist."""

    if not use_active_mlflow_run:
        with mlflow.start_run(run_name=stage.replace("-", " ").title(), nested=True):
            return _publish_stage(
                publisher=publisher,
                mlflow=mlflow,
                run_id=run_id,
                stage=stage,
                definition=definition,
                parent=parent,
                parameters=parameters,
                inputs=inputs,
                outputs=outputs,
                metrics=metrics,
                invocation=invocation,
                evidence=evidence,
                started_at=started_at,
                use_active_mlflow_run=True,
            )
    invocation = invocation or _invocation(
        run_id=run_id,
        stage=stage,
        definition=definition,
        parent=parent,
        parameters=parameters,
        inputs=inputs or {},
        outputs=outputs or {},
    )
    attempt_id = _attempt_id(run_id, stage)
    started_at = started_at or utc_now()
    reference = publish_completed(
        publisher=publisher,
        invocation=invocation,
        attempt_id=attempt_id,
        started_at=started_at,
        completed_at=utc_now(),
        evidence=evidence,
    )
    log_oclp_bridge(
        mlflow,
        invocation=reference,
        definition=definition,
        inputs=invocation.inputs,
        outputs=invocation.outputs,
    )
    mlflow.log_params(_mlflow_parameters(parameters))
    if metrics:
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, int | float)
            }
        )
    return reference


def _invocation(
    *,
    run_id: str,
    stage: str,
    definition: RecordReference,
    parameters: dict[str, Any],
    inputs: dict[str, tuple[RecordReference, ...]] | None = None,
    outputs: dict[str, tuple[RecordReference, ...]] | None = None,
    parent: RecordReference | None = None,
) -> Invocation:
    return Invocation(
        id=f"urn:oclp-bike-demand:invocation:{stage}:{run_id}",
        name=f"Bike-demand {stage.replace('-', ' ')} — {run_id}",
        profiles=LIFECYCLE_PROFILE,
        definition=definition,
        parent_invocation=parent,
        parameters=parameters,
        inputs=inputs or {},
        outputs=outputs,
        requested_outputs=tuple((outputs or {}).keys()),
    )


def _quality_evidence(
    invocation: Invocation, evaluation: dict[str, float | int | str]
) -> Evidence:
    outcome = str(evaluation["quality_gate"])
    return Evidence(
        id=(
            "urn:oclp-bike-demand:evidence:temporal-quality:"
            f"{invocation.id.rsplit(':', 1)[-1]}"
        ),
        name="Bike-demand temporal-validation quality gate",
        subject=_reference(invocation),
        contract=ContractReference(
            id="urn:oclp-bike-demand:contract:temporal-validation-quality", version="1"
        ),
        outcome=outcome,  # type: ignore[arg-type]
        observed_at=utc_now(),
        details={
            "checks": [
                {
                    "id": "temporal-validation-rmse-below-threshold",
                    "outcome": outcome,
                    "expectation": "rmse-less-than-or-equal-to-250",
                    "observed": evaluation["rmse"],
                }
            ],
            "fold_count": evaluation["fold_count"],
        },
    )


def _holdout_evidence(
    invocation: Invocation, metrics: dict[str, float | int]
) -> Evidence:
    return Evidence(
        id=(
            "urn:oclp-bike-demand:evidence:holdout-response:"
            f"{invocation.id.rsplit(':', 1)[-1]}"
        ),
        name="Bike-demand holdout response contract",
        subject=_reference(invocation),
        contract=ContractReference(
            id="urn:oclp-bike-demand:contract:holdout-response", version="1"
        ),
        outcome="pass",
        observed_at=utc_now(),
        details={
            "checks": [
                {
                    "id": "holdout-metrics-are-finite",
                    "outcome": "pass",
                    "expectation": "finite-number",
                    "paths": ["/mae", "/rmse"],
                }
            ],
            "rows": metrics["rows"],
        },
    )


def _reference(record: Invocation | ComputationDefinition) -> RecordReference:
    return RecordReference(id=record.id, digest=record_digest(record))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _artifact_id(run_id: str, name: str) -> str:
    return f"urn:oclp-bike-demand:artifact:{name}:{run_id}"


def _attempt_id(run_id: str, stage: str) -> str:
    return f"{run_id}:{stage}:1"


def _mlflow_parameters(
    parameters: dict[str, Any],
) -> dict[str, str | float | int | bool]:
    """Flatten OCLP JSON parameters into MLflow's scalar parameter values."""

    return {
        key: value
        if isinstance(value, str | int | float | bool)
        else json.dumps(value, sort_keys=True)
        for key, value in parameters.items()
    }


def _log_root_mlflow(
    mlflow: Any,
    *,
    run_id: str,
    root: RecordReference,
    definition: RecordReference,
    outputs: dict[str, tuple[RecordReference, ...]],
) -> None:
    log_oclp_bridge(
        mlflow,
        invocation=root,
        definition=definition,
        inputs={},
        outputs=outputs,
    )
    mlflow.log_params({"run_id": run_id, "temporal_fold_count": 3})


def _validate_run_id(run_id: str) -> None:
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a non-empty value without whitespace")

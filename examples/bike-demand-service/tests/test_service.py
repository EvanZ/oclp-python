"""Contract tests for release-backed local FastAPI inference."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from fastapi.testclient import TestClient
from oclp import (
    Artifact,
    CatBoostModelArtifact,
    Event,
    Evidence,
    Execution,
    GitSource,
    JsonArtifact,
    OclpRun,
    record_digest,
)
from oclp.catalog.duckdb import DuckdbCatalog
from oclp.models import RecordReference
from oclp.publishing import LocalArtifactPublisher

from bike_demand_service.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    model_features,
)
from bike_demand_service.environment import DemoEnvironment
from bike_demand_service.service import create_app


def test_predict_uses_exact_manifest_model_and_persists_request_response(
    tmp_path: Path,
) -> None:
    """A served request binds the release's model Artifact, request, and response."""

    environment = DemoEnvironment(project_root=tmp_path)
    environment.prepare()
    payload = _prediction_payload()
    model = _fitted_model(payload)
    with LocalArtifactPublisher(
        catalog_path=environment.catalog_path,
        record_root=environment.oclp_root,
        payload_root=environment.run_root("training"),
    ) as publisher:
        published_model = CatBoostModelArtifact().persist(
            publisher=publisher,
            artifact_id="urn:oclp-bike-demand:artifact:test-release-model",
            name="Test released CatBoost model",
            relative_path="training/model.cbm",
            value=model,
            created_at=datetime.now(UTC),
        )
        with OclpRun(
            publisher=publisher,
            namespace="urn:oclp-bike-demand",
            run_id="test-release",
            source=GitSource(
                repository="https://github.com/example/oclp-python.git",
                commit="a" * 40,
            ),
        ) as observed:
            feature_contract = JsonArtifact().handle(
                publisher.json_artifact(
                    artifact_id=(
                        "urn:oclp-bike-demand:artifact:test-release-feature-contract"
                    ),
                    name="Test feature contract",
                    relative_path="training/feature-contract.json",
                    value={"features": list(FEATURE_COLUMNS)},
                    created_at=datetime.now(UTC),
                )
            )
            release = observed.publish_artifact_set(
                key="test-release",
                name="Test bike-demand release",
                members={
                    "model": (CatBoostModelArtifact().handle(published_model), "model"),
                    "feature-contract": (feature_contract, "serving-contract"),
                },
                materialize_manifest=True,
                manifest_name="Test release manifest",
            )
            assert release.manifest is not None
            manifest = release.manifest.path

    app = create_app(release_manifest_path=manifest, environment=environment)
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post("/predict", json=payload)

    assert health.status_code == 200
    assert health.json()["model_release_id"] == release.artifact_set.id
    assert response.status_code == 200
    body = response.json()
    assert body["model_release_id"] == release.artifact_set.id
    assert isinstance(body["prediction"], float)
    assert body["execution_id"].startswith(
        "urn:oclp-bike-demand:execution:predict-bike-demand-request:inference-"
    )

    with DuckdbCatalog(environment.catalog_path) as catalog:
        records = catalog.records()

    execution = next(
        record
        for record in records
        if isinstance(record, Execution) and record.id == body["execution_id"]
    )
    assert execution.inputs["model_release"] == (release.reference,)
    assert len(execution.inputs["prediction_request"]) == 1
    response_reference = execution.outputs["prediction_response"][0]
    assert response_reference.id == body["response_artifact_id"]

    request_artifact = next(
        record
        for record in records
        if isinstance(record, Artifact)
        and record.id == execution.inputs["prediction_request"][0].id
    )
    response_artifact = next(
        record
        for record in records
        if isinstance(record, Artifact) and record.id == response_reference.id
    )
    assert json.loads(_read_file_location(request_artifact)) == payload
    assert json.loads(_read_file_location(response_artifact)) == {
        "model_release_id": body["model_release_id"],
        "prediction": body["prediction"],
        "request_id": body["request_id"],
    }
    evidence = [
        record
        for record in records
        if isinstance(record, Evidence)
        and record.subject
        == RecordReference(id=execution.id, digest=record_digest(execution))
    ]
    assert [(record.name, record.outcome) for record in evidence] == [
        ("Prediction response validation", "pass")
    ]
    events = [
        record
        for record in records
        if isinstance(record, Event) and record.execution == RecordReference(
            id=execution.id, digest=record_digest(execution)
        )
    ]
    events.sort(key=lambda event: event.sequence)
    assert [(event.event_type, event.status) for event in events] == [
        ("execution-started", None),
        ("artifacts-published", None),
        ("evidence-published", None),
        ("execution-terminal", "succeeded"),
    ]


def _prediction_payload() -> dict[str, int | float]:
    return {
        "season": 1,
        "yr": 0,
        "mnth": 1,
        "hr": 8,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.24,
        "atemp": 0.29,
        "hum": 0.81,
        "windspeed": 0.0,
    }


def _fitted_model(payload: dict[str, int | float]) -> CatBoostRegressor:
    """Fit a tiny deterministic model compatible with the real service contract."""

    rows = []
    for offset in range(4):
        row = dict(payload)
        row["hr"] = int(payload["hr"]) + offset
        row["temp"] = float(payload["temp"]) + offset * 0.01
        rows.append(row)
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    model = CatBoostRegressor(
        iterations=4,
        depth=2,
        learning_rate=0.3,
        allow_writing_files=False,
        random_seed=7,
        verbose=False,
    )
    model.fit(
        model_features(frame),
        [100.0, 110.0, 120.0, 130.0],
        cat_features=list(CATEGORICAL_FEATURES),
    )
    return model


def _read_file_location(artifact: Artifact) -> str:
    assert len(artifact.locations) == 1
    return Path(artifact.locations[0].removeprefix("file://")).read_text()

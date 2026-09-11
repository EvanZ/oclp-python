"""Release-backed FastAPI inference with local request-scoped OCLP records.

This deliberately small service proves the central deployment handoff: an HTTP
request consumes a previously published OCLP release ArtifactSet, verifies the
members needed for serving, and scores with its exact CatBoost model Artifact.
Every accepted request becomes a durable JSON Artifact, and every response
becomes an output Artifact of a real ``predict-bike-demand`` Execution in the
local OCLP store.

It is a correctness-first demonstration, rather than a production serving
architecture. In particular, it loads and verifies the release model for each
request and records every request and response locally. Sampling, redaction,
async publication, caching, and operational telemetry can be layered on later
without changing the release-to-Execution contract established here.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from oclp import (
    ArtifactHandle,
    ArtifactSetHandle,
    CatBoostModelArtifact,
    GitSource,
    JsonArtifact,
    OclpRun,
    artifact_set_input,
    capture_git_source_overlay,
    computation,
    evidence,
    json_artifact,
    load_release_manifest,
    output_artifact_id,
    source_from_git_checkout,
)
from oclp.publishing import LocalArtifactPublisher
from pydantic import BaseModel, ConfigDict

from bike_demand_service.data import FEATURE_COLUMNS, model_features
from bike_demand_service.environment import DemoEnvironment


class PredictionRequest(BaseModel):
    """One feature-complete bike-demand scoring request."""

    model_config = ConfigDict(extra="forbid")

    season: int
    yr: int
    mnth: int
    hr: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float


class PredictionResponse(BaseModel):
    """HTTP response paired with a durable OCLP response Artifact."""

    request_id: str
    response_id: str
    prediction: float
    model_release_id: str
    execution_id: str


@json_artifact(
    name="Bike demand prediction request",
    description_from_docstring=True,
)
def persist_prediction_request(*, payload: dict[str, object]) -> dict[str, object]:
    """Persist the accepted HTTP payload as an external input Artifact."""

    return payload


@evidence(name="Prediction response validation", description_from_docstring=True)
def prediction_response_validation(
    prediction_response: dict[str, object],
) -> Literal["pass", "fail", "error"]:
    """Confirm that a produced inference response is safe to return."""

    request_id = prediction_response.get("request_id")
    response_id = prediction_response.get("response_id")
    model_release_id = prediction_response.get("model_release_id")
    prediction = prediction_response.get("prediction")
    if not isinstance(request_id, str) or not request_id:
        return "fail"
    if not isinstance(response_id, str) or not response_id:
        return "fail"
    if not isinstance(model_release_id, str) or not model_release_id:
        return "fail"
    if not isinstance(prediction, (int, float)) or isinstance(prediction, bool):
        return "fail"
    return "pass" if isfinite(float(prediction)) else "fail"


@computation(
    name="Bike demand prediction",
    description_from_docstring=True,
    inputs={
        "model_release": artifact_set_input(
            {
                "model": CatBoostModelArtifact,
                "feature-contract": JsonArtifact,
            }
        ),
        "prediction_request": JsonArtifact,
    },
    outputs={
        "prediction_response": JsonArtifact(
            name="Bike demand prediction response",
            description=(
                "A release-pinned bike-demand prediction returned to the HTTP caller."
            ),
        ),
    },
    requires=(prediction_response_validation,),
)
def predict_bike_demand(
    model_release: ArtifactSetHandle,
    prediction_request: ArtifactHandle,
) -> dict[str, object]:
    """Score one verified request with the exact released ArtifactSet.

    ``model_release`` remains an ArtifactSet handle inside the function rather
    than being reduced to an arbitrary model path. The runtime records the
    exact ArtifactSet UUID reference as the Execution input, verifies that it exposes
    the serving members declared above, and this body materializes only the
    model and feature contract it actually needs.
    """

    try:
        request_payload = json.loads(prediction_request.read_verified_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction request Artifact must contain JSON") from error
    if not isinstance(request_payload, dict):
        raise ValueError("prediction request Artifact must contain a JSON object")

    model = model_release.load_member("model", CatBoostRegressor)
    feature_contract = model_release.load_member("feature-contract", dict[str, object])
    feature_columns = _release_feature_columns(feature_contract)
    frame = pd.DataFrame([request_payload], columns=feature_columns)
    prediction = float(model.predict(model_features(frame))[0])
    if not isfinite(prediction):
        raise ValueError("released model returned a non-finite prediction")
    return {
        "prediction_response": {
            # This is the UUID of the persisted request Artifact, not an
            # application-created correlation label.
            "request_id": prediction_request.artifact.id,
            "response_id": output_artifact_id("prediction_response"),
            "prediction": prediction,
            "model_release_id": model_release.artifact_set.id,
        }
    }


def create_app(
    *,
    release_manifest_path: Path,
    environment: DemoEnvironment | None = None,
) -> FastAPI:
    """Create a service pinned to the model named by ``release_manifest_path``.

    The SDK parses and verifies the manifest once as an immutable ArtifactSet
    selection. Each request then validates the declared serving members and
    reloads their verified payloads through SDK adapters inside
    ``predict_bike_demand``.
    """

    environment = environment or DemoEnvironment.default()
    environment.prepare()
    release = load_release_manifest(release_manifest_path)
    app = FastAPI(
        title="OCLP bike-demand inference demo",
        version="0.1.0",
    )
    app.state.model_release = release

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report the precise release configured for this process."""

        return {
            "status": "ok",
            "model_release_id": release.artifact_set.id,
            "release_manifest": str(release_manifest_path),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Persist, score, and return one release-pinned prediction request."""

        request_storage_key = uuid4().hex
        with LocalArtifactPublisher(
            catalog_path=environment.catalog_path,
            record_root=environment.oclp_root,
            payload_root=environment.inference_root(request_storage_key),
        ) as publisher:
            source = source_from_git_checkout(
                environment.project_root,
                path="examples/bike-demand-service/src/bike_demand_service/service.py",
            )
            if isinstance(source, GitSource) and source.dirty:
                source = capture_git_source_overlay(
                    environment.project_root,
                    source=source,
                    publisher=publisher,
                    name="Bike-demand service source overlay",
                    relative_path=f"source-overlays/{request_storage_key}",
                )
            with OclpRun(
                publisher=publisher,
                source=source,
            ) as observed:
                request_artifact = persist_prediction_request(
                    payload=request.model_dump(mode="json"),
                )
                result = predict_bike_demand(
                    release,
                    request_artifact,
                )
                execution = observed.execution_for(result)
                response_artifact = observed.outputs_for(result)[
                    "prediction_response"
                ]

        response = result["prediction_response"]
        if not isinstance(response, dict):  # pragma: no cover - function contract.
            raise HTTPException(status_code=500, detail="invalid prediction response")
        if response.get("response_id") != response_artifact.artifact.id:
            raise HTTPException(
                status_code=500,
                detail="prediction response does not identify its Artifact",
            )
        return PredictionResponse(
            request_id=str(response["request_id"]),
            response_id=str(response["response_id"]),
            prediction=float(response["prediction"]),
            model_release_id=str(response["model_release_id"]),
            execution_id=execution.id,
        )

    return app


def _release_feature_columns(feature_contract: object) -> tuple[str, ...]:
    """Read and validate the feature schema required by this serving release."""

    if not isinstance(feature_contract, dict):
        raise ValueError("release feature-contract must be a JSON object")
    raw_features = feature_contract.get("features")
    if not isinstance(raw_features, list) or not all(
        isinstance(feature, str) and feature for feature in raw_features
    ):
        raise ValueError("release feature-contract.features must be string names")
    feature_columns = tuple(raw_features)
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "release feature contract is not compatible with this bike-demand "
            "service implementation"
        )
    return feature_columns

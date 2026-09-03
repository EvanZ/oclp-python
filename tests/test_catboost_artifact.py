"""Opt-in tests for the SDK-owned native CatBoost Artifact integration."""

from __future__ import annotations

import pytest

catboost = pytest.importorskip("catboost")
pandas = pytest.importorskip("pandas")

from oclp import (  # noqa: E402
    CatBoostModelArtifact,
    GitSource,
    OclpRun,
    computation,
)
from oclp.models import Execution  # noqa: E402
from oclp.publishing import LocalArtifactPublisher  # noqa: E402


@computation(
    id="urn:example:computation:train-catboost",
    name="Train CatBoost",
    outputs={"model": CatBoostModelArtifact(name="Example CatBoost model")},
)
def train_catboost() -> object:
    """Return the fitted model itself; the SDK owns native file persistence."""

    model = catboost.CatBoostRegressor(
        iterations=2,
        depth=2,
        learning_rate=0.2,
        allow_writing_files=False,
        verbose=False,
    )
    features = pandas.DataFrame({"temperature": [1.0, 2.0, 3.0, 4.0]})
    return model.fit(features, [1.0, 1.5, 2.5, 4.0])


@computation(
    id="urn:example:computation:score-catboost",
    name="Score CatBoost",
    inputs={"model": CatBoostModelArtifact},
)
def score_catboost(model: catboost.CatBoostRegressor) -> float:
    """Require the SDK to load the verified native model before this call."""

    features = pandas.DataFrame({"temperature": [2.5]})
    return float(model.predict(features)[0])


def test_catboost_artifact_round_trips_through_sdk_owned_persistence(tmp_path) -> None:
    source = GitSource(
        repository="https://github.com/example/models.git",
        commit="a" * 40,
    )
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="catboost-round-trip",
            source=source,
        ) as observed:
            trained = train_catboost()
            model_artifact = observed.outputs_for(trained)["model"]
            prediction = score_catboost(model_artifact)
        records = publisher.records()

    assert isinstance(prediction, float)
    assert model_artifact.artifact.media_type == "application/x-catboost-model"
    assert model_artifact.path.suffix == ".cbm"
    score_execution = next(
        record
        for record in records
        if isinstance(record, Execution)
        and record.id == "urn:example:execution:score-catboost:catboost-round-trip"
    )
    assert score_execution.inputs == {"model": (model_artifact.reference,)}

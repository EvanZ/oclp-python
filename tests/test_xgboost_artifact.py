"""Opt-in tests for the SDK-owned native XGBoost Artifact integration."""

from __future__ import annotations

import pytest

try:
    import xgboost
except Exception as error:  # pragma: no cover - platform optional-runtime path.
    pytest.skip(
        f"XGBoost runtime is unavailable: {error}",
        allow_module_level=True,
    )
pandas = pytest.importorskip("pandas")

from oclp import (  # noqa: E402
    GitSource,
    OclpRun,
    XGBoostModelArtifact,
    computation,
)
from oclp.models import Execution  # noqa: E402
from oclp.publishing import LocalArtifactPublisher  # noqa: E402


@computation(
    id="urn:example:computation:train-xgboost",
    name="Train XGBoost",
    outputs={"model": XGBoostModelArtifact(name="Example XGBoost model")},
)
def train_xgboost() -> object:
    """Return the fitted model itself; the SDK owns UBJSON persistence."""

    model = xgboost.XGBRegressor(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.2,
        n_jobs=1,
        random_state=7,
    )
    features = pandas.DataFrame({"temperature": [1.0, 2.0, 3.0, 4.0]})
    return model.fit(features, [1.0, 1.5, 2.5, 4.0])


@computation(
    id="urn:example:computation:score-xgboost",
    name="Score XGBoost",
    inputs={"model": XGBoostModelArtifact},
)
def score_xgboost(model: xgboost.XGBRegressor) -> float:
    """Require the SDK to load the verified native model before this call."""

    features = pandas.DataFrame({"temperature": [2.5]})
    return float(model.predict(features)[0])


def test_xgboost_artifact_round_trips_through_sdk_owned_persistence(tmp_path) -> None:
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
            run_id="xgboost-round-trip",
            source=source,
        ) as observed:
            trained = train_xgboost()
            model_artifact = observed.outputs_for(trained)["model"]
            prediction = score_xgboost(model_artifact)
        records = publisher.records()

    assert isinstance(prediction, float)
    assert model_artifact.artifact.media_type == "application/x-xgboost-ubjson"
    assert model_artifact.path.suffix == ".ubj"
    score_execution = next(
        record
        for record in records
        if isinstance(record, Execution)
        and record.id == "urn:example:execution:score-xgboost:xgboost-round-trip"
    )
    assert score_execution.inputs == {"model": (model_artifact.reference,)}

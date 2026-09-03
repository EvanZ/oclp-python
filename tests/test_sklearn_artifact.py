"""Opt-in tests for the SDK-owned sklearn ``skops`` Artifact integration."""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")
pytest.importorskip("skops.io")

from sklearn.linear_model import LinearRegression  # noqa: E402

from oclp import (  # noqa: E402
    GitSource,
    OclpRun,
    SklearnModelArtifact,
    computation,
)
from oclp.models import Execution  # noqa: E402
from oclp.publishing import LocalArtifactPublisher  # noqa: E402


@computation(
    id="urn:example:computation:train-sklearn",
    name="Train sklearn",
    outputs={"model": SklearnModelArtifact(name="Example sklearn model")},
)
def train_sklearn() -> LinearRegression:
    """Return the fitted estimator; the SDK owns skops materialization."""

    return LinearRegression().fit([[1.0], [2.0], [3.0]], [1.0, 1.5, 2.5])


@computation(
    id="urn:example:computation:score-sklearn",
    name="Score sklearn",
    inputs={"model": SklearnModelArtifact},
)
def score_sklearn(model: LinearRegression) -> float:
    """Require the SDK to load the verified estimator before this call."""

    return float(model.predict([[2.5]])[0])


def test_sklearn_artifact_round_trips_through_sdk_owned_persistence(tmp_path) -> None:
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
            run_id="sklearn-round-trip",
            source=source,
        ) as observed:
            trained = train_sklearn()
            model_artifact = observed.outputs_for(trained)["model"]
            prediction = score_sklearn(model_artifact)
        records = publisher.records()

    assert isinstance(prediction, float)
    assert model_artifact.artifact.media_type == "application/x-skops"
    assert model_artifact.path.suffix == ".skops"
    score_execution = next(
        record
        for record in records
        if isinstance(record, Execution)
        and record.id == "urn:example:execution:score-sklearn:sklearn-round-trip"
    )
    assert score_execution.inputs == {"model": (model_artifact.reference,)}

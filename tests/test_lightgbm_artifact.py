"""Opt-in tests for the SDK-owned native LightGBM Artifact integration."""

from __future__ import annotations

import pytest

try:
    import lightgbm
except Exception as error:  # pragma: no cover - platform optional-runtime path.
    pytest.skip(
        f"LightGBM runtime is unavailable: {error}",
        allow_module_level=True,
    )

from oclp import (  # noqa: E402
    GitSource,
    LightGBMModelArtifact,
    OclpRun,
    computation,
)
from oclp.models import Execution  # noqa: E402
from oclp.publishing import LocalArtifactPublisher  # noqa: E402


@computation(
    id="urn:example:computation:train-lightgbm",
    name="Train LightGBM",
    outputs={"model": LightGBMModelArtifact(name="Example LightGBM model")},
)
def train_lightgbm() -> lightgbm.Booster:
    """Return the native Booster; the SDK owns model-file persistence."""

    dataset = lightgbm.Dataset([[1.0], [2.0], [3.0], [4.0]], label=[1.0, 1.5, 2.5, 4.0])
    return lightgbm.train(
        {"objective": "regression", "verbosity": -1, "num_threads": 1},
        dataset,
        num_boost_round=2,
    )


@computation(
    id="urn:example:computation:score-lightgbm",
    name="Score LightGBM",
    inputs={"model": LightGBMModelArtifact},
)
def score_lightgbm(model: lightgbm.Booster) -> float:
    """Require the SDK to load the verified native model before this call."""

    return float(model.predict([[2.5]])[0])


def test_lightgbm_artifact_round_trips_through_sdk_owned_persistence(tmp_path) -> None:
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
            source=source,
        ) as observed:
            trained = train_lightgbm()
            model_artifact = observed.outputs_for(trained)["model"]
            prediction = score_lightgbm(model_artifact)
        records = publisher.records()

    assert isinstance(prediction, float)
    assert model_artifact.artifact.media_type == "application/x-lightgbm-model"
    assert model_artifact.path.suffix == ".txt"
    score_execution = next(
        record
        for record in records
        if isinstance(record, Execution) and record.name == "Score LightGBM"
    )
    assert score_execution.inputs == {"model": (model_artifact.reference,)}

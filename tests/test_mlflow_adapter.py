"""Contract tests for the optional SDK-owned MLflow mirror."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from oclp import (
    BytesArtifact,
    Event,
    GitSource,
    JsonArtifact,
    MlflowAdapter,
    MlflowModelRegistration,
    RunArtifactSet,
    computation,
    observe_run,
    run,
)
from oclp.publishing import LocalArtifactPublisher


@computation(
    name="Train adapter test model",
    outputs={
        "model": BytesArtifact(
            name="Test CatBoost model",
            media_type="application/x-catboost-model",
            suffix="cbm",
        ),
    },
)
def train_adapter_test_model(*, depth: int) -> bytes:
    return f"depth={depth}".encode()


@computation(
    name="Adapter diagnostic computation",
    outputs={"result": JsonArtifact(name="Adapter result")},
)
def computation_with_adapter_failure() -> dict[str, bool]:
    return {"ok": True}


class _FakeMlflowClient:
    def __init__(self, **_kwargs: object) -> None:
        self.created_experiments: list[tuple[str, str | None]] = []
        self.registered_models: list[str] = []
        self.model_versions: list[dict[str, object]] = []

    def get_experiment_by_name(self, _name: str) -> None:
        return None

    def create_experiment(self, name: str, artifact_location: str | None = None) -> str:
        self.created_experiments.append((name, artifact_location))
        return "experiment-id"

    def get_registered_model(self, _name: str) -> None:
        raise LookupError("not registered")

    def create_registered_model(self, name: str) -> None:
        self.registered_models.append(name)

    def create_model_version(self, **kwargs: object) -> None:
        self.model_versions.append(kwargs)


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.tags: dict[str, str] = {}
        self.records: dict[str, object] = {}
        self.parameters: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[tuple[str, str | None]] = []
        self.end_statuses: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, *, run_name: str | None) -> SimpleNamespace:
        assert run_name == "MLflow adapter workflow"
        return SimpleNamespace(info=SimpleNamespace(run_id="mlflow-run-id"))

    def set_tags(self, values: dict[str, str]) -> None:
        self.tags.update(values)

    def set_tag(self, name: str, value: str) -> None:
        self.tags[name] = value

    def log_dict(self, value: object, path: str) -> None:
        self.records[path] = value

    def log_params(self, values: dict[str, str]) -> None:
        self.parameters.update(values)

    def log_metrics(self, values: dict[str, float]) -> None:
        self.metrics.update(values)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def end_run(self, *, status: str) -> None:
        self.end_statuses.append(status)


@pytest.fixture
def fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeMlflow, _FakeMlflowClient]:
    """Install an in-memory optional dependency without importing MLflow."""

    mlflow = _FakeMlflow()
    client = _FakeMlflowClient()
    module = ModuleType("mlflow")
    module.set_tracking_uri = mlflow.set_tracking_uri  # type: ignore[attr-defined]
    module.set_experiment = mlflow.set_experiment  # type: ignore[attr-defined]
    module.start_run = mlflow.start_run  # type: ignore[attr-defined]
    module.set_tags = mlflow.set_tags  # type: ignore[attr-defined]
    module.set_tag = mlflow.set_tag  # type: ignore[attr-defined]
    module.log_dict = mlflow.log_dict  # type: ignore[attr-defined]
    module.log_params = mlflow.log_params  # type: ignore[attr-defined]
    module.log_metrics = mlflow.log_metrics  # type: ignore[attr-defined]
    module.log_artifact = mlflow.log_artifact  # type: ignore[attr-defined]
    module.end_run = mlflow.end_run  # type: ignore[attr-defined]
    tracking = ModuleType("mlflow.tracking")
    tracking.MlflowClient = lambda **_kwargs: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    return mlflow, client


def test_mlflow_adapter_mirrors_records_models_and_explicit_registration(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    mlflow, client = fake_mlflow
    adapter = MlflowAdapter(
        experiment_name="oclp-tests",
        tracking_uri="sqlite:///test.db",
        model_registration=MlflowModelRegistration(
            artifact_name="Test CatBoost model",
            registered_model_name="oclp-test-model",
        ),
    )

    @run(
        name="MLflow adapter workflow",
        artifact_sets=(
            RunArtifactSet(
                name="Adapter model release",
                members={"model": (train_adapter_test_model.output("model"), "model")},
            ),
        ),
        adapters=(adapter,),
    )
    def workflow() -> None:
        train_adapter_test_model(depth=6)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with observe_run(
            workflow,
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/adapter-test.git",
                commit="a" * 40,
            ),
        ) as observed:
            assert observed.adapters[0] is not adapter
            workflow()

    assert mlflow.tracking_uri == "sqlite:///test.db"
    assert mlflow.experiment_name == "oclp-tests"
    assert mlflow.tags["oclp.sdk.adapter"] == "mlflow"
    assert "oclp.run.id" in mlflow.tags
    assert "oclp.execution.id" in mlflow.tags
    assert any(path.startswith("oclp/records/execution/") for path in mlflow.records)
    assert any(path.startswith("oclp/records/computation/") for path in mlflow.records)
    assert any(path.startswith("oclp/records/artifact_set/") for path in mlflow.records)
    assert mlflow.parameters["Train-adapter-test-model.depth"] == "6"
    assert len(mlflow.artifacts) == 1
    assert client.registered_models == ["oclp-test-model"]
    assert client.model_versions[0]["run_id"] == "mlflow-run-id"
    assert mlflow.end_statuses == ["FINISHED"]


def test_non_strict_adapter_failure_becomes_an_oclp_diagnostic_event(
    tmp_path: Path,
) -> None:
    class BrokenAdapter:
        strict = False

        def on_run_start(self, _observed: object) -> None:
            raise RuntimeError("MLflow is unavailable")

    @run(name="Adapter diagnostic workflow", adapters=(BrokenAdapter(),))
    def workflow() -> None:
        computation_with_adapter_failure()

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with observe_run(
            workflow,
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/adapter-test.git",
                commit="a" * 40,
            ),
        ):
            workflow()
        records = publisher.records()

    diagnostics = [
        record
        for record in records
        if isinstance(record, Event) and record.event_type == "adapter-failed"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].diagnostic is not None
    assert diagnostics[0].diagnostic.stage == "integration"
    assert diagnostics[0].diagnostic.message == "MLflow is unavailable"

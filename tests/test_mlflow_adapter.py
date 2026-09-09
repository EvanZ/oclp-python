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
    MlflowMetrics,
    MlflowModelRegistration,
    artifact_set,
    computation,
    mlflow,
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


@artifact_set(
    name="Adapter model release",
    output_port="model",
    role="model",
)
@computation(
    name="Train adapter release model",
    outputs={
        "model": BytesArtifact(
            name="Adapter release model",
            media_type="application/x-catboost-model",
            suffix="cbm",
        ),
    },
)
def train_adapter_release_model(*, depth: int) -> bytes:
    return f"depth={depth}".encode()


@computation(
    name="Adapter diagnostic computation",
    outputs={"result": JsonArtifact(name="Adapter result")},
)
def computation_with_adapter_failure() -> dict[str, bool]:
    return {"ok": True}


@mlflow(
    metrics=(
        MlflowMetrics(
            output_port="metrics",
            prefix="fold",
            dimensions=("fold_number",),
        ),
    ),
)
@computation(
    name="Publish fold metrics",
    outputs={"metrics": JsonArtifact(name="Fold metrics")},
)
def publish_fold_metrics(*, fold_number: int) -> dict[str, object]:
    return {
        "metrics": {
            "rmse": float(fold_number),
            "rows": fold_number * 10,
            "label": "not-a-metric",
            "nested": {"ignored": True},
            "passed": True,
        }
    }


@mlflow(metrics=(MlflowMetrics(output_port="metrics", prefix="fold"),))
@computation(
    name="Publish undimensioned fold metrics",
    outputs={"metrics": JsonArtifact(name="Undimensioned fold metrics")},
)
def publish_undimensioned_fold_metrics(*, fold_number: int) -> dict[str, object]:
    return {"metrics": {"rmse": float(fold_number), "rows": fold_number * 10}}


@mlflow(payloads=("report",))
@computation(
    name="Publish adapter report",
    outputs={"report": JsonArtifact(name="Adapter report")},
)
def publish_adapter_report() -> dict[str, object]:
    return {"report": {"status": "ready"}}


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
        self.run_names: list[str | None] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.end_statuses: list[str] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(self, *, run_name: str | None) -> SimpleNamespace:
        self.run_names.append(run_name)
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
            artifact_name="Adapter release model",
            registered_model_name="oclp-test-model",
        ),
    )

    @run(
        name="MLflow adapter workflow",
        adapters=(adapter,),
    )
    def workflow() -> None:
        train_adapter_release_model(depth=6)

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
    assert len(mlflow.parameters) == 1
    parameter_name, parameter_value = next(iter(mlflow.parameters.items()))
    assert parameter_name.startswith("Train-adapter-release-model.")
    assert parameter_name.endswith(".depth")
    assert parameter_value == "6"
    assert len(mlflow.artifacts) == 1
    assert client.registered_models == ["oclp-test-model"]
    assert client.model_versions[0]["run_id"] == "mlflow-run-id"
    assert mlflow.run_names == ["MLflow adapter workflow"]
    assert mlflow.end_statuses == ["FINISHED"]


def test_mlflow_adapter_scopes_same_named_payloads_by_artifact_id(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    """Repeated folds must not overwrite a same-named model in MLflow."""

    mlflow, _client = fake_mlflow
    adapter = MlflowAdapter(experiment_name="oclp-tests")

    @run(name="Repeated model payloads", adapters=(adapter,))
    def workflow() -> None:
        train_adapter_test_model(depth=4)
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
        ):
            workflow()

    destinations = [destination for _path, destination in mlflow.artifacts]
    assert len(destinations) == 2
    assert len(set(destinations)) == 2
    assert all(
        destination is not None
        and destination.startswith("oclp/payloads/Test-CatBoost-model/")
        for destination in destinations
    )


def test_mlflow_declaration_uploads_only_its_selected_extra_payload(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    """Non-model payloads are opt-in through the local output-port policy."""

    mlflow_client, _client = fake_mlflow
    adapter = MlflowAdapter(experiment_name="oclp-tests")

    @run(name="Selected report payload", adapters=(adapter,))
    def workflow() -> None:
        publish_adapter_report()

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

    assert len(mlflow_client.artifacts) == 1
    assert mlflow_client.artifacts[0][1] is not None
    assert mlflow_client.artifacts[0][1].startswith("oclp/payloads/Adapter-report/")


def test_mlflow_declaration_is_inert_without_an_adapter(tmp_path: Path) -> None:
    """A local projection declaration never requires MLflow at runtime."""

    @run(name="No MLflow adapter")
    def workflow() -> None:
        publish_fold_metrics(fold_number=1)

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
        assert publisher.records()


def test_mlflow_adapter_extracts_selected_json_output_metrics(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    mlflow, _client = fake_mlflow
    adapter = MlflowAdapter(
        experiment_name="oclp-tests", tracking_uri="sqlite:///test.db"
    )

    @run(name="MLflow metrics workflow", adapters=(adapter,))
    def workflow() -> None:
        publish_fold_metrics(fold_number=1)
        publish_fold_metrics(fold_number=2)

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

    assert mlflow.metrics == {
        "fold.fold_number-1.rmse": 1.0,
        "fold.fold_number-1.rows": 10.0,
        "fold.fold_number-2.rmse": 2.0,
        "fold.fold_number-2.rows": 20.0,
    }
    assert sorted(mlflow.parameters.values()) == ["1", "2"]
    assert mlflow.end_statuses == ["FINISHED"]


def test_mlflow_metric_output_collision_is_diagnostic_unless_strict(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    mlflow, _client = fake_mlflow
    adapter = MlflowAdapter(
        experiment_name="oclp-tests", tracking_uri="sqlite:///test.db"
    )

    @run(name="Non-strict MLflow collision workflow", adapters=(adapter,))
    def workflow() -> None:
        publish_undimensioned_fold_metrics(fold_number=1)
        publish_undimensioned_fold_metrics(fold_number=2)

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

    assert mlflow.metrics == {"fold.rmse": 1.0, "fold.rows": 10.0}
    diagnostics = [
        record
        for record in records
        if isinstance(record, Event) and record.event_type == "adapter-failed"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0].diagnostic is not None
    assert "would collide" in diagnostics[0].diagnostic.message


def test_strict_mlflow_metric_output_collision_fails_the_workflow(
    tmp_path: Path, fake_mlflow: tuple[_FakeMlflow, _FakeMlflowClient]
) -> None:
    mlflow, _client = fake_mlflow
    adapter = MlflowAdapter(
        experiment_name="oclp-tests", tracking_uri="sqlite:///test.db", strict=True
    )

    @run(name="Strict MLflow collision workflow", adapters=(adapter,))
    def workflow() -> None:
        publish_undimensioned_fold_metrics(fold_number=1)
        publish_undimensioned_fold_metrics(fold_number=2)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with pytest.raises(ValueError, match="would collide"):
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

    assert mlflow.end_statuses == ["FAILED"]
    assert sum(isinstance(record, Event) for record in records) >= 6


def test_mlflow_metric_output_requires_declared_json_output() -> None:
    with pytest.raises(ValueError, match="application/json"):
        mlflow(metrics=(MlflowMetrics(output_port="model"),))(
            train_adapter_test_model
        )


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

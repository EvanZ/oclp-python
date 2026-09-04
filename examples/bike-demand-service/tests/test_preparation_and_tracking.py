"""Fast contract tests for the demo's local preparation and tracking layers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pandas as pd
import pytest
from oclp import (
    Artifact,
    ArtifactHandle,
    ArtifactIntegrityError,
    CsvArtifact,
    GitSource,
    JsonArtifact,
    OclpRun,
    artifact_type,
    computation,
    computation_input_artifact_types,
    computation_record,
    computation_template,
)
from oclp.models import Execution, PortDefinition, RecordReference
from oclp.publishing import LocalArtifactPublisher

import bike_demand_service.data as data_module
from bike_demand_service.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    download_source_artifact,
    download_source_csv,
    prepare_features,
)
from bike_demand_service.mlflow import (
    MLflowSettings,
    create_mlflow_tracker,
)
from bike_demand_service.modeling import (
    create_training_plan,
    evaluate_folds,
    score_holdout,
    temporal_validation_quality,
)


@computation(
    id="urn:oclp-bike-demand:test-computation:inspect-source-representation",
    name="Test source representation adapter",
    input_ports=(
        PortDefinition(
            name="source_snapshot",
            media_types=(
                "text/csv",
                "application/vnd.apache.parquet",
                "application/json",
            ),
        ),
    ),
)
def _inspect_source_representation(source_snapshot: pd.DataFrame) -> dict[str, object]:
    """Test that each durable source form reaches a typed pandas boundary."""

    normalized = data_module._normalize_source_frame(source_snapshot)
    return {
        "columns": list(normalized.columns),
        "row_count": int(len(normalized)),
        "first_timestamp": data_module._timestamp_text(
            normalized[data_module.TIMESTAMP_COLUMN].min()
        ),
        "last_timestamp": data_module._timestamp_text(
            normalized[data_module.TIMESTAMP_COLUMN].max()
        ),
        "target_sum": float(normalized[TARGET_COLUMN].sum()),
    }


def test_preparation_excludes_target_derived_fields_and_creates_temporal_folds() -> (
    None
):
    source = _source_rows(20)

    prepared = prepare_features(source, {"temporal_fold_count": 3})

    assert tuple(prepared["features"].columns) == (
        "timestamp",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
    assert len(prepared["fold_definition"]["folds"]) == 3
    assert prepared["features"]["timestamp"].is_monotonic_increasing
    assert set(prepared["feature_contract"]["excluded_source_columns"]) == {
        "casual",
        "dteday",
        "instant",
        "registered",
    }


def test_local_artifact_publisher_writes_a_payload_and_canonical_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "oclp"
    run_root = tmp_path / "run"
    with LocalArtifactPublisher(
        catalog_path=root / "catalog.duckdb",
        record_root=root,
        payload_root=run_root,
    ) as publisher:
        published = publisher.artifact_for_bytes(
            name="Test payload",
            relative_path="payload.txt",
            content=b"bike demand\n",
            media_type="text/plain",
            created_at=datetime.now(UTC),
        )
        records = publisher.records()

    assert published.path.read_bytes() == b"bike demand\n"
    assert len(records) == 1
    assert isinstance(records[0], Artifact)
    from oclp import record_digest

    digest = record_digest(published.artifact)
    assert (root / "artifact" / digest.value[:2] / f"{digest.value}.json").exists()


def test_csv_artifact_decorator_requires_an_active_run() -> None:
    with pytest.raises(RuntimeError, match="require an active OclpRun"):
        download_source_csv()


def test_artifact_decorated_ingest_persists_its_returned_dataframe_as_source_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_rows = _source_rows(2)
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=source_rows.drop(columns=[TARGET_COLUMN]),
            targets=source_rows[[TARGET_COLUMN]],
        )
    )
    monkeypatch.setattr(data_module, "fetch_ucirepo", lambda *, id: dataset)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ) as observed:
            source_snapshot = download_source_csv()
            with pytest.raises(ValueError, match="no Execution"):
                observed.execution_for(source_snapshot)
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ) as observed:
            repeated_snapshot = download_source_csv()
        records = publisher.records()

    assert isinstance(source_snapshot, ArtifactHandle)
    assert source_snapshot.path.read_text().startswith("instant,dteday,season")
    assert source_snapshot.reference != repeated_snapshot.reference
    assert source_snapshot.artifact.digest == repeated_snapshot.artifact.digest
    assert source_snapshot.artifact.media_type == "text/csv"
    assert sum(isinstance(record, Artifact) for record in records) == 2
    assert all(not isinstance(record, Execution) for record in records)
    template = artifact_type(download_source_csv)
    assert isinstance(template, CsvArtifact)
    assert template.index is False
    assert template.lineterminator == "\n"


def test_source_factory_adapts_csv_parquet_and_table_json_to_equivalent_frames(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_rows = _source_rows(20)
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=source_rows.drop(columns=[TARGET_COLUMN]),
            targets=source_rows[[TARGET_COLUMN]],
        )
    )
    monkeypatch.setattr(data_module, "fetch_ucirepo", lambda *, id: dataset)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ):
            source_artifacts = {
                storage_format: download_source_artifact(storage_format)
                for storage_format in ("csv", "parquet", "json")
            }
            summaries = {
                storage_format: _inspect_source_representation(source_artifact)
                for storage_format, source_artifact in source_artifacts.items()
            }
        records = publisher.records()

    assert all(
        isinstance(source_artifact, ArtifactHandle)
        for source_artifact in source_artifacts.values()
    )
    assert (
        len(
            {
                source_artifact.artifact.digest.value
                for source_artifact in source_artifacts.values()
            }
        )
        == 3
    )
    assert len({repr(summary) for summary in summaries.values()}) == 1

    inspections = [
        record
        for record in records
        if isinstance(record, Execution)
        and record.computation.id
        == computation_record(
            _inspect_source_representation,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ).id
    ]
    assert {execution.inputs["source_snapshot"][0] for execution in inspections} == {
        source_artifact.reference for source_artifact in source_artifacts.values()
    }


def test_preparation_binds_and_adapts_the_csv_source_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_rows = _source_rows(20)
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=source_rows.drop(columns=[TARGET_COLUMN]),
            targets=source_rows[[TARGET_COLUMN]],
        )
    )
    monkeypatch.setattr(data_module, "fetch_ucirepo", lambda *, id: dataset)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ) as observed:
            source_snapshot = download_source_csv()
            assert isinstance(source_snapshot, ArtifactHandle)
            training_plan = create_training_plan(
                materialization_id="canonical-source",
                fold_count=3,
            )

            prepared = prepare_features(source_snapshot, training_plan)
            preparation_ref = observed.execution_for(prepared)

        execution = next(
            record
            for record in publisher.records()
            if isinstance(record, Execution) and record.id == preparation_ref.id
        )

    assert execution.inputs["source_snapshot"] == (source_snapshot.reference,)
    assert execution.inputs["training_plan"] == (training_plan.reference,)
    assert computation_input_artifact_types(prepare_features) == {
        "source_snapshot": CsvArtifact,
        "training_plan": JsonArtifact,
    }


def test_preparation_rejects_an_artifact_handle_of_the_wrong_representation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_rows = _source_rows(20)
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=source_rows.drop(columns=[TARGET_COLUMN]),
            targets=source_rows[[TARGET_COLUMN]],
        )
    )
    monkeypatch.setattr(data_module, "fetch_ucirepo", lambda *, id: dataset)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ):
            source_snapshot = download_source_artifact("json")
            training_plan = create_training_plan(
                materialization_id="wrong-source-representation",
                fold_count=3,
            )

            with pytest.raises(TypeError, match=r"requires CsvArtifact \(text/csv\)"):
                prepare_features(source_snapshot, training_plan)


def test_csv_artifact_input_rejects_tampered_payload_before_computation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_rows = _source_rows(20)
    dataset = SimpleNamespace(
        data=SimpleNamespace(
            features=source_rows.drop(columns=[TARGET_COLUMN]),
            targets=source_rows[[TARGET_COLUMN]],
        )
    )
    monkeypatch.setattr(data_module, "fetch_ucirepo", lambda *, id: dataset)

    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ):
            source_snapshot = download_source_csv()
            training_plan = create_training_plan(
                materialization_id="tampered-source",
                fold_count=3,
            )
            source_snapshot.path.write_text("tampered\n")

            with pytest.raises(ArtifactIntegrityError, match="expected sha256"):
                prepare_features(source_snapshot, training_plan)


def test_training_plan_is_an_acquired_json_artifact(tmp_path: Path) -> None:
    root = tmp_path / "oclp"
    with LocalArtifactPublisher(
        catalog_path=root / "catalog.duckdb",
        record_root=root,
        payload_root=tmp_path / "run",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=GitSource(
                repository="https://github.com/example/bike-demand.git",
                commit="a" * 40,
            ),
        ) as observed:
            plan = create_training_plan(
                materialization_id="bike-demand-test", fold_count=3
            )
            with pytest.raises(ValueError, match="no Execution"):
                observed.execution_for(plan)

    from uuid import UUID

    assert UUID(plan.artifact.id).version == 4
    assert plan.path.suffix == ".json"
    assert plan.artifact.media_type == "application/json"


def test_mlflow_bridge_tags_one_run_with_oclp_references(tmp_path: Path) -> None:
    settings = MLflowSettings(root=tmp_path / "mlflow")
    tracker = create_mlflow_tracker(settings)
    execution = _reference("urn:oclp-bike-demand:execution:test")
    computation = _reference("urn:oclp-bike-demand:computation:test")
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "oclp" / "catalog.duckdb",
        record_root=tmp_path / "oclp" / "records",
        payload_root=tmp_path / "oclp" / "payloads",
    ) as publisher:
        payload = JsonArtifact(name="Bridge payload").handle(
            publisher.json_artifact(
                artifact_id=_id("artifact:bridge-payload"),
                name="Bridge payload",
                relative_path="bridge-payload.json",
                value={"rows": 12},
                created_at=datetime.now(UTC),
            )
        )

    with tracker.run("bridge test"):
        run_id = tracker.active_run_id()
        tracker.attach_execution(
            execution=execution,
            computation=computation,
            inputs={},
            outputs={},
            artifacts={"report": payload},
        )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["oclp.execution.id"] == execution.id
    assert run.data.tags["oclp.computation.id"] == computation.id
    mirrored = Path(
        client.download_artifacts(
            run_id,
            "oclp/outputs/report/bridge-payload.json",
            dst_path=str(tmp_path / "download"),
        )
    )
    assert mirrored.read_text() == payload.path.read_text()


def test_quality_checked_computations_declare_required_evidence_evaluators() -> None:
    evaluation = computation_template(evaluate_folds)
    scoring = computation_template(score_holdout)

    assert evaluation.required_evaluators[0].__oclp_evidence_template__.name == (
        "Temporal validation quality"
    )
    assert scoring.required_evaluators[0].__oclp_evidence_template__.name == (
        "Holdout response validation"
    )


def test_temporal_validation_quality_uses_the_persisted_threshold() -> None:
    evaluation = {"rmse": 96.13, "temporal_validation_rmse_max": 1}

    assert temporal_validation_quality(evaluation) == "fail"
    assert temporal_validation_quality(
        {**evaluation, "temporal_validation_rmse_max": 100}
    ) == "pass"


def _source_rows(count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "instant": index + 1,
                "dteday": (pd.Timestamp("2011-01-01") + timedelta(days=index)).date(),
                "season": 1,
                "yr": 0,
                "mnth": 1,
                "hr": index % 24,
                "holiday": 0,
                "weekday": index % 7,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.2,
                "atemp": 0.3,
                "hum": 0.5,
                "windspeed": 0.1,
                "casual": 10,
                "registered": 30,
                "cnt": 40,
            }
        )
    return pd.DataFrame(rows)


def _reference(identifier: str) -> RecordReference:
    return RecordReference(id=_id(identifier))


def _id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"test:bike-demand:{name}"))

"""Fast contract tests for the demo's local preparation and tracking layers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from oclp import Artifact
from oclp.models import Digest, RecordReference

from bike_demand_service.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    prepare_features,
)
from bike_demand_service.publication import LocalPublisher
from bike_demand_service.tracking import (
    MLflowSettings,
    configure_mlflow,
    log_oclp_bridge,
)


def test_preparation_excludes_target_derived_fields_and_creates_temporal_folds() -> (
    None
):
    source = _source_rows(20)

    prepared = prepare_features(source, fold_count=3)

    assert tuple(prepared.frame.columns) == (
        "timestamp",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
    assert len(prepared.folds) == 3
    assert prepared.frame["timestamp"].is_monotonic_increasing
    assert set(prepared.feature_contract["excluded_source_columns"]) == {
        "casual",
        "dteday",
        "instant",
        "registered",
    }


def test_local_publisher_writes_a_payload_and_canonical_record(tmp_path: Path) -> None:
    root = tmp_path / "oclp"
    run_root = tmp_path / "run"
    with LocalPublisher(
        catalog_path=root / "catalog.duckdb",
        record_root=root,
        run_root=run_root,
    ) as publisher:
        published = publisher.artifact_for_bytes(
            artifact_id="urn:oclp-bike-demand:artifact:test-payload",
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
    assert published.reference.digest is not None
    assert (root / "artifact" / published.reference.digest.value[:2]).exists()


def test_mlflow_bridge_tags_one_run_with_oclp_references(tmp_path: Path) -> None:
    settings = MLflowSettings(root=tmp_path / "mlflow")
    mlflow = configure_mlflow(settings)
    invocation = _reference("urn:oclp-bike-demand:invocation:test")
    definition = _reference("urn:oclp-bike-demand:definition:test")

    with mlflow.start_run(run_name="bridge test"):
        log_oclp_bridge(
            mlflow,
            invocation=invocation,
            definition=definition,
            inputs={},
            outputs={},
        )

    runs = mlflow.search_runs(experiment_names=[settings.experiment_name])
    assert runs.iloc[0]["tags.oclp.invocation.id"] == invocation.id
    assert runs.iloc[0]["tags.oclp.definition.id"] == definition.id


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
    return RecordReference(id=identifier, digest=Digest(value="0" * 64))

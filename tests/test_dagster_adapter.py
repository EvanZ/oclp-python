"""Integration tests for the optional Dagster asset adapter."""

from __future__ import annotations

import pytest

from oclp import JsonArtifact, OpaqueSource, computation, new_lifecycle, run
from oclp.dagster import dagster_asset, dg_workflow
from oclp.models import Artifact, Execution
from oclp.publishing import LocalArtifactPublisher

dg = pytest.importorskip("dagster")


@computation(
    name="Build Dagster adapter report",
    outputs={"report": JsonArtifact(name="Dagster adapter report")},
)
def build_report() -> dict[str, int]:
    """Produce one ordinary OCLP-decorated output."""

    return {"answer": 42}


@run(name="Dagster adapter test workflow")
def build_report_workflow() -> dict[str, int]:
    """Invoke the real decorated Computation inside one observed workflow."""

    return build_report()


def test_dagster_asset_observes_existing_oclp_workflow(tmp_path):
    records = tmp_path / "records"
    lifecycle = new_lifecycle()

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    @dg.asset
    @dagster_asset(
        workflow=build_report_workflow,
        publisher=publisher_for,
        source=OpaqueSource(reason="Dagster adapter test source"),
        lifecycle=lifecycle,
    )
    def dagster_report(context: dg.AssetExecutionContext) -> dict[str, int]:
        return build_report_workflow()

    result = dg.materialize([dagster_report], raise_on_error=True)

    materialization = result.get_asset_materialization_events()[0]
    metadata = materialization.event_specific_data.materialization.metadata
    assert metadata["oclp.run.id"].value
    assert metadata["oclp.run.name"].value == "Dagster adapter test workflow"
    assert metadata["oclp.record_root"].value == str(records)
    assert metadata["oclp.status"].value == "succeeded"
    assert metadata["oclp.dagster.run_id"].value
    assert metadata["oclp.dagster.asset_key"].value == "dagster_report"
    assert metadata["oclp.dagster.retry_number"].value == 0

    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    assert [record.kind for record in observed_records].count("computation") == 1
    assert [record.kind for record in observed_records].count("execution") == 1
    binding = lifecycle.profile_bindings()["lifecycle"]
    assert (
        next(
            record for record in observed_records if isinstance(record, Execution)
        ).profiles["lifecycle"]
        == binding
    )
    assert next(
        record
        for record in observed_records
        if isinstance(record, Artifact) and record.name == "Dagster adapter report"
    ).profiles == {"lifecycle": binding}


def test_dg_workflow_creates_the_dagster_asset_without_a_second_decorator(tmp_path):
    records = tmp_path / "records"
    lifecycle = new_lifecycle()

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    @dg_workflow(
        workflow=build_report_workflow,
        publisher=publisher_for,
        source=OpaqueSource(reason="Dagster workflow projection test source"),
        asset_key="workflow_report",
        group_name="reports",
        lifecycle=lambda _context: lifecycle,
    )
    def workflow_report(context: dg.AssetExecutionContext) -> dict[str, int]:
        return build_report_workflow()

    result = dg.materialize([workflow_report], raise_on_error=True)

    materialization = result.get_asset_materialization_events()[0]
    metadata = materialization.event_specific_data.materialization.metadata
    assert metadata["oclp.dagster.asset_key"].value == "workflow_report"
    assert (
        workflow_report.group_names_by_key[dg.AssetKey("workflow_report")] == "reports"
    )

    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    assert [record.kind for record in observed_records].count("computation") == 1
    assert [record.kind for record in observed_records].count("execution") == 1
    assert (
        next(
            record for record in observed_records if isinstance(record, Execution)
        ).profiles["lifecycle"]
        == lifecycle.profile_bindings()["lifecycle"]
    )


def test_dagster_asset_rejects_unrecognized_context_fields(tmp_path):
    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=tmp_path / "catalog.duckdb",
            record_root=tmp_path / "records",
            payload_root=tmp_path / "payloads",
        )

    try:
        dagster_asset(
            workflow=build_report_workflow,
            publisher=publisher_for,
            source=OpaqueSource(reason="Dagster adapter test source"),
            context_fields=("environment",),
        )
    except ValueError as error:
        assert "unknown: environment" in str(error)
    else:  # pragma: no cover - guards the explicit context allow-list.
        raise AssertionError("dagster_asset accepted an arbitrary context field")

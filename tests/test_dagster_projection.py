"""Integration coverage for granular OCLP-to-Dagster projections."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

import pytest

from oclp import (
    JsonArtifact,
    OpaqueSource,
    computation,
    json_artifact,
    many,
    new_lifecycle,
    run,
)
from oclp.dagster import (
    DAGSTER_PROFILE,
    dagster_run_context,
    dg_artifact,
    dg_artifact_set,
    dg_computation,
    oclp_artifact_io_manager,
)
from oclp.models import Artifact, ArtifactSet, Execution
from oclp.publishing import LocalArtifactPublisher

dg = pytest.importorskip("dagster")


@run(name="Dagster projection test workflow")
def projection_workflow() -> None:
    """Declare the shared OCLP run policy for projected test steps."""


@json_artifact(name="Dagster projection source")
def acquire_source() -> dict[str, int]:
    """Acquire the first durable value in the test graph."""

    return {"value": 40}


@computation(
    name="Increment source value",
    inputs={"source": JsonArtifact},
    outputs={"incremented": JsonArtifact(name="Incremented source value")},
)
def increment_source(source: dict[str, int]) -> dict[str, int]:
    """Turn the acquired source into the first derived Artifact."""

    return {"value": source["value"] + 1}


@computation(
    name="Double source value",
    inputs={"incremented": JsonArtifact},
    outputs={"doubled": JsonArtifact(name="Doubled source value")},
)
def double_source(incremented: dict[str, int]) -> dict[str, int]:
    """Consume the exact first output through the second graph edge."""

    return {"value": incremented["value"] * 2}


@computation(
    name="Split source value",
    outputs={
        "left": JsonArtifact(name="Left source value"),
        "right": JsonArtifact(name="Right source value"),
    },
)
def split_source_value() -> dict[str, dict[str, int]]:
    """Produce two durable outputs in one real OCLP Execution."""

    return {"left": {"value": 20}, "right": {"value": 22}}


@computation(
    name="Sum split source values",
    inputs={"values": many(JsonArtifact)},
    outputs={"total": JsonArtifact(name="Total source value")},
)
def sum_split_source_values(values: tuple[dict[str, int], ...]) -> dict[str, int]:
    """Consume an ordered collection of exact upstream Artifact handles."""

    return {"value": sum(value["value"] for value in values)}


_flaky_attempts = 0


@computation(
    name="Retry once",
    outputs={"value": JsonArtifact(name="Retried value")},
)
def retry_once() -> dict[str, int]:
    """Fail its first Dagster attempt, retaining both OCLP Executions."""

    global _flaky_attempts
    _flaky_attempts += 1
    if _flaky_attempts == 1:
        raise RuntimeError("planned retry")
    return {"attempt": _flaky_attempts}


def _publisher_for(tmp_path):
    records = tmp_path / "records"

    def publisher(context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=(
                tmp_path
                / "payloads"
                / context.run.run_id
                / context.get_step_execution_context().step.key
                / f"attempt-{context.retry_number}"
            ),
        )

    return publisher, records


def _records(records):
    with LocalArtifactPublisher(
        catalog_path=records / "catalog.duckdb",
        record_root=records,
        payload_root=records / "unused-payloads",
    ) as publisher:
        return publisher.records()


def test_dagster_run_context_rehydrates_one_run_id_in_separate_workers():
    class AssetKey:
        def __init__(self, value: str) -> None:
            self.value = value

        def to_user_string(self) -> str:
            return self.value

    class WorkerContext:
        def __init__(self, asset_key: str) -> None:
            self.run = SimpleNamespace(run_id="scheduled-run-without-a-uuid")
            self.asset_key = AssetKey(asset_key)
            self.has_partition_key = False
            self.retry_number = 0

        def get_step_execution_context(self):
            return SimpleNamespace(
                step=SimpleNamespace(key=f"step:{self.asset_key.value}")
            )

    first = dagster_run_context(WorkerContext("first"))
    second = dagster_run_context(WorkerContext("second"))

    assert first.run_id == second.run_id
    assert first.profile.asset_key == "first"
    assert second.profile.asset_key == "second"


def test_projected_artifact_and_computations_form_one_dagster_graph(tmp_path):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster projection test source")

    @dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="raw_numbers",
    )
    def raw() -> object:
        return acquire_source()

    incremented = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="incremented_numbers",
        inputs={"source": dg.AssetIn(key=dg.AssetKey("raw_numbers"))},
    )(increment_source)
    doubled = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="doubled_numbers",
        inputs={"incremented": dg.AssetIn(key=dg.AssetKey("incremented_numbers"))},
    )(double_source)

    result = dg.materialize([raw, incremented, doubled], raise_on_error=True)

    materializations = result.get_asset_materialization_events()
    assert len(materializations) == 3
    metadata = [
        event.event_specific_data.materialization.metadata for event in materializations
    ]
    run_ids = {item["oclp.run.id"].value for item in metadata}
    assert len(run_ids) == 1
    assert UUID(run_ids.pop()) == UUID(result.run_id)
    assert metadata[0]["oclp.artifact.id"].value
    assert metadata[1]["oclp.execution.id"].value
    assert metadata[2]["oclp.output.doubled.id"].value

    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    assert len(executions) == 2
    assert {execution.profiles["run"]["run_id"] for execution in executions} == {
        result.run_id
    }
    assert {
        execution.profiles[DAGSTER_PROFILE]["asset_key"] for execution in executions
    } == {"incremented_numbers", "doubled_numbers"}
    executions_by_asset = {
        execution.profiles[DAGSTER_PROFILE]["asset_key"]: execution
        for execution in executions
    }
    assert (
        executions_by_asset["doubled_numbers"].inputs["incremented"]
        == executions_by_asset["incremented_numbers"].outputs["incremented"]
    )


def test_application_profiles_preserve_scheduler_facts(
    tmp_path,
):
    publisher, records = _publisher_for(tmp_path)
    profile = {
        "example_release": {
            "version": "1",
            "release_cycle_id": "cycle-123",
        }
    }
    raw = dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=OpaqueSource(reason="application profile test source"),
        asset_key="profiled_raw",
        application_profiles=profile,
    )(acquire_source)
    incremented = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=OpaqueSource(reason="application profile test source"),
        asset_key="profiled_incremented",
        inputs={"source": dg.AssetIn(key=dg.AssetKey("profiled_raw"))},
        application_profiles=lambda _context: profile,
    )(increment_source)

    dg.materialize([raw, incremented], raise_on_error=True)

    published = _records(records)
    artifacts = [record for record in published if isinstance(record, Artifact)]
    executions = [record for record in published if isinstance(record, Execution)]
    assert artifacts
    assert all(
        artifact.profiles == profile
        for artifact in artifacts
        if artifact.name in {"Dagster projection source", "Incremented source value"}
    )
    assert executions[0].profiles["example_release"] == profile["example_release"]
    assert DAGSTER_PROFILE in executions[0].profiles


def test_dg_computation_retains_each_retry_attempt_in_one_oclp_run(tmp_path):
    global _flaky_attempts
    _flaky_attempts = 0
    publisher, records = _publisher_for(tmp_path)
    retried = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=OpaqueSource(reason="Dagster retry test source"),
        asset_key="retried_numbers",
        retry_policy=dg.RetryPolicy(max_retries=1),
    )(retry_once)

    result = dg.materialize([retried], raise_on_error=True)

    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    assert len(executions) == 2
    assert {execution.profiles["run"]["run_id"] for execution in executions} == {
        result.run_id
    }
    assert {
        execution.profiles[DAGSTER_PROFILE]["retry_number"] for execution in executions
    } == {0, 1}


def test_dagster_selection_materializes_only_the_selected_real_boundaries(tmp_path):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster subset test source")
    raw = dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="raw_numbers",
    )(acquire_source)
    incremented = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="incremented_numbers",
        inputs={"source": dg.AssetIn(key=dg.AssetKey("raw_numbers"))},
    )(increment_source)
    doubled = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="doubled_numbers",
        inputs={"incremented": dg.AssetIn(key=dg.AssetKey("incremented_numbers"))},
    )(double_source)

    result = dg.materialize(
        [raw, incremented, doubled],
        selection=dg.AssetSelection.assets("raw_numbers", "incremented_numbers"),
        raise_on_error=True,
    )

    materialized_keys = {
        event.asset_key.to_user_string()
        for event in result.get_asset_materialization_events()
    }
    assert materialized_keys == {"raw_numbers", "incremented_numbers"}
    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    assert len(executions) == 1
    assert executions[0].profiles[DAGSTER_PROFILE]["asset_key"] == "incremented_numbers"


def test_declarative_proxy_uses_an_imported_computation_without_a_duplicate(tmp_path):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster proxy projection test source")

    @dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="proxy_source",
    )
    def proxy_source(context) -> object:
        assert context.run.run_id
        return acquire_source()

    @dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        target=increment_source,
        asset_key="proxy_incremented",
        inputs={"source": dg.AssetIn(key=dg.AssetKey("proxy_source"))},
        context_parameter="context",
    )
    def proxy_increment(source, context):
        assert context.run.run_id
        return increment_source(source)

    result = dg.materialize([proxy_source, proxy_increment], raise_on_error=True)

    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    assert len(executions) == 1
    assert executions[0].profiles[DAGSTER_PROFILE]["asset_key"] == "proxy_incremented"
    assert {
        event.asset_key.to_user_string()
        for event in result.get_asset_materialization_events()
    } == {"proxy_source", "proxy_incremented"}


def test_dg_computation_projects_atomic_multi_outputs_and_ordered_fan_in(tmp_path):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster multi-asset projection test source")
    lifecycle = new_lifecycle()

    split = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        outputs={
            "left": dg.AssetOut(key="left_number"),
            "right": dg.AssetOut(key="right_number"),
        },
        lifecycle=lifecycle,
    )(split_source_value)

    total = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="total_number",
        inputs={
            "values": (
                dg.AssetIn(key=dg.AssetKey("left_number")),
                dg.AssetIn(key=dg.AssetKey("right_number")),
            ),
        },
        lifecycle=lambda _context: lifecycle,
    )(sum_split_source_values)

    @dg_artifact_set(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="number_release",
        name="Projected number release",
        inputs={
            "left": dg.AssetIn(key=dg.AssetKey("left_number")),
            "right": dg.AssetIn(key=dg.AssetKey("right_number")),
        },
        members={
            "left": ("left", "candidate"),
            "right": ("right", "candidate"),
        },
        application_profiles={
            "example_release": {
                "version": "1",
                "release_cycle_id": "cycle-123",
            }
        },
        lifecycle=lifecycle,
    )
    def release() -> None:
        """Assemble the visible release collection from selected assets."""

    result = dg.materialize([split, total, release], raise_on_error=True)

    materialized_keys = {
        event.asset_key.to_user_string()
        for event in result.get_asset_materialization_events()
    }
    assert materialized_keys == {
        "left_number",
        "right_number",
        "total_number",
        "number_release",
    }
    records = _records(records)
    executions = [record for record in records if isinstance(record, Execution)]
    assert len(executions) == 2
    total_execution = next(
        execution
        for execution in executions
        if execution.profiles[DAGSTER_PROFILE]["asset_key"] == "total_number"
    )
    assert len(total_execution.inputs["values"]) == 2
    assert (
        total_execution.profiles["lifecycle"]
        == lifecycle.profile_bindings()["lifecycle"]
    )
    artifact_sets = [record for record in records if isinstance(record, ArtifactSet)]
    assert [artifact_set.name for artifact_set in artifact_sets] == [
        "Projected number release"
    ]
    assert artifact_sets[0].profiles == {
        "example_release": {"version": "1", "release_cycle_id": "cycle-123"},
        "lifecycle": lifecycle.profile_bindings()["lifecycle"],
    }
    manifests = [
        record
        for record in records
        if isinstance(record, Artifact) and record.name == "Projected number release"
    ]
    assert manifests[0].profiles is not None
    assert manifests[0].profiles["example_release"]["release_cycle_id"] == "cycle-123"
    assert (
        manifests[0].profiles["lifecycle"] == lifecycle.profile_bindings()["lifecycle"]
    )
    output_artifacts = [
        record
        for record in records
        if isinstance(record, Artifact)
        and record.name
        in {"Left source value", "Right source value", "Total source value"}
    ]
    assert output_artifacts
    assert all(
        artifact.profiles is not None
        and artifact.profiles["lifecycle"] == lifecycle.profile_bindings()["lifecycle"]
        for artifact in output_artifacts
    )


def test_oclp_artifact_io_manager_rehydrates_a_partitioned_handle_in_a_later_run(
    tmp_path,
):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster persistent handle test source")
    partitions = dg.StaticPartitionsDefinition(["cycle-a"])
    io_manager_key = "oclp_artifact_io"
    resource = oclp_artifact_io_manager(
        catalog_path=records / "catalog.duckdb",
        storage_root=tmp_path / "dagster-handles",
    )

    raw = dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="persistent_raw",
        partitions_def=partitions,
        io_manager_key=io_manager_key,
    )(acquire_source)
    incremented = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="persistent_incremented",
        inputs={"source": dg.AssetIn(key=dg.AssetKey("persistent_raw"))},
        partitions_def=partitions,
        io_manager_key=io_manager_key,
        run_name=lambda context: f"Increment persisted {context.partition_key}",
    )(increment_source)

    first = dg.materialize(
        [raw, incremented],
        selection=dg.AssetSelection.assets("persistent_raw"),
        partition_key="cycle-a",
        resources={io_manager_key: resource},
        raise_on_error=True,
    )
    second = dg.materialize(
        [raw, incremented],
        selection=dg.AssetSelection.assets("persistent_incremented"),
        partition_key="cycle-a",
        resources={io_manager_key: resource},
        raise_on_error=True,
    )

    assert {
        event.asset_key.to_user_string()
        for event in first.get_asset_materialization_events()
    } == {"persistent_raw"}
    assert {
        event.asset_key.to_user_string()
        for event in second.get_asset_materialization_events()
    } == {"persistent_incremented"}
    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    assert len(executions) == 1
    assert executions[0].profiles[DAGSTER_PROFILE]["partition_key"] == "cycle-a"
    assert executions[0].profiles["run"]["run_name"] == "Increment persisted cycle-a"


def test_partition_mappings_fan_out_and_fan_in_exact_artifact_handles(tmp_path):
    publisher, records = _publisher_for(tmp_path)
    source = OpaqueSource(reason="Dagster partition mapping test source")
    cycles = dg.StaticPartitionsDefinition(["cycle-a"])
    folds = dg.StaticPartitionsDefinition(["fold-1", "fold-2"])
    fold_partitions = dg.MultiPartitionsDefinition({"cycle": cycles, "fold": folds})
    cycle_mapping = dg.MultiToSingleDimensionPartitionMapping("cycle")
    io_manager_key = "oclp_artifact_io"
    resource = oclp_artifact_io_manager(
        catalog_path=records / "catalog.duckdb",
        storage_root=tmp_path / "dagster-handles",
    )

    raw = dg_artifact(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="partition_raw",
        partitions_def=cycles,
        io_manager_key=io_manager_key,
    )(acquire_source)
    fold_value = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="partition_fold_value",
        inputs={
            "source": dg.AssetIn(
                key=dg.AssetKey("partition_raw"),
                partition_mapping=cycle_mapping,
            )
        },
        partitions_def=fold_partitions,
        io_manager_key=io_manager_key,
    )(increment_source)
    aggregate = dg_computation(
        workflow=projection_workflow,
        publisher=publisher,
        source=source,
        asset_key="partition_total",
        inputs={
            "values": dg.AssetIn(
                key=dg.AssetKey("partition_fold_value"),
                partition_mapping=cycle_mapping,
            )
        },
        partitions_def=cycles,
        io_manager_key=io_manager_key,
    )(sum_split_source_values)
    assets = [raw, fold_value, aggregate]

    dg.materialize(
        assets,
        selection=dg.AssetSelection.assets("partition_raw"),
        partition_key="cycle-a",
        resources={io_manager_key: resource},
        raise_on_error=True,
    )
    for fold_key in ("fold-1", "fold-2"):
        dg.materialize(
            assets,
            selection=dg.AssetSelection.assets("partition_fold_value"),
            partition_key=dg.MultiPartitionKey({"cycle": "cycle-a", "fold": fold_key}),
            resources={io_manager_key: resource},
            raise_on_error=True,
        )
    result = dg.materialize(
        assets,
        selection=dg.AssetSelection.assets("partition_total"),
        partition_key="cycle-a",
        resources={io_manager_key: resource},
        raise_on_error=True,
    )

    assert {
        event.asset_key.to_user_string()
        for event in result.get_asset_materialization_events()
    } == {"partition_total"}
    executions = [
        record for record in _records(records) if isinstance(record, Execution)
    ]
    aggregate_execution = next(
        execution
        for execution in executions
        if execution.profiles[DAGSTER_PROFILE]["asset_key"] == "partition_total"
    )
    assert len(aggregate_execution.inputs["values"]) == 2

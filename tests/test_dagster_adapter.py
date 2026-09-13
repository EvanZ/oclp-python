"""Integration tests for the optional Dagster asset adapter."""

from __future__ import annotations

from typing import NamedTuple

import pytest

from oclp import (
    ArtifactHandle,
    JsonArtifact,
    OpaqueSource,
    assemble_artifact_set,
    computation,
    json_artifact,
    new_lifecycle,
)
from oclp.dagster import dagster_adapter, oclp_dagster_resource
from oclp.models import Artifact, ArtifactSet, Execution
from oclp.publishing import LocalArtifactPublisher

dg = pytest.importorskip("dagster")


_DECLARATION_ADAPTER_LIFECYCLE = new_lifecycle()


class _RecordingRunAdapter:
    """Record the runtime hooks forwarded by the Dagster resource."""

    strict = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_run_start(self, _observed: object) -> None:
        self.calls.append("run-start")

    def on_execution(self, _observed: object, **_kwargs: object) -> None:
        self.calls.append("execution")

    def on_run_end(
        self,
        _observed: object,
        _error: BaseException | None,
    ) -> None:
        self.calls.append("run-end")


@computation(
    name="Build native Dagster report",
    outputs={"report": JsonArtifact(name="Native Dagster report")},
    adapters=(dagster_adapter(lifecycle=_DECLARATION_ADAPTER_LIFECYCLE),),
)
def build_native_report() -> dict[str, int]:
    """Produce a canonical Computation directly materialized by Dagster."""

    return {"answer": 42}


@json_artifact(
    name="Native Dagster source",
    adapters=(dagster_adapter(),),
)
def build_native_source() -> dict[str, int]:
    """Produce a canonical Artifact directly materialized by Dagster."""

    return {"value": 7}


@computation(
    name="Split native Dagster report",
    outputs={
        "left": JsonArtifact(name="Native left report"),
        "right": JsonArtifact(name="Native right report"),
    },
    adapters=(dagster_adapter(),),
)
def split_native_report() -> dict[str, dict[str, int]]:
    """Produce two canonical OCLP outputs through one Dagster multi-asset."""

    return {"left": {"value": 1}, "right": {"value": 2}}


class _MixedNativeOutputs(NamedTuple):
    """Named domain result spanning native-only and OCLP-only outputs."""

    shared: dict[str, int]
    dagster_only: dict[str, int]
    oclp_only: dict[str, int]


@computation(
    name="Partially shared native Dagster report",
    outputs={
        "shared": JsonArtifact(name="Shared native report"),
        "oclp_only": JsonArtifact(name="OCLP-only report"),
    },
    adapters=(dagster_adapter(),),
)
def partially_shared_native_report() -> _MixedNativeOutputs:
    """Return one named result across the union of both output sets."""

    return _MixedNativeOutputs(
        shared={"value": 1},
        dagster_only={"value": 2},
        oclp_only={"value": 3},
    )


@computation(
    name="Explicitly bound native Dagster report",
    outputs={"canonical_report": JsonArtifact(name="Canonical report")},
    adapters=(
        dagster_adapter(output_bindings={"native_report": "canonical_report"}),
    ),
)
def explicitly_bound_native_report() -> dict[str, dict[str, int]]:
    """Return named values when native and canonical port names differ."""

    return {
        "canonical_report": {"value": 4},
        "dagster_only": {"value": 5},
    }


@computation(
    name="Build configured native Dagster report",
    outputs={"report": JsonArtifact(name="Configured native Dagster report")},
    adapters=(dagster_adapter(),),
)
def build_configured_native_report(
    *,
    multiplier: int = 2,
) -> dict[str, int]:
    """Return a report whose declared parameter comes from native asset config."""

    return {"answer": 21 * multiplier}


@computation(
    name="Build partitioned native Dagster report",
    outputs={"report": JsonArtifact(name="Partitioned native Dagster report")},
    adapters=(
        dagster_adapter(
            run_name=lambda context: f"Native report partition {context.partition_key}"
        ),
    ),
)
def build_partitioned_native_report() -> dict[str, int]:
    """Produce a report whose OCLP Run label identifies its partition."""

    return {"answer": 42}


@assemble_artifact_set(
    name="Native Dagster release",
    members={
        "source": ("source", "training-data"),
        "report": ("report", "evaluation"),
    },
    adapters=(dagster_adapter(),),
)
def assemble_native_release(
    source: ArtifactHandle,
    report: ArtifactHandle,
) -> None:
    """Assemble two exact OCLP Artifact handles without a synthetic execution."""


def test_dagster_adapter_lives_inside_a_canonical_computation(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    native_report = dg.asset(
        key="native_report",
        description="Build native Dagster report",
        required_resource_keys={"oclp"},
    )(build_native_report)

    result = dg.materialize(
        [native_report],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="native Dagster adapter test source"),
            )
        },
        raise_on_error=True,
    )

    materialization = result.get_asset_materialization_events()[0]
    metadata = materialization.event_specific_data.materialization.metadata
    assert metadata["oclp.run.name"].value == "Build native Dagster report"
    assert metadata["oclp.dagster.asset_key"].value == "native_report"
    assert isinstance(metadata["oclp.execution.id"].value, str)
    assert isinstance(metadata["oclp.output.report.id"].value, str)

    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    assert [record.kind for record in observed_records].count("computation") == 1
    assert [record.kind for record in observed_records].count("execution") == 1
    binding = _DECLARATION_ADAPTER_LIFECYCLE.profile_bindings()["lifecycle"]
    assert (
        next(
            record for record in observed_records if isinstance(record, Execution)
        ).profiles["lifecycle"]
        == binding
    )


def test_dagster_resource_activates_configured_runtime_adapters(tmp_path):
    records = tmp_path / "records"
    recorder = _RecordingRunAdapter()

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    native_report = dg.asset(key="native_report", required_resource_keys={"oclp"})(
        build_native_report
    )

    result = dg.materialize(
        [native_report],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(
                    reason="native Dagster runtime adapter test source"
                ),
                adapters=(recorder,),
            )
        },
        raise_on_error=True,
    )

    assert result.success
    assert recorder.calls == ["run-start", "execution", "run-end"]


def test_dagster_adapter_observes_a_canonical_artifact(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    native_source = dg.asset(key="native_source", required_resource_keys={"oclp"})(
        build_native_source
    )

    result = dg.materialize(
        [native_source],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="native Dagster artifact test source"),
            )
        },
        raise_on_error=True,
    )

    assert result.success
    materialization = result.get_asset_materialization_events()[0]
    metadata = materialization.event_specific_data.materialization.metadata
    artifact_id = metadata["oclp.artifact.id"].value
    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    artifact = next(
        record for record in observed_records if isinstance(record, Artifact)
    )
    assert artifact.name == "Native Dagster source"
    assert artifact_id == artifact.id


def test_dagster_adapter_forwards_matching_native_config_to_parameters(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    configured = dg.asset(
        key="build_configured_native_report",
        config_schema={"multiplier": dg.Field(int, default_value=2)},
        required_resource_keys={"oclp"},
    )(build_configured_native_report)

    result = dg.materialize(
        [configured],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="native Dagster config adapter test source"),
            )
        },
        run_config={
            "ops": {
                "build_configured_native_report": {
                    "config": {"multiplier": 3},
                }
            }
        },
        raise_on_error=True,
    )

    assert result.success
    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    execution = next(
        record for record in observed_records if isinstance(record, Execution)
    )
    assert execution.parameters == {"multiplier": 3}


def test_dagster_adapter_accepts_an_application_owned_partition_run_name(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    partitioned = dg.asset(
        key="partitioned_native_report",
        partitions_def=dg.StaticPartitionsDefinition(["one"]),
        required_resource_keys={"oclp"},
    )(build_partitioned_native_report)

    result = dg.materialize(
        [partitioned],
        partition_key="one",
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(
                    reason="native Dagster partition run-name test source"
                ),
            )
        },
        raise_on_error=True,
    )

    materialization = result.get_asset_materialization_events()[0]
    metadata = materialization.event_specific_data.materialization.metadata
    assert metadata["oclp.run.name"].value == "Native report partition one"


def test_dagster_adapter_returns_canonical_outputs_to_a_native_multi_asset(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    split = dg.multi_asset(
        outs={
            "left": dg.AssetOut(key="native_left"),
            "right": dg.AssetOut(key="native_right"),
        },
        required_resource_keys={"oclp"},
    )(split_native_report)

    result = dg.materialize(
        [split],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="native Dagster multi-asset test source"),
            )
        },
        raise_on_error=True,
    )

    assert result.success
    materializations = result.get_asset_materialization_events()
    assert len(materializations) == 2
    metadata_by_asset_key = {
        event.event_specific_data.materialization.asset_key.to_user_string(): (
            event.event_specific_data.materialization.metadata
        )
        for event in materializations
    }
    assert isinstance(
        metadata_by_asset_key["native_left"]["oclp.output.left.id"].value, str
    )
    assert isinstance(
        metadata_by_asset_key["native_right"]["oclp.output.right.id"].value,
        str,
    )
    assert {
        metadata["oclp.run.name"].value
        for metadata in metadata_by_asset_key.values()
    } == {"Split native Dagster report"}


def test_dagster_adapter_bridges_only_the_shared_named_outputs(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    partial = dg.multi_asset(
        outs={
            "shared": dg.AssetOut(key="native_shared"),
            "dagster_only": dg.AssetOut(key="native_dagster_only"),
        },
        required_resource_keys={"oclp"},
    )(partially_shared_native_report)

    result = dg.materialize(
        [partial],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="partial native Dagster output test source"),
            )
        },
        raise_on_error=True,
    )

    assert result.output_for_node(
        "partially_shared_native_report", "dagster_only"
    ) == {"value": 2}
    metadata_by_asset_key = {
        event.event_specific_data.materialization.asset_key.to_user_string(): (
            event.event_specific_data.materialization.metadata
        )
        for event in result.get_asset_materialization_events()
    }
    shared_metadata = metadata_by_asset_key["native_shared"]
    dagster_only_metadata = metadata_by_asset_key["native_dagster_only"]
    assert isinstance(shared_metadata["oclp.output.shared.id"].value, str)
    assert "oclp.output.oclp_only.id" not in shared_metadata
    assert "oclp.output.shared.id" not in dagster_only_metadata
    assert isinstance(dagster_only_metadata["oclp.execution.id"].value, str)

    with publisher_for(None) as publisher:
        artifacts = [
            record for record in publisher.records() if isinstance(record, Artifact)
        ]
    assert {artifact.name for artifact in artifacts} == {
        "Shared native report",
        "OCLP-only report",
    }


def test_dagster_adapter_accepts_an_explicit_native_to_oclp_output_binding(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    bound = dg.multi_asset(
        outs={
            "native_report": dg.AssetOut(key="native_report"),
            "dagster_only": dg.AssetOut(key="native_only"),
        },
        required_resource_keys={"oclp"},
    )(explicitly_bound_native_report)

    result = dg.materialize(
        [bound],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(
                    reason="explicit native Dagster output test source"
                ),
            )
        },
        raise_on_error=True,
    )

    metadata_by_asset_key = {
        event.event_specific_data.materialization.asset_key.to_user_string(): (
            event.event_specific_data.materialization.metadata
        )
        for event in result.get_asset_materialization_events()
    }
    assert isinstance(
        metadata_by_asset_key["native_report"]["oclp.output.canonical_report.id"].value,
        str,
    )
    assert "oclp.output.canonical_report.id" not in metadata_by_asset_key["native_only"]


def test_dagster_adapter_assembles_a_canonical_artifact_set(tmp_path):
    records = tmp_path / "records"

    def publisher_for(_context):
        return LocalArtifactPublisher(
            catalog_path=records / "catalog.duckdb",
            record_root=records,
            payload_root=tmp_path / "payloads",
        )

    source = dg.asset(key="native_release_source", required_resource_keys={"oclp"})(
        build_native_source
    )
    report = dg.asset(key="native_release_report", required_resource_keys={"oclp"})(
        build_native_report
    )
    release = dg.asset(
        key="native_release",
        ins={
            "source": dg.AssetIn(key=dg.AssetKey("native_release_source")),
            "report": dg.AssetIn(key=dg.AssetKey("native_release_report")),
        },
        required_resource_keys={"oclp"},
    )(assemble_native_release)

    result = dg.materialize(
        [source, report, release],
        resources={
            "oclp": oclp_dagster_resource(
                publisher=publisher_for,
                source=OpaqueSource(reason="native ArtifactSet adapter test source"),
            )
        },
        raise_on_error=True,
    )

    assert result.success
    with publisher_for(None) as publisher:
        observed_records = publisher.records()
    artifact_set = next(
        record for record in observed_records if isinstance(record, ArtifactSet)
    )
    assert artifact_set.name == "Native Dagster release"
    assert {member.role for member in artifact_set.members} == {
        "training-data",
        "evaluation",
    }
    release_materialization = next(
        event
        for event in result.get_asset_materialization_events()
        if event.event_specific_data.materialization.asset_key
        == dg.AssetKey("native_release")
    )
    metadata = release_materialization.event_specific_data.materialization.metadata
    assert metadata["oclp.artifact_set.id"].value == artifact_set.id
    assert metadata["oclp.run.name"].value == "Native Dagster release"

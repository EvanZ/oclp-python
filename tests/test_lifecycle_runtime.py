"""Runtime propagation coverage for the portable lifecycle profile."""

from __future__ import annotations

from oclp import (
    JsonArtifact,
    OclpRun,
    OpaqueSource,
    computation,
    json_artifact,
    lifecycle_from_id,
    new_lifecycle,
    observe_run,
    run,
)
from oclp.models import Artifact, ArtifactSet, Execution
from oclp.publishing import LocalArtifactPublisher


@run(name="Lifecycle runtime test workflow")
def lifecycle_workflow() -> None:
    """Declare the ordinary observed-workflow policy used by this test."""


@json_artifact(name="Lifecycle runtime source")
def lifecycle_source() -> dict[str, int]:
    return {"value": 1}


@computation(
    name="Increment lifecycle runtime source",
    inputs={"source": JsonArtifact},
    outputs={"result": JsonArtifact(name="Lifecycle runtime result")},
)
def increment_lifecycle_source(source: dict[str, int]) -> dict[str, int]:
    return {"value": source["value"] + 1}


def test_observe_run_attaches_a_lifecycle_to_concrete_outputs(tmp_path) -> None:
    lifecycle = new_lifecycle()
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with observe_run(
            lifecycle_workflow,
            publisher=publisher,
            source=OpaqueSource(reason="lifecycle runtime test"),
            lifecycle=lifecycle,
        ) as observed:
            source = lifecycle_source()
            result = increment_lifecycle_source(source)
            release = observed.publish_artifact_set(
                name="Lifecycle runtime release",
                members={
                    "result": (observed.outputs_for(result)["result"], "candidate")
                },
                materialize_manifest=True,
                manifest_name="Lifecycle runtime release manifest",
            )
        records = publisher.records()

    binding = lifecycle.profile_bindings()["lifecycle"]
    artifacts = [record for record in records if isinstance(record, Artifact)]
    executions = [record for record in records if isinstance(record, Execution)]
    artifact_sets = [record for record in records if isinstance(record, ArtifactSet)]

    assert artifacts
    assert all(artifact.profiles is not None for artifact in artifacts)
    assert all(artifact.profiles["lifecycle"] == binding for artifact in artifacts)
    assert len(executions) == 1
    assert executions[0].profiles["lifecycle"] == binding
    assert artifact_sets[0].profiles == {"lifecycle": binding}
    assert release.manifest is not None
    assert release.manifest.artifact.profiles is not None
    assert release.manifest.artifact.profiles["lifecycle"] == binding


def test_direct_oclp_run_can_join_an_existing_lifecycle_for_later_work(
    tmp_path,
) -> None:
    lifecycle = lifecycle_from_id("6df61052-c89f-4b0b-a758-d1f2c91cae0e")
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        with OclpRun(
            publisher=publisher,
            source=OpaqueSource(reason="lifecycle later work test"),
            lifecycle=lifecycle,
        ):
            source = lifecycle_source()
            increment_lifecycle_source(source)
        records = publisher.records()

    binding = lifecycle.profile_bindings()["lifecycle"]
    artifact = next(record for record in records if isinstance(record, Artifact))
    execution = next(record for record in records if isinstance(record, Execution))
    assert artifact.profiles is not None
    assert artifact.profiles["lifecycle"] == binding
    assert execution.profiles["lifecycle"] == binding

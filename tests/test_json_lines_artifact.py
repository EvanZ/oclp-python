"""Tests for the dependency-free JSON Lines Artifact integration."""

from __future__ import annotations

from oclp import GitSource, JsonLinesArtifact, OclpRun, computation
from oclp.publishing import LocalArtifactPublisher


@computation(
    id="urn:example:computation:publish-json-lines",
    name="Publish JSON Lines",
    outputs={"records": JsonLinesArtifact(name="Example records")},
)
def publish_json_lines() -> list[dict[str, int]]:
    return [{"sequence": 1}, {"sequence": 2}]


@computation(
    id="urn:example:computation:count-json-lines",
    name="Count JSON Lines",
    inputs={"records": JsonLinesArtifact},
)
def count_json_lines(records: list[dict[str, int]]) -> int:
    return sum(record["sequence"] for record in records)


def test_json_lines_artifact_round_trips_mapping_records(tmp_path) -> None:
    source = GitSource(
        repository="https://github.com/example/artifacts.git",
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
            records = publish_json_lines()
            handle = observed.outputs_for(records)["records"]
            total = count_json_lines(handle)

    assert total == 3
    assert handle.artifact.media_type == "application/x-ndjson"
    assert handle.path.suffix == ".jsonl"
    assert handle.read_verified_bytes() == b'{"sequence":1}\n{"sequence":2}\n'

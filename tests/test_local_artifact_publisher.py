"""Tests for the SDK's local Artifact publisher."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from oclp import Artifact
from oclp.publishing import ArtifactIdentityConflictError, LocalArtifactPublisher


def test_local_artifact_publisher_writes_payloads_and_canonical_records(tmp_path):
    records = tmp_path / "records"
    payloads = tmp_path / "payloads"
    with LocalArtifactPublisher(
        catalog_path=records / "catalog.duckdb",
        record_root=records,
        payload_root=payloads,
    ) as publisher:
        published = publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:report",
            name="Report",
            relative_path="reports/report.txt",
            content=b"ready\n",
            media_type="text/plain",
            created_at=datetime.now(UTC),
        )
        records_in_catalog = publisher.records()

    assert published.path.read_bytes() == b"ready\n"
    assert published.artifact.locations == (published.path.resolve().as_uri(),)
    assert len(records_in_catalog) == 1
    assert isinstance(records_in_catalog[0], Artifact)
    assert published.reference.digest is not None
    assert (records / "artifact" / published.reference.digest.value[:2]).exists()


def test_local_artifact_publisher_reuses_an_immutable_artifact_when_declaration_matches(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        first = publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:immutable-source",
            name="Immutable source",
            relative_path="first/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
        second = publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:immutable-source",
            name="Immutable source",
            relative_path="second/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )

        assert first.reference == second.reference
        assert first.artifact.created_at == second.artifact.created_at
        assert second.path.read_bytes() == b"unchanged\n"
        assert len(publisher.records()) == 1


def test_local_artifact_publisher_preserves_a_revised_application_owned_name(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        first = publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:immutable-source",
            name="Source data",
            relative_path="first/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
        revised = publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:immutable-source",
            name="Raw source data",
            relative_path="second/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )

        assert revised.reference != first.reference
        assert revised.artifact.name == "Raw source data"
        assert revised.artifact.digest == first.artifact.digest
        assert len(publisher.records()) == 2


def test_local_artifact_publisher_rejects_different_bytes_for_an_immutable_id(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        publisher.artifact_for_bytes(
            artifact_id="urn:example:artifact:immutable-source",
            name="Immutable source",
            relative_path="source.txt",
            content=b"first\n",
            media_type="text/plain",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(
            ArtifactIdentityConflictError,
            match="immutable Artifact ID",
        ):
            publisher.artifact_for_bytes(
                artifact_id="urn:example:artifact:immutable-source",
                name="Immutable source",
                relative_path="source.txt",
                content=b"different\n",
                media_type="text/plain",
                created_at=datetime.now(UTC),
            )

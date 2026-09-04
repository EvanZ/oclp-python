"""Tests for the SDK's local Artifact publisher."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from oclp import Artifact, record_digest
from oclp.catalog import CatalogIntegrityError
from oclp.publishing import LocalArtifactPublisher


def test_local_artifact_publisher_writes_payloads_and_canonical_records(tmp_path):
    records = tmp_path / "records"
    payloads = tmp_path / "payloads"
    with LocalArtifactPublisher(
        catalog_path=records / "catalog.duckdb",
        record_root=records,
        payload_root=payloads,
    ) as publisher:
        published = publisher.artifact_for_bytes(
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
    digest = record_digest(published.artifact)
    assert (records / "artifact" / digest.value[:2] / f"{digest.value}.json").exists()


def test_local_artifact_publisher_makes_distinct_records_per_materialization(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        first = publisher.artifact_for_bytes(
            name="Immutable source",
            relative_path="first/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
        second = publisher.artifact_for_bytes(
            name="Immutable source",
            relative_path="second/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        )

        assert first.reference != second.reference
        assert first.artifact.digest == second.artifact.digest
        assert second.path.read_bytes() == b"unchanged\n"
        assert len(publisher.records()) == 2


def test_local_artifact_publisher_publishes_a_new_record_when_metadata_changes(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        first = publisher.artifact_for_bytes(
            name="Source data",
            relative_path="first/source.txt",
            content=b"unchanged\n",
            media_type="text/plain",
            created_at=datetime(2026, 8, 31, tzinfo=UTC),
        )
        revised = publisher.artifact_for_bytes(
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


def test_local_artifact_publisher_rejects_reusing_a_uuid_for_different_records(
    tmp_path,
):
    with LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    ) as publisher:
        artifact_id = str(uuid4())
        publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name="Immutable source",
            relative_path="source.txt",
            content=b"first\n",
            media_type="text/plain",
            created_at=datetime.now(UTC),
        )

        with pytest.raises(
            CatalogIntegrityError,
            match="already identifies different immutable bytes",
        ):
            publisher.artifact_for_bytes(
                artifact_id=artifact_id,
                name="Immutable source",
                relative_path="source.txt",
                content=b"different\n",
                media_type="text/plain",
                created_at=datetime.now(UTC),
            )

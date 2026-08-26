"""Integration tests for the optional DuckDB OCLP catalog."""

from __future__ import annotations

import pytest

from oclp import Artifact, record_digest
from oclp.catalog import (
    AmbiguousRecordReferenceError,
    CatalogIntegrityError,
    RecordNotFoundError,
)
from oclp.catalog.duckdb import DuckdbCatalog
from oclp.models import Digest, RecordReference


def test_duckdb_catalog_resolves_exact_records_and_content_locations(tmp_path):
    artifact = Artifact(
        id="urn:example:artifact:report",
        media_type="application/json",
        digest=Digest(value="a" * 64),
        size=2,
        locations=("s3://example/reports/report.json",),
    )
    with DuckdbCatalog(tmp_path / "catalog.duckdb") as catalog:
        reference = catalog.publish(artifact)

        assert catalog.resolve(reference) == artifact
        assert catalog.records() == (artifact,)
        assert catalog.locations_for(reference) == ("s3://example/reports/report.json",)

        catalog.add_location(artifact.digest, "file:///cache/report.json")
        assert catalog.locations_for(reference) == (
            "file:///cache/report.json",
            "s3://example/reports/report.json",
        )
        assert catalog.artifacts_for_content(artifact.digest) == (artifact,)


def test_duckdb_catalog_rejects_missing_ambiguous_and_mismatched_references(tmp_path):
    first = Artifact(
        id="urn:example:artifact:mutable-name",
        media_type="text/plain",
        digest=Digest(value="b" * 64),
        size=1,
        locations=("s3://example/first.txt",),
    )
    second = first.model_copy(update={"locations": ("s3://example/second.txt",)})
    with DuckdbCatalog(tmp_path / "catalog.duckdb") as catalog:
        first_reference = catalog.publish(first)
        catalog.publish(second)

        with pytest.raises(AmbiguousRecordReferenceError):
            catalog.resolve(RecordReference(id=first.id))
        with pytest.raises(RecordNotFoundError):
            catalog.get("c" * 64)
        with pytest.raises(CatalogIntegrityError):
            catalog.resolve(
                RecordReference(
                    id="urn:example:artifact:wrong", digest=first_reference.digest
                )
            )
        assert record_digest(first) != record_digest(second)

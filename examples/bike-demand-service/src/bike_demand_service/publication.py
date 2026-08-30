"""Materialize byte payloads and publish their OCLP records locally."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oclp import Artifact, canonical_json_bytes
from oclp.catalog.duckdb import DuckdbCatalog
from oclp.models import Digest, OclpRecord, RecordReference


@dataclass(frozen=True)
class PublishedArtifact:
    """An Artifact together with its content-bound OCLP record reference."""

    artifact: Artifact
    path: Path
    reference: RecordReference


class LocalPublisher:
    """Keep canonical OCLP records and their payload bytes easy to inspect."""

    def __init__(
        self, *, catalog_path: Path, record_root: Path, run_root: Path
    ) -> None:
        self._catalog = DuckdbCatalog(catalog_path)
        self.record_root = record_root
        self.run_root = run_root
        self.record_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Close the local record index."""

        self._catalog.close()

    def __enter__(self) -> LocalPublisher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def publish(self, record: OclpRecord) -> RecordReference:
        """Index a record and persist its canonical document by record digest."""

        reference = self._catalog.publish(record)
        digest = reference.digest
        assert digest is not None
        target = (
            self.record_root / record.kind / digest.value[:2] / f"{digest.value}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(canonical_json_bytes(record))
        return reference

    def artifact_for_bytes(
        self,
        *,
        artifact_id: str,
        name: str,
        relative_path: str,
        content: bytes,
        media_type: str,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Write immutable bytes, hash them, and publish the matching Artifact."""

        target = self.run_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        artifact = Artifact(
            id=artifact_id,
            name=name,
            profiles=profiles,
            media_type=media_type,
            digest=Digest(value=hashlib.sha256(content).hexdigest()),
            size=len(content),
            created_at=created_at,
            locations=(target.resolve().as_uri(),),
            schema_uri=schema_uri,
        )
        return PublishedArtifact(
            artifact=artifact,
            path=target,
            reference=self.publish(artifact),
        )

    def json_artifact(
        self,
        *,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: Any,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Materialize human-readable deterministic JSON as an Artifact."""

        content = (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        return self.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type="application/json",
            created_at=created_at,
            profiles=profiles,
            schema_uri=schema_uri,
        )

    def artifact_for_file(
        self,
        *,
        artifact_id: str,
        name: str,
        relative_path: str,
        source_path: Path,
        media_type: str,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Publish an existing local file without making its bytes opaque."""

        return self.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=source_path.read_bytes(),
            media_type=media_type,
            created_at=created_at,
            profiles=profiles,
            schema_uri=schema_uri,
        )

    def records(self) -> tuple[OclpRecord, ...]:
        """Return every record currently available to the local catalog."""

        return self._catalog.records()


def utc_now() -> datetime:
    """Return an explicit-offset timestamp suitable for OCLP records."""

    return datetime.now(UTC)

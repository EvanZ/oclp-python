"""A small local publisher for OCLP records and immutable Artifact payloads.

This is intentionally a filesystem implementation, not a hosted registry or
an orchestration runtime. Applications choose names, locations, profiles, and
when a record is published; opaque identities come from the SDK.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oclp import canonical_json_bytes
from oclp.canonical import record_digest
from oclp.catalog.duckdb import DuckdbCatalog
from oclp.models import Artifact, Digest, OclpRecord, RecordReference, new_record_id


@dataclass(frozen=True)
class PublishedArtifact:
    """An Artifact together with its locally materialized payload and reference."""

    artifact: Artifact
    path: Path
    reference: RecordReference


class LocalArtifactPublisher:
    """Persist canonical OCLP records and immutable payload bytes locally.

    The publisher is generic: it knows how to write bytes, JSON, and existing
    files. It has no policy for a project's Artifact IDs, names, run layout,
    schemas, profiles, or computation boundaries. A project can put those
    conventions in its own run-scoped facade. A Core record UUID identifies one
    immutable record, so a changed artifact record must receive a new UUID.
    """

    def __init__(
        self,
        *,
        catalog_path: Path,
        record_root: Path,
        payload_root: Path,
    ) -> None:
        self._catalog = DuckdbCatalog(catalog_path)
        self.record_root = record_root
        self.payload_root = payload_root
        self.record_root.mkdir(parents=True, exist_ok=True)
        self.payload_root.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Close the local record catalog."""

        self._catalog.close()

    def __enter__(self) -> LocalArtifactPublisher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def publish(self, record: OclpRecord) -> RecordReference:
        """Index a record and write its canonical JSON by integrity digest."""

        reference = self._catalog.publish(record)
        digest = record_digest(record)
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
        artifact_id: str | None = None,
        name: str,
        relative_path: str,
        content: bytes,
        media_type: str,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        annotations: dict[str, Any] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Write immutable bytes, hash them, and publish the matching Artifact."""

        content_digest = Digest(value=hashlib.sha256(content).hexdigest())
        target = self.payload_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_annotations = annotations or {}
        target.write_bytes(content)
        artifact = Artifact(
            id=artifact_id or new_record_id(),
            name=name,
            profiles=profiles,
            annotations=resolved_annotations,
            media_type=media_type,
            digest=content_digest,
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
        artifact_id: str | None = None,
        name: str,
        relative_path: str,
        value: Any,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        annotations: dict[str, Any] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Materialize deterministic, human-readable JSON as an Artifact."""

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
            annotations=annotations,
            schema_uri=schema_uri,
        )

    def artifact_for_file(
        self,
        *,
        artifact_id: str | None = None,
        name: str,
        relative_path: str,
        source_path: Path,
        media_type: str,
        created_at: datetime,
        profiles: dict[str, dict[str, Any]] | None = None,
        annotations: dict[str, Any] | None = None,
        schema_uri: str | None = None,
    ) -> PublishedArtifact:
        """Copy an existing immutable file into the payload store as an Artifact."""

        return self.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=source_path.read_bytes(),
            media_type=media_type,
            created_at=created_at,
            profiles=profiles,
            annotations=annotations,
            schema_uri=schema_uri,
        )

    def records(self) -> tuple[OclpRecord, ...]:
        """Return every record currently available to this local publisher."""

        return self._catalog.records()


def utc_now() -> datetime:
    """Return an explicit-offset timestamp suitable for OCLP record fields."""

    return datetime.now(UTC)

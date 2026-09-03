"""DuckDB-backed reference catalog for immutable OCLP records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError as error:  # pragma: no cover - depends on installation
    raise ModuleNotFoundError(
        "DuckdbCatalog requires the optional dependency; install oclp[duckdb]."
    ) from error

from oclp.canonical import canonical_json_bytes, record_digest
from oclp.catalog.base import (
    AmbiguousRecordReferenceError,
    CatalogIntegrityError,
    RecordNotFoundError,
)
from oclp.models import Artifact, Digest, OclpRecord, RecordReference
from oclp.validation import parse_record

_RECORD_KINDS = (
    "artifact",
    "artifact_set",
    "computation",
    "contract",
    "execution",
    "evidence",
    "event",
)


class DuckdbCatalog:
    """Local, single-writer OCLP resolver and Artifact-location index.

    The catalog stores immutable canonical records by *record* digest. Its
    mutable location index is keyed by an Artifact's separate content digest;
    it is implementation metadata and never rewrites an OCLP record.
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        database_path = str(database)
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(database_path)
        self._initialize()

    def close(self) -> None:
        """Close the local DuckDB connection."""

        self._connection.close()

    def __enter__(self) -> DuckdbCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def publish(self, record: OclpRecord) -> RecordReference:
        """Store one canonical immutable record and return its bound reference."""

        digest = record_digest(record)
        canonical_json = canonical_json_bytes(record).decode("utf-8")
        existing = self._connection.execute(
            "SELECT canonical_json FROM oclp_records WHERE record_digest = ?",
            [digest.value],
        ).fetchone()
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO oclp_records
                    (record_digest, record_id, record_kind, canonical_json)
                VALUES (?, ?, ?, ?)
                """,
                [digest.value, record.id, record.kind, canonical_json],
            )
        elif existing[0] != canonical_json:
            raise CatalogIntegrityError(
                f"record digest collision for sha256:{digest.value}"
            )

        if isinstance(record, Artifact):
            self._connection.execute(
                """
                INSERT OR IGNORE INTO oclp_artifacts
                    (
                        record_digest, content_algorithm, content_digest,
                        media_type, size, schema_uri
                    )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    digest.value,
                    record.digest.algorithm,
                    record.digest.value,
                    record.media_type,
                    record.size,
                    record.schema_uri,
                ],
            )
            for location in record.locations:
                self.add_location(record.digest, location)
        return RecordReference(id=record.id, digest=digest)

    def ingest(self, records: Iterable[OclpRecord]) -> None:
        """Idempotently index an iterable of already-parsed OCLP records."""

        for record in records:
            self.publish(record)

    def ingest_directory(self, root: str | Path) -> None:
        """Index the SDK's simple content-addressed JSON-directory convention."""

        root_path = Path(root)
        for kind in _RECORD_KINDS:
            for path in sorted((root_path / kind).glob("*/*.json")):
                self.publish(parse_record(json.loads(path.read_text())))

    def resolve(self, reference: RecordReference) -> OclpRecord:
        """Resolve one reference, enforcing exact digest and ID bindings."""

        if reference.digest is not None:
            record = self.get(reference.digest)
            if record.id != reference.id:
                raise CatalogIntegrityError(
                    "resolved record ID does not match the integrity-bound reference"
                )
            return record

        matches = self._connection.execute(
            """
            SELECT record_digest FROM oclp_records
            WHERE record_id = ? ORDER BY record_digest
            """,
            [reference.id],
        ).fetchall()
        if not matches:
            raise RecordNotFoundError(f"no record found for ID {reference.id!r}")
        if len(matches) > 1:
            raise AmbiguousRecordReferenceError(
                f"ID-only reference {reference.id!r} matches {len(matches)} records"
            )
        return self.get(Digest(value=matches[0][0]))

    def get(self, digest: Digest | str) -> OclpRecord:
        """Return and verify the canonical record identified by its record digest."""

        value = digest.value if isinstance(digest, Digest) else digest
        row = self._connection.execute(
            "SELECT canonical_json FROM oclp_records WHERE record_digest = ?", [value]
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"no record found for digest sha256:{value}")
        record = parse_record(json.loads(row[0]))
        if record_digest(record).value != value:
            raise CatalogIntegrityError(
                f"stored record bytes do not match digest sha256:{value}"
            )
        return record

    def records(self) -> tuple[OclpRecord, ...]:
        """Return every stored record after individual digest verification."""

        digests = self._connection.execute(
            "SELECT record_digest FROM oclp_records ORDER BY record_digest"
        ).fetchall()
        return tuple(self.get(Digest(value=row[0])) for row in digests)

    def add_location(self, content: Digest, location: str) -> None:
        """Add a mutable retrieval hint for immutable Artifact content."""

        self._connection.execute(
            """
            INSERT OR IGNORE INTO oclp_artifact_locations
                (content_algorithm, content_digest, location)
            VALUES (?, ?, ?)
            """,
            [content.algorithm, content.value, location],
        )

    def locations_for(self, reference: RecordReference) -> tuple[str, ...]:
        """Return all known retrieval hints for one resolved Artifact reference."""

        record = self.resolve(reference)
        if not isinstance(record, Artifact):
            raise TypeError("locations can only be resolved for Artifact references")
        rows = self._connection.execute(
            """
            SELECT location FROM oclp_artifact_locations
            WHERE content_algorithm = ? AND content_digest = ?
            ORDER BY location
            """,
            [record.digest.algorithm, record.digest.value],
        ).fetchall()
        return tuple(row[0] for row in rows)

    def artifacts_for_content(self, content: Digest) -> tuple[Artifact, ...]:
        """Find all stored Artifact-record revisions for immutable content bytes."""

        rows = self._connection.execute(
            """
            SELECT record_digest FROM oclp_artifacts
            WHERE content_algorithm = ? AND content_digest = ?
            ORDER BY record_digest
            """,
            [content.algorithm, content.value],
        ).fetchall()
        return tuple(self.get(Digest(value=row[0])) for row in rows)  # type: ignore[return-value]

    def artifacts_for_id(self, artifact_id: str) -> tuple[Artifact, ...]:
        """Return every immutable Artifact record known for one logical ID."""

        rows = self._connection.execute(
            """
            SELECT record_digest FROM oclp_records
            WHERE record_id = ? AND record_kind = 'artifact'
            ORDER BY record_digest
            """,
            [artifact_id],
        ).fetchall()
        return tuple(self.get(Digest(value=row[0])) for row in rows)  # type: ignore[return-value]

    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oclp_records (
                record_digest VARCHAR PRIMARY KEY,
                record_id VARCHAR NOT NULL,
                record_kind VARCHAR NOT NULL,
                canonical_json VARCHAR NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS oclp_records_by_id ON oclp_records (record_id)"
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oclp_artifacts (
                record_digest VARCHAR PRIMARY KEY,
                content_algorithm VARCHAR NOT NULL,
                content_digest VARCHAR NOT NULL,
                media_type VARCHAR NOT NULL,
                size BIGINT NOT NULL,
                schema_uri VARCHAR
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oclp_artifact_locations (
                content_algorithm VARCHAR NOT NULL,
                content_digest VARCHAR NOT NULL,
                location VARCHAR NOT NULL,
                PRIMARY KEY (content_algorithm, content_digest, location)
            )
            """
        )

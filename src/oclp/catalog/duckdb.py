"""DuckDB-backed reference catalog for immutable OCLP records."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import duckdb
    import fasteners
except ModuleNotFoundError as error:  # pragma: no cover - depends on installation
    raise ModuleNotFoundError(
        "DuckdbCatalog requires optional dependencies; install oclp[duckdb]."
    ) from error

from oclp.canonical import canonical_json_bytes, record_digest
from oclp.catalog.base import (
    CatalogIntegrityError,
    RecordNotFoundError,
)
from oclp.models import Artifact, Digest, OclpRecord, RecordReference
from oclp.validation import parse_record

_RECORD_KINDS = (
    "artifact",
    "artifact_set",
    "computation",
    "execution",
    "evidence",
    "event",
)


class DuckdbCatalog:
    """Local OCLP resolver and Artifact-location index backed by DuckDB.

    Each Core record has one opaque UUID identity. The catalog retains a
    canonical record digest as storage-integrity metadata; it is not copied
    into protocol references. Its mutable location index is keyed by an
    Artifact's separate content digest.

    A filesystem-backed catalog acquires a short-lived
    :class:`fasteners.InterProcessLock` for every catalog operation, including
    connection setup. That lets independent local Dagster runs safely share a
    single DuckDB file without holding a lock while user Computations train or
    otherwise perform long-running work. The in-memory mode retains one normal
    connection for test and ephemeral use.
    """

    def __init__(self, database: str | Path = ":memory:") -> None:
        self._database_path = str(database)
        self._memory_connection: Any | None = None
        self._lock: Any | None = None
        if self._database_path == ":memory:":
            self._memory_connection = duckdb.connect(self._database_path)
            self._initialize(self._memory_connection)
            return

        path = Path(self._database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = fasteners.InterProcessLock(f"{path}.lock")
        with self._operation() as connection:
            self._initialize(connection)

    def close(self) -> None:
        """Close the in-memory catalog connection, if this catalog owns one."""

        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None

    def __enter__(self) -> DuckdbCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def publish(self, record: OclpRecord) -> RecordReference:
        """Store one canonical immutable record and return its UUID reference."""

        with self._operation() as connection:
            return self._publish(connection, record)

    def _publish(self, connection: Any, record: OclpRecord) -> RecordReference:
        """Publish one record while a catalog connection is exclusively held."""

        digest = record_digest(record)
        canonical_json = canonical_json_bytes(record).decode("utf-8")
        existing = connection.execute(
            "SELECT record_digest, canonical_json "
            "FROM oclp_records WHERE record_id = ?",
            [record.id],
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO oclp_records
                    (record_digest, record_id, record_kind, canonical_json)
                VALUES (?, ?, ?, ?)
                """,
                [digest.value, record.id, record.kind, canonical_json],
            )
        elif existing[1] != canonical_json:
            # A UUID is never a logical name or revision key: it identifies
            # precisely this immutable record. Revisions receive fresh UUIDs.
            raise CatalogIntegrityError(
                f"record ID {record.id!r} already identifies different immutable bytes"
            )

        if isinstance(record, Artifact):
            connection.execute(
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
                self._add_location(connection, record.digest, location)
        return RecordReference(id=record.id)

    def ingest(self, records: Iterable[OclpRecord]) -> None:
        """Idempotently index an iterable of already-parsed OCLP records."""

        with self._operation() as connection:
            for record in records:
                self._publish(connection, record)

    def ingest_directory(self, root: str | Path) -> None:
        """Index the SDK's simple content-addressed JSON-directory convention."""

        root_path = Path(root)
        with self._operation() as connection:
            for kind in _RECORD_KINDS:
                for path in sorted((root_path / kind).glob("*/*.json")):
                    self._publish(
                        connection, parse_record(json.loads(path.read_text()))
                    )

    def resolve(self, reference: RecordReference) -> OclpRecord:
        """Resolve one reference by UUID and verify stored canonical bytes."""

        with self._operation() as connection:
            return self._resolve(connection, reference)

    def _resolve(self, connection: Any, reference: RecordReference) -> OclpRecord:
        """Resolve one reference using an already-locked connection."""

        row = connection.execute(
            "SELECT record_digest FROM oclp_records WHERE record_id = ?",
            [reference.id],
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"no record found for ID {reference.id!r}")
        return self._get(connection, Digest(value=row[0]))

    def get(self, digest: Digest | str) -> OclpRecord:
        """Return and verify the canonical record identified by its record digest."""

        with self._operation() as connection:
            return self._get(connection, digest)

    def _get(self, connection: Any, digest: Digest | str) -> OclpRecord:
        """Resolve one immutable canonical record with an active connection."""

        value = digest.value if isinstance(digest, Digest) else digest
        row = connection.execute(
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

        with self._operation() as connection:
            digests = connection.execute(
                "SELECT record_digest FROM oclp_records ORDER BY record_digest"
            ).fetchall()
            return tuple(self._get(connection, Digest(value=row[0])) for row in digests)

    def add_location(self, content: Digest, location: str) -> None:
        """Add a mutable retrieval hint for immutable Artifact content."""

        with self._operation() as connection:
            self._add_location(connection, content, location)

    def _add_location(self, connection: Any, content: Digest, location: str) -> None:
        """Index one Artifact location with an active connection."""

        connection.execute(
            """
            INSERT OR IGNORE INTO oclp_artifact_locations
                (content_algorithm, content_digest, location)
            VALUES (?, ?, ?)
            """,
            [content.algorithm, content.value, location],
        )

    def locations_for(self, reference: RecordReference) -> tuple[str, ...]:
        """Return all known retrieval hints for one resolved Artifact reference."""

        with self._operation() as connection:
            record = self._resolve(connection, reference)
            if not isinstance(record, Artifact):
                raise TypeError(
                    "locations can only be resolved for Artifact references"
                )
            rows = connection.execute(
                """
                SELECT location FROM oclp_artifact_locations
                WHERE content_algorithm = ? AND content_digest = ?
                ORDER BY location
                """,
                [record.digest.algorithm, record.digest.value],
            ).fetchall()
            return tuple(row[0] for row in rows)

    def artifacts_for_content(self, content: Digest) -> tuple[Artifact, ...]:
        """Find Artifact records that describe the immutable content bytes."""

        with self._operation() as connection:
            rows = connection.execute(
                """
                SELECT record_digest FROM oclp_artifacts
                WHERE content_algorithm = ? AND content_digest = ?
                ORDER BY record_digest
                """,
                [content.algorithm, content.value],
            ).fetchall()
            return tuple(self._get(connection, Digest(value=row[0])) for row in rows)  # type: ignore[return-value]

    @contextmanager
    def _operation(self) -> Generator[Any, None, None]:
        """Yield a connection while preventing cross-process DuckDB conflicts."""

        if self._memory_connection is not None:
            yield self._memory_connection
            return
        if self._lock is None:
            raise RuntimeError("DuckdbCatalog is closed")
        acquired = self._lock.acquire(blocking=True)
        if not acquired:  # pragma: no cover - blocking acquisition should not fail.
            raise RuntimeError(
                f"could not acquire DuckDB lock for {self._database_path}"
            )
        connection = None
        try:
            connection = duckdb.connect(self._database_path)
            yield connection
        finally:
            if connection is not None:
                connection.close()
            self._lock.release()

    def _initialize(self, connection: Any) -> None:
        """Create the small catalog schema while its connection is locked."""

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oclp_records (
                record_digest VARCHAR PRIMARY KEY,
                record_id VARCHAR NOT NULL UNIQUE,
                record_kind VARCHAR NOT NULL,
                canonical_json VARCHAR NOT NULL
            )
            """
        )
        connection.execute(
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oclp_artifact_locations (
                content_algorithm VARCHAR NOT NULL,
                content_digest VARCHAR NOT NULL,
                location VARCHAR NOT NULL,
                PRIMARY KEY (content_algorithm, content_digest, location)
            )
            """
        )

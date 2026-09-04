# DuckDB catalog

The optional `DuckdbCatalog` is a small local reference implementation of OCLP
record resolution. It is not required by the protocol, and it is not a hosted
registry or artifact store.

It stores canonical immutable records by *record digest*. For `Artifact`
records it additionally maintains a mutable location index keyed by the
Artifact's separate *content digest*. Updating that local index never rewrites
an OCLP record.

## Publish and resolve records

```python
from pathlib import Path

from oclp.catalog.duckdb import DuckdbCatalog

with DuckdbCatalog(Path(".oclp/catalog.duckdb")) as catalog:
    reference = catalog.publish(artifact)
    resolved = catalog.resolve(reference)

assert resolved == artifact
```

`publish` returns a content-bound `RecordReference`. `resolve` verifies the
stored canonical record digest and ID before returning it. An ID-only reference
is accepted only when exactly one stored record has that ID; otherwise it is
rejected as absent or ambiguous.

## Artifact locations

Locations are retrieval hints, not the identity of an Artifact. The catalog can
associate an additional location with existing immutable content:

```python
catalog.add_location(artifact.digest, "file:///cache/daily.json")
locations = catalog.locations_for(reference)
```

Applications remain responsible for retrieving the content and verifying its
bytes against `artifact.digest`. The local catalog does not fetch content or
silently choose a storage location for the application.

## Ingest a record directory

The catalog can ingest a simple OCLP JSON directory layout:

```python
with DuckdbCatalog(".oclp/catalog.duckdb") as catalog:
    catalog.ingest_directory("data/oclp")
```

The directory convention is SDK infrastructure, not a required OCLP storage
format. Any compliant resolver may use a registry, bundle reader, database, or
other mechanism while honoring exact UUID record references and verified
payload digests.

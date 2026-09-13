# Sources and publishing

OCLP records need two application-provided runtime facts: an implementation
source that says what code produced the observation, and a publisher that
persists immutable records and payload bytes.

## Select the implementation source

Pass an exact source basis when observing a run. `GitSource` is the usual
choice for code in a repository; `ArtifactSource`, `ServiceSource`, and
`OpaqueSource` cover other implementation boundaries:

```python
from oclp import GitSource, observe_run


source = GitSource(
    repository="https://github.com/example/demand.git",
    commit="0123456789abcdef0123456789abcdef01234567",
    path="src/demand/training.py",
)

with observe_run(workflow, publisher=publisher, source=source):
    workflow()
```

The runtime binds the selected source to the Computations, Executions, and
Evidence it publishes. It does not infer a revision from a function name.

## Publish locally

`LocalArtifactPublisher` writes immutable payloads and canonical record JSON,
and maintains a local DuckDB catalog:

```python
from pathlib import Path

from oclp.publishing import LocalArtifactPublisher


with LocalArtifactPublisher(
    catalog_path=Path("data/oclp/catalog.duckdb"),
    record_root=Path("data/oclp"),
    payload_root=Path("data/runs/release-001"),
) as publisher:
    with observe_run(workflow, publisher=publisher, source=source):
        workflow()
```

The publisher is application bootstrap, not a Computation parameter or an
Artifact representation decision. For catalog querying, ingestion, and payload
locations, see the [DuckDB catalog integration](catalog.md).

## Publish without decorators

The decorator API is recommended for normal Python functions. Publishers also
expose explicit methods for cases with domain-specific serialization or dynamic
membership. Those calls must still publish valid Core records and real
references; they do not create a hidden workflow model.

# API Overview

This page is a lookup map for the public SDK API. The Core API guides explain
when and how to use each record type; this index points to the relevant
declarations and runtime helpers.

| Need | Primary API | Guide |
| --- | --- | --- |
| Persist one external or source value | `@json_artifact`, `@csv_artifact`, `@parquet_artifact`, and other Artifact decorators | [Artifacts](artifacts.md) |
| Declare reusable work and publish its real invocation | `@computation`, `computation_record`, `OclpRun.outputs_for()` | [Computations and Executions](computations-and-executions.md) |
| Record a quality or acceptance conclusion | `@evidence`, `evaluate_evidence` | [Evidence](evidence.md) |
| Package exact Artifacts | `ComputationArtifactSet`, `@artifact_set`, `assemble_artifact_set`, `artifact_set_input` | [Artifact sets](artifact-sets.md) |
| Observe related real Executions | `@run`, `observe_run`, `OclpRun`, `active_run` | [Runs](runs.md) |
| Bind code and persist records | `GitSource`, `ArtifactSource`, `ServiceSource`, `OpaqueSource`, `LocalArtifactPublisher` | [Sources and publishing](sources-and-publishing.md) |
| Parse, digest, and validate portable records | `parse_record`, `canonical_json_bytes`, `record_digest`, validation helpers | [Canonical records and validation](canonical-records.md) |

## Core record models

The immutable Pydantic models live in `oclp.models`:

```python
from oclp.models import (
    Artifact,
    ArtifactSet,
    Computation,
    Evidence,
    Event,
    Execution,
    RecordReference,
)
```

Their field semantics and conformance requirements are defined by the
[OCLP specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/).
Use the SDK declarations for ordinary application code; instantiate model
objects directly when implementing a store, ingesting records, or performing
a low-level integration.

## Optional integrations

- [Dagster](dagster.md) activates canonical OCLP declarations inside native
  Dagster assets.
- [MLflow](mlflow.md) mirrors OCLP observations for experiment tracking.
- [DuckDB catalog](catalog.md) provides local record and Artifact-location
  storage.

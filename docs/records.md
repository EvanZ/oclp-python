# Records and canonicalization

The SDK exposes immutable Pydantic models for the six OCLP core record kinds:

| Kind | SDK model | Role |
| --- | --- | --- |
| `definition` | `ComputationDefinition` | A reusable computation interface and implementation basis. |
| `invocation` | `Invocation` | One request to execute a Definition. |
| `artifact` | `Artifact` | Immutable content bytes, such as an input, output, log, or package. |
| `artifact_set` | `ArtifactSet` | A named, exact collection of Artifacts. |
| `evidence` | `Evidence` | A contract evaluation about a record. |
| `event` | `LifecycleEvent` | An ordered observation about an execution attempt. |

The API names are Python conveniences. Field semantics, required fields, and
conformance requirements belong to the [normative specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/).

## Core API

```python
from oclp import (
    canonical_json_bytes,
    parse_record,
    record_digest,
    validate_derivation_graph,
    validate_invocation_hierarchy,
)
```

- `parse_record(value)` validates a JSON-compatible value and returns the
  appropriate typed record.
- `canonical_json_bytes(record)` emits the RFC 8785 canonical JSON bytes used
  for an OCLP record digest.
- `record_digest(record)` returns the SHA-256 digest of those canonical bytes.
- `validate_derivation_graph(records)` validates resolved input/output
  derivation bindings and rejects cycles.
- `validate_invocation_hierarchy(records)` validates the separate parent-child
  Invocation hierarchy and rejects orchestration cycles.

## Logical identity and immutable revisions

An OCLP `id` is a logical name. A record digest binds one exact canonical record
revision. An Artifact also has a content digest for the bytes it describes.
Applications should publish `RecordReference` values with a digest whenever an
exact immutable revision is required.

```python
from oclp import record_digest
from oclp.models import RecordReference

reference = RecordReference(
    id=artifact.id,
    digest=record_digest(artifact),
)
```

This distinction lets one logical Artifact gain a new immutable record revision
when its metadata changes without claiming that its payload bytes changed.

## Profiles

Profiles add opt-in semantic layers without expanding the portable core. This
SDK includes helpers for the draft `dataset-snapshot`, `execution-context`, and
`lifecycle` profiles under `oclp.profiles`.

Profile bindings are carried by the core `profiles` field. A producer emits
`profiles: null` when no profile applies. A consumer that needs profile-specific
meaning validates the named profile in addition to the core record.

# Getting started

## Install

Install the package directly from this repository while it is pre-release:

```bash
pip install "oclp[duckdb] @ git+https://github.com/EvanZ/oclp-python.git@main"
```

Use an immutable commit SHA rather than `main` when building a reproducible
deployment. Omit `[duckdb]` if the local catalog is not needed.

For SDK development:

```bash
git clone https://github.com/EvanZ/oclp-python.git
cd oclp-python
uv sync --all-groups
```

## Create and digest a record

An `Artifact` describes immutable content. Its logical `id` is distinct from
the SHA-256 `digest` of the content bytes.

```python
from datetime import UTC, datetime

from oclp import Artifact, canonical_json_bytes, record_digest
from oclp.models import Digest

artifact = Artifact(
    id="urn:example:artifact:daily-report",
    name="Daily report",
    media_type="application/json",
    digest=Digest(value="a" * 64),
    size=42,
    created_at=datetime.now(UTC),
    locations=("s3://example-reports/daily.json",),
)

canonical_record = canonical_json_bytes(artifact)
record_reference_digest = record_digest(artifact)
```

`record_digest` hashes the canonical JSON record, while `artifact.digest`
identifies the Artifact's described payload bytes. They are intentionally
different identities.

## Parse untrusted JSON

Use `parse_record` to validate a JSON-compatible object against the closed core
record vocabulary:

```python
from oclp import parse_record

record = parse_record({
    "oclp_version": "0.1.0-draft",
    "kind": "artifact",
    "id": "urn:example:artifact:daily-report",
    "media_type": "application/json",
    "digest": {"algorithm": "sha256", "value": "a" * 64},
    "size": 42,
})
```

See the [normative core specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/)
for the complete field contract.

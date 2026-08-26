# OCLP Python SDK

`oclp` is the reference Python implementation of the [Open Computation
Lifecycle Protocol (OCLP)](https://github.com/EvanZ/open-computation-lifecycle).
It provides strict record models, RFC 8785 canonical JSON, SHA-256 record
digests, record and graph validation, profile helpers, and an optional local
DuckDB catalog.

The SDK implements the standard; it does not define it. The canonical source
for protocol meaning, schemas, examples, and cross-language conformance vectors
is the [OCLP standard documentation](https://evanz.github.io/open-computation-lifecycle/).

## What it is for

Use the SDK when a Python application needs to describe durable computation
lineage without adopting an orchestration engine or a hosted provenance
service. An application chooses when to create `Definition`, `Invocation`,
`Artifact`, `ArtifactSet`, `Evidence`, and `Event` records. OCLP preserves and
validates those observations; it does not run the computation.

```text
application code --> oclp records --> catalog / files / registry --> consumers
                                  \-----------------------------> Cyclops
```

Cyclops is a separate read-only explorer. It reads published OCLP records but
does not change how a producer executes a job.

## Boundaries

The SDK intentionally does not provide a scheduler, a workflow DSL, automatic
dependency capture, a remote artifact store, or a model tracker. Those are
application or ecosystem concerns that can produce and consume the same OCLP
records.

Start with [getting started](getting-started.md), then see [records and
canonicalization](records.md) for the core SDK operations.

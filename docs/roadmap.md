# SDK roadmap

This page tracks proposed implementation work for the Python SDK. It does not
extend or redefine the OCLP Core protocol.

An active roadmap item links to its GitHub issue. When the work is implemented
or declined, close that issue and update this entry to record the outcome.

## Durable record stores

### S3-compatible canonical RecordStore

Implement an SDK-backed publisher and reader for S3-compatible object storage.
It should preserve OCLP's canonical-record model: the record UUID remains the
protocol identity, while a SHA-256 digest addresses immutable canonical record
bytes under digest-sharded keys such as `records/artifact/77/77af…c511.json`.
Artifact payload bytes may be stored separately and addressed by their payload
digest.

The store must use conditional, create-once writes, so repeat publication is
safe and accepted records are never overwritten. It should be usable directly
by SDK consumers and support Cyclops loading or an explicit local mirror;
DuckDB remains an optional local index rather than the source of truth.

Track in [#1](https://github.com/EvanZ/oclp-python/issues/1).

## Delivery and integration interfaces

These items implement Core records without making a particular database,
object store, stream broker, or observability vendor part of the protocol.

### Declarative run-level ArtifactSets — implemented

Let a real `@run` declare an ArtifactSet from exact outputs of its child
Computations. This should make a cross-computation release boundary visible at
the run declaration while preserving direct, immutable collection publication:
no synthetic release Computation, Execution, or Events. It should validate
that each required declared member resolves once in the completed run, support
an optional release-manifest sidecar, and retain `OclpRun.publish_artifact_set`
for genuinely dynamic collections. Distributed contribution coordination is
out of scope.

Implemented in [#8](https://github.com/EvanZ/oclp-python/issues/8).

### RecordStore and exporter interfaces

Define small SDK interfaces for an authoritative durable `RecordStore` and
optional delivery exporters. A file-backed store and the optional DuckDB
catalog remain useful local building blocks: the catalog is an index that can
be rebuilt from canonical records, not the required source of truth.

Track in [#5](https://github.com/EvanZ/oclp-python/issues/5).

### Kafka delivery exporter

Export selected records only after durable publication. The exporter must
tolerate retries and duplicate delivery; consumers deduplicate using stable
record identity and content binding. Kafka is a delivery sink, not the sole
OCLP system of record.

Track in [#6](https://github.com/EvanZ/oclp-python/issues/6).

### OTLP delivery exporter

Project selected Event, Evidence, and Diagnostic facts to an OpenTelemetry
Collector. The canonical OCLP record remains authoritative; OTLP is a derived,
lossy operational view.

Track in [#3](https://github.com/EvanZ/oclp-python/issues/3).

### Artifact resolution interface

Define an SDK interface for resolving an immutable Artifact identity to
available mutable locations without copying those locations onto every
Execution reference.

Track in [#2](https://github.com/EvanZ/oclp-python/issues/2).

### Framework adapters

Build adapters for orchestration systems, ML training and evaluation, batch
processing, and request-scoped inference. Each adapter should record observed
materializations without becoming an orchestrator or workflow DSL.

Track in [#4](https://github.com/EvanZ/oclp-python/issues/4).

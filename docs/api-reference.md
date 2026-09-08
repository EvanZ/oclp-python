# API reference

This is the public, developer-facing API of the `oclp` package. It covers the
decorators, runtime helpers, record models, local storage, and validation tools
that an application may intentionally use. Import from `oclp` unless a section
explicitly names another module.

This page is a practical SDK contract, not a restatement of the protocol. Core
record field semantics remain defined by the [OCLP
specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/).

## Choose an API

| Need | API |
| --- | --- |
| Persist an acquired value | An Artifact decorator such as `@json_artifact` |
| Declare and observe application work | `@computation` and `OclpRun` / `observe_run(...)` |
| Require a validation gate | `@evidence` and `requires=(...)` |
| Build a release/package | `ComputationArtifactSet`, `RunArtifactSet`, or `publish_artifact_set(...)` |
| Capture a Git source basis | `source_from_git_checkout(...)` |
| Reproduce dirty source changes | `capture_git_source_overlay(...)` |
| Publish locally | `LocalArtifactPublisher` |
| Verify or inspect a graph | `parse_record(...)` and `validate_*` helpers |

## Core records

The SDK re-exports immutable Pydantic models for all Core record kinds:

```python
from oclp import Artifact, ArtifactSet, Computation, Evidence, Event, Execution
from oclp.models import (
    ArtifactSetMember,
    Digest,
    GitSource,
    OpaqueSource,
    PortDefinition,
    RecordReference,
    ServiceSource,
)
```

Construct these directly when the application already has a durable fact to
record. Otherwise prefer the decorators and runtime below, which create UUIDs,
references, Event chronology, and Artifact bindings consistently. Supporting
Core models include `Diagnostic`, `ParameterDefinition`, `GitCheckout`, and
`ArtifactSource`; see [records and canonicalization](records.md).

## Artifact boundaries

### `ArtifactType`

`ArtifactType` declares a durable representation. Pass a concrete *class* to a
Computation input and a named concrete *instance* to an output:

```python
from oclp import CsvArtifact, JsonArtifact, computation


@computation(
    name="Summarize demand",
    inputs={"features": CsvArtifact},
    outputs={"summary": JsonArtifact(name="Demand summary")},
)
def summarize_demand(features) -> dict[str, object]:
    return {"rows": len(features)}
```

Input classes validate an Artifact media type. Output instances additionally
carry a required application `name` and may contain `annotations`,
`schema_uri`, and serialization options.

`outputs` determines how a callable result becomes persisted output ports. With
**one** declared output, the entire return value is materialized at that port.
The returned `{"rows": ...}` in this example therefore becomes the JSON
Artifact bound to `outputs["summary"]`; the return mapping does not need a
`"summary"` key. With **multiple** declared outputs, the return value must be
a mapping or object with one field for each output-port name:

```python
@computation(
    name="Score demand",
    outputs={
        "predictions": JsonArtifact(name="Demand predictions"),
        "metrics": JsonArtifact(name="Demand metrics"),
    },
)
def score_demand(...) -> dict[str, object]:
    return {
        "predictions": {"values": [12.0, 14.0]},
        "metrics": {"mae": 1.2},
    }
```

Persistence occurs only inside an active `OclpRun`; outside it, the decorated
function returns its ordinary Python value.

### Acquisition decorators

Use these when a function obtains an external value that must become an
Artifact before it reaches a Computation. They require an active `OclpRun` and
return an `ArtifactHandle`.

```python
from oclp import json_artifact


@json_artifact(name="Training plan")
def read_training_plan() -> dict[str, object]:
    return {"folds": 3}
```

| Decorator | Durable representation |
| --- | --- |
| `json_artifact(...)` | JSON or pandas table JSON |
| `json_lines_artifact(...)` | JSON Lines |
| `csv_artifact(...)` | CSV |
| `parquet_artifact(...)` | Parquet |
| `arrow_ipc_artifact(...)` | Arrow IPC |
| `npy_artifact(...)`, `npz_artifact(...)` | NumPy `.npy` / `.npz` |
| `yaml_artifact(...)`, `toml_artifact(...)`, `xml_artifact(...)` | YAML / TOML / XML |

The corresponding `JsonArtifact`, `CsvArtifact`, `ParquetArtifact`,
`ArrowIpcArtifact`, `NpyArtifact`, `NpzArtifact`, `YamlArtifact`,
`TomlArtifact`, and `XmlArtifact` classes are output declarations. The SDK
also supplies `BytesArtifact`, `FileArtifact`, `CatBoostModelArtifact`,
`XGBoostModelArtifact`, `LightGBMModelArtifact`, and `SklearnModelArtifact`.
See [artifact formats and integrations](integrations.md) for value types,
media types, options, and package extras.

### Handles and adapters

An `ArtifactHandle` exposes:

```python
handle.artifact               # Core Artifact
handle.reference              # RecordReference
handle.path                   # local payload path
handle.read_verified_bytes()  # SHA-256 verification, then bytes
```

`ArtifactAdapterRegistry` converts verified payloads into downstream Python
types. `DEFAULT_ARTIFACT_ADAPTERS` provides shipped adapters:

```python
frame = DEFAULT_ARTIFACT_ADAPTERS.load(handle, pandas.DataFrame)
```

Pass a custom `ArtifactAdapterRegistry` as `artifact_adapters=` to `OclpRun`
or `observe_run(...)`. Subclass `ArtifactAdapter` for application formats by
implementing `supports(handle, target_type)` and `load(handle, target_type)`.
`artifact_handle(published)` converts a `PublishedArtifact` to its handle, and
`artifact_type(function)` returns the representation bound to an acquisition
callable. Representation and integrity failures raise `ArtifactAdapterError`
or its `ArtifactIntegrityError` subtype.

## Computations

### `@computation(...)`

```python
@computation(
    name="...",
    input_ports=(),
    inputs=None,
    output_ports=(),
    outputs=None,
    artifact_set=None,
    requires=None,
    profiles=None,
    annotations=None,
)
def work(...): ...
```

`name` is required and application-owned. `inputs` maps parameter/port names
to an `ArtifactType`, `many(ArtifactType)`, or `artifact_set_input(...)`.
`outputs` maps output-port names to named `ArtifactType` instances. In an
active runtime, the SDK emits one source-bound Computation per callable per
observed run and an Execution for each call. Outside a runtime, the callable
retains normal Python behavior.

Within that same active runtime, passing an exact raw value returned from one
decorated Computation to another decorated Computation automatically reuses the
published Artifact binding for the downstream input. Passing an
`ArtifactHandle` instead requests an explicit verified payload load through
the consuming parameter's adapter. Handles are required once a process or run
boundary removes in-memory object identity.

Use `input_ports` and `output_ports` directly only to declare portable
interfaces that do not need SDK-managed payload serialization. `requires`
contains evaluators decorated with `@evidence`.

| Helper | Use |
| --- | --- |
| `computation_template(function)` | Read static declaration metadata. |
| `computation_record(function, source=...)` | Materialize a source-bound Core Computation directly. |
| `computation_input_artifact_types(function)` | Read declared Artifact inputs. |
| `computation_output(function, port)` or `function.output(port)` | Refer to an output in `RunArtifactSet`. |
| `many(ArtifactType)` | Declare a many-cardinality input. |
| `artifact_set_input({"model": CatBoostModelArtifact, ...})` | Declare named ArtifactSet members required as one input. |

### `ComputationArtifactSet`

Use this when outputs from *one* computation also form one package. It adds an
ArtifactSet output to the same real Execution; it does not create a synthetic
packaging computation.

```python
@computation(
    name="Train candidate",
    outputs={
        "model": CatBoostModelArtifact(name="Candidate model"),
        "metrics": JsonArtifact(name="Candidate metrics"),
    },
    artifact_set=ComputationArtifactSet(
        name="Candidate release",
        members={"model": ("model", "model"), "metrics": ("metrics", "metrics")},
        port="release",
    ),
)
def train_candidate(...): ...
```

`members` maps a set member name to `(output_port, optional_role)`. `port`
defaults to `"artifact_set"`.

## Evidence

```python
@evidence(name="Candidate quality", profiles=None, annotations=None)
def candidate_quality(result) -> Literal["pass", "fail", "error"]:
    return "pass"
```

Attach it with `@computation(..., requires=(candidate_quality,))`. Required
evidence is evaluated for each Execution; a successful Execution requires all
gates to pass.

| Helper | Use |
| --- | --- |
| `evidence_template(function)` | Read static evaluator metadata. |
| `evidence_implementation(function, source=...)` | Bind an evaluator to a source basis. |
| `evaluate_evidence(function, *args, subject=..., source=..., id=..., observed_at=..., **kwargs)` | Materialize Evidence directly. |

Evaluator exceptions and invalid outcomes become `outcome: "error"` Evidence
with a Diagnostic rather than disappearing as logs.

## Runs and release sets

### `@run(...)` and `observe_run(...)`

```python
@run(name="Daily training", artifact_sets=())
def train(*, observed): ...


with observe_run(train, publisher=publisher, source=source) as observed:
    train(observed=observed)
```

`@run` declares an application workflow; it is not a Core record or parent
Execution. `observe_run` creates a fresh run UUID, activates the runtime, and
applies the same `profiles.run` binding to the real child Executions. Supply
`run_id=` only when the application already owns that concrete UUID.
`run_template(workflow)` returns the static `RunTemplate` declaration attached
by `@run`.

For scoped observation that is not a batch run, use:

```python
with OclpRun(publisher=publisher, source=source) as observed:
    ...
```

`active_run()` returns that context or `None`.

### `RunArtifactSet`

Use `RunArtifactSet` for a package assembled from outputs of several child
computations:

```python
@run(
    name="Daily training",
    artifact_sets=(
        RunArtifactSet(
            name="Validated release",
            members={
                "model": (train_model.output("model"), "model"),
                "metrics": (evaluate_model.output("metrics"), "metrics"),
            },
            materialize_manifest=True,
            manifest_name="Validated release manifest",
        ),
    ),
)
def train(...): ...
```

The SDK resolves declared members only after a successful `observe_run` context.
Retrieve the resulting `ArtifactSetHandle` with
`observed.artifact_set("Validated release")`. Each member must resolve exactly
once; use the dynamic API below when that cannot be known beforehand.

| `OclpRun` API | Result |
| --- | --- |
| `outputs_for(call_result)` | `{port: ArtifactHandle}` for an observed call. |
| `artifact_set_outputs_for(call_result)` | `{port: ArtifactSetHandle}` for computation-level sets. |
| `execution_for(call_result)` / `computation_for(call_result)` | Exact Core reference. |
| `evidence_for(call_result)` | Evidence emitted for that call. |
| `publish_artifact_set(name=..., members=..., materialize_manifest=False, manifest_name=None)` | Publish an explicitly dynamic collection. |
| `artifact_set(name)` | Retrieve a completed declared run-level set. |

`ArtifactSetHandle.member(name)` returns a member handle;
`load_member(name, target_type)` verifies and adapts it. Use
`load_release_manifest(path)` for a materialized release manifest.

### `output_artifact_id(port)`

Within an observed computation body, use this only when a payload must include
the UUID of the Artifact that will describe it:

```python
return {"response_id": output_artifact_id("response"), "score": score}
```

The SDK reserves that UUID before executing the body and publishes the output
with exactly that identity. Most outputs do not need this helper.

## Sources

```python
source = source_from_git_checkout(project_root, path="src/model")
```

This returns `GitSource`, including `dirty: true` when applicable. To capture
the exact dirty diff as a reproducible source basis, call:

```python
source = capture_git_source_overlay(
    project_root,
    source=source,
    publisher=publisher,
    name="Source overlay",
    relative_path="source-overlays/current",
    untracked_files=(),
)
```

The helper publishes `git diff HEAD` as a patch Artifact and binds its
ArtifactSet through `GitSource.overlay`. It refuses to capture untracked files
until the application explicitly selects them.

Use Core `ServiceSource` for a deployed service rather than a Git checkout,
and `OpaqueSource` only when no portable source model applies.

## Publishing and local catalog

### `LocalArtifactPublisher`

Import it from `oclp.publishing`:

```python
publisher = LocalArtifactPublisher(
    catalog_path=Path("data/oclp.duckdb"),
    record_root=Path("data/oclp/record"),
    payload_root=Path("data/runs"),
)
```

| API | Use |
| --- | --- |
| `publish(record)` | Write and index a canonical Core record. |
| `artifact_for_bytes(...)` | Persist bytes and publish its Artifact. |
| `json_artifact(...)` | Persist deterministic JSON and publish its Artifact. |
| `artifact_for_file(...)` | Copy a file into the payload store and publish its Artifact. |
| `records()` | Read local catalog records. |
| `close()` / context manager | Release the local catalog. |

`PublishedArtifact` contains `artifact`, `path`, and `reference`. `utc_now()`
returns a timestamp suitable for record fields.

### `DuckdbCatalog`

Import `DuckdbCatalog` from `oclp.catalog.duckdb`. It is a local, rebuildable
index and resolver—not the authoritative record store.

| API | Use |
| --- | --- |
| `publish(record)`, `ingest(records)`, `ingest_directory(root)` | Add records. |
| `resolve(reference)`, `get(digest)` | Resolve and verify a record. |
| `add_location(content_digest, location)`, `locations_for(reference)` | Maintain mutable retrieval hints. |
| `artifacts_for_content(content_digest)` | Find identical payload content. |
| `records()` | Return indexed records. |

The catalog raises `RecordNotFoundError`, `AmbiguousRecordReferenceError`,
`CatalogIntegrityError`, and `CatalogResolutionError` for resolution failures.

## Canonicalization and validation

```python
from oclp import (
    canonical_json_bytes,
    parse_record,
    record_digest,
    validate_derivation_graph,
    validate_execution_acceptance,
    validate_execution_hierarchy,
)
```

| API | Use |
| --- | --- |
| `canonical_json_bytes(record)` | Produce RFC 8785 canonical record bytes. |
| `record_digest(record)` | SHA-256 digest of those canonical bytes. |
| `parse_record(value)` | Parse JSON-compatible data into a typed record. |
| `validate_derivation_graph(records)` | Validate input/output bindings and parameters. |
| `validate_execution_hierarchy(records)` | Validate parent-child Execution relationships. |
| `validate_execution_acceptance(records)` | Verify required passing Evidence. |

Validation raises `DerivationValidationError`, `OrchestrationValidationError`,
`AcceptanceValidationError`, or `ParameterValidationError`.

## Compatibility boundary

Do not depend on names prefixed with `_`, decorator attributes such as
`__oclp_computation_template__`, local record-file paths, or DuckDB table
layouts. They are implementation details. Depend on the APIs documented here,
Core records, and the explicit `oclp.catalog.duckdb.DuckdbCatalog` integration
point.

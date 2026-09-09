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
| Build a release/package | `ComputationArtifactSet`, `@artifact_set`, or `publish_artifact_set(...)` |
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

For output metadata that depends on one call's ordinary parameters, provide an
`annotation_factory`. It declares the parameters it needs by name; the SDK
matches those names to the decorated Computation call and passes their actual
values when it materializes that output. The factory returns JSON-compatible
annotations for that one immutable Artifact record. It need not declare every
Computation argument. Static `annotations` remain in place and the factory's
values are added to them; a factory value wins if both maps use the same key:

```python
def fold_annotations(*, fold_number: int) -> dict[str, int | str]:
    return {
        "fold_number": fold_number,
        "split_strategy": "temporal",
    }


@computation(
    name="Train temporal fold",
    outputs={
        "model": CatBoostModelArtifact(
            name="Temporal fold model",
            annotations={"model_family": "catboost"},
            annotation_factory=fold_annotations,
        ),
    },
)
def train_fold(*, fold_number: int): ...
```

For `train_fold(fold_number=2)`, the SDK calls
`fold_annotations(fold_number=2)` and persists this model Artifact metadata:

```json
{
  "model_family": "catboost",
  "fold_number": 2,
  "split_strategy": "temporal"
}
```

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
@run(
    name="Daily training",
    adapters=(),
    required_evidence_policy="continue",
)
def train(*, observed): ...


with observe_run(train, publisher=publisher, source=source) as observed:
    train(observed=observed)
```

`@run` declares an application workflow; it is not a Core record or parent
Execution. `observe_run` creates a fresh run UUID, activates the runtime, and
applies the same `profiles.run` binding to the real child Executions. Supply
`run_id=` only when the application already owns that concrete UUID.
`run_template(workflow)` returns the static `RunTemplate` declaration attached
by `@run`. `adapters` holds optional SDK integration objects. It can also be
supplied to `observe_run(...)` for bootstrap-time configuration.

Set `required_evidence_policy="raise"` when a workflow must stop after a real
Execution publishes failed required Evidence. The default, `"continue"`,
preserves the failed Execution and leaves application control flow unchanged.
With `"raise"`, the runtime raises `RequiredEvidenceFailedError` only *after*
outputs, every Evidence record, and the failed terminal Event have been
published. Its `execution` and `evidence` attributes identify those records.

An adapter declared on `@run` is a reusable configuration template, not its
active per-run session. The runtime calls an adapter's optional `for_run()`
method to obtain a fresh active instance for each observation. Access that
active instance from an `OclpRun` by its class:

```python
with observe_run(workflow, publisher=publisher, source=source) as observed:
    workflow()
    mlflow = observed.adapter(MlflowAdapter)
    mlflow.log_metrics({"validation_rmse": 42.1})
```

`OclpRun.adapter(AdapterClass)` returns the exactly one active adapter that is
an instance of `AdapterClass`. It raises when no adapter or more than one
adapter matches; this makes an ambiguous integration configuration visible.
The returned object is useful for application-selected external context only;
the adapter itself automatically mirrors OCLP records.

### `MlflowAdapter` (optional)

Install the optional extra first:

```bash
pip install 'oclp[mlflow]'
```

Then attach one adapter to an observed run:

```python
from oclp import MlflowAdapter, observe_run, run

@run(
    name="Daily training",
    adapters=(
        MlflowAdapter(
            experiment_name="daily-training",
            tracking_uri="sqlite:///mlflow.db",
        ),
    ),
)
def workflow(): ...

with observe_run(workflow, publisher=publisher, source=source):
    workflow()
```

The runtime calls `for_run()` when available, so an adapter declaration
receives a fresh active mirror instance for each observation.
When neither `tracking_uri` nor `artifact_location` is set, the local SDK
publisher default is a SQLite MLflow store and artifact directory beside the
OCLP record directory.

The adapter mirrors canonical OCLP record JSON, UUID tags, typed Execution
parameters, and top-level numeric Evidence details. Execution parameter keys
include their OCLP Execution UUID because MLflow parameters are immutable
within one MLflow run while a Computation may run repeatedly with different
arguments. It mirrors model payloads by default. OCLP is always authoritative:
the adapter never reads MLflow to create, validate, or query OCLP provenance.
Each mirrored payload is stored beneath its OCLP Artifact UUID, so repeated
same-named outputs (for example, temporal-fold models) remain distinct in the
MLflow artifact store.

### `@mlflow` and `MlflowMetrics`

Use `@mlflow` outside `@computation` to keep MLflow presentation policy next
to the output it projects:

```python
from oclp import JsonArtifact, MlflowMetrics, computation, mlflow

@mlflow(
    metrics=(
        MlflowMetrics(
            output_port="metrics",
            prefix="fold",
            dimensions=("fold_number",),
        ),
    ),
)
@computation(
    name="Train fold",
    outputs={"metrics": JsonArtifact(name="Fold metrics")},
)
def train_fold(*, fold_number: int) -> dict[str, object]:
    return {"metrics": {"rmse": 42.1}}
```

The selected output must be an `application/json` Artifact. The adapter reads
its verified bytes and logs only top-level numeric scalar fields. `dimensions`
must name declared scalar Execution parameters and prevent repeated invocations
from producing the same MLflow key. An attempted collision is an adapter
failure: it follows the normal Diagnostic behavior unless `strict=True`.

`@mlflow(payloads=("report",))` similarly mirrors a named non-model output
payload. Payload names are exact output ports, not artifact display names.
Model payloads always mirror; canonical OCLP record JSON always mirrors. The
declaration is inert when the active run has no `MlflowAdapter`, so the same
Computation works unchanged in environments without MLflow. Execution
parameters already mirror automatically, so the decorator is needed only for
metric projections and additional payloads.

#### Workflow-level MLflow parameters

Use the same decorator outside `@run` when a small, explicit set of workflow
arguments should also appear in MLflow. The mapping is **MLflow parameter name
to workflow argument name**:

```python
from oclp import MlflowAdapter, mlflow, run

@mlflow(
    run_parameters={
        "release_id": "release_id",
        "temporal_fold_count": "fold_count",
    },
)
@run(
    name="Daily training",
    adapters=(MlflowAdapter(experiment_name="daily-training"),),
)
def train(*, release_id: str, fold_count: int) -> None:
    ...
```

The adapter logs these as `workflow.release_id` and
`workflow.temporal_fold_count`, and tags them with
`oclp.mlflow.workflow_parameter.<name>=true`. This namespace keeps them
separate from the automatically mirrored, Execution-UUID-scoped OCLP
parameters. It also makes their provenance clear: they are application-chosen
workflow context, not Core Execution parameters.

Every selected argument must be JSON-compatible. The decorator rejects unknown
argument names when the module loads. A conflicting value for the same
`workflow.*` key within one MLflow run follows the normal adapter policy:
it becomes an integration Diagnostic by default and raises when
`MlflowAdapter(strict=True)` is configured. The declaration is inert without
an MLflow adapter and never creates an OCLP record, Execution, or derivation
edge.

`run_parameters` belongs only on `@mlflow` outside `@run`; `metrics` and
`payloads` belong only on `@mlflow` outside `@computation`.

After the adapter is declared, an application can retrieve its active session
with `observed.adapter(MlflowAdapter)` and deliberately add domain-specific
comparison values without recreating OCLP reference wiring:

```python
mlflow = observed.adapter(MlflowAdapter)
mlflow.log_metrics({"validation_rmse": 42.1})
mlflow.log_parameters({"candidate_family": "CatBoostRegressor"})
```

Use this only for values whose MLflow presentation is an application decision;
the adapter already mirrors OCLP Execution parameters and Evidence records.

By default, an MLflow error does not undo OCLP publication. The runtime emits
an `adapter-failed` Event with an integration Diagnostic on the next real
Execution. Use `strict=True` to make the mirror failure fail the application
workflow. For explicit registry publication, nominate the exact Artifact by
its application-owned name:

```python
from oclp import MlflowAdapter, MlflowModelRegistration

adapter = MlflowAdapter(
    experiment_name="daily-training",
    model_registration=MlflowModelRegistration(
        artifact_name="Validated model",
        registered_model_name="demand-model",
    ),
)
```

Registration is opt-in; the adapter does not infer promotion policy.

For scoped observation that is not a batch run, use:

```python
with OclpRun(publisher=publisher, source=source) as observed:
    ...
```

`active_run()` returns that context or `None`.

### `@artifact_set`

Use `@artifact_set` for a package assembled from outputs of one or several
child computations. Apply it outside the `@computation` that emits the
members. A single `members` mapping keeps related outputs from one computation
together:

```python
@artifact_set(
    name="Validated release",
    members={
        "model": ("model", "model"),
        "metrics": ("metrics", "validation-report"),
    },
)
@computation(
    name="Train model",
    outputs={
        "model": CatBoostModelArtifact(name="Validated model"),
        "metrics": JsonArtifact(name="Validation metrics"),
    },
)
def train_model(...): ...


@artifact_set(
    name="Validated release",
    output_port="features",
    role="training-data",
)
@computation(
    name="Prepare features",
    outputs={"features": CsvArtifact(name="Training features")},
)
def prepare_features(...): ...


@run(
    name="Daily training",
)
def train(...): ...
```

#### Decorator order

`@computation` must be closest to the function. Both `@artifact_set` and
`@mlflow` read its declared output ports, so they must be applied outside it.
When a Computation uses both, their order relative to one another does not
matter:

```python
@artifact_set(name="Validated release", output_port="metrics")
@mlflow(metrics=(MlflowMetrics(output_port="metrics", prefix="validation"),))
@computation(...)
def evaluate_model(...): ...
```

Python applies decorators from the bottom up. Therefore placing `@computation`
outside either declaration decorator fails because the declaration would run
before a Computation template exists.

The SDK resolves the local declarations only after a successful observed
context. Retrieve the resulting `ArtifactSetHandle` with
`observed.artifact_set("Validated release")`. Each member must materialize
exactly once. The `members` keys are the member names; the concise
`output_port=` form uses the output port as its default member name, with
`member_name=` available for a one-output override. Use the dynamic API below
when members cannot be known beforehand. The SDK creates a durable package
representation automatically, using the same `name` as the ArtifactSet.

| `OclpRun` API | Result |
| --- | --- |
| `outputs_for(call_result)` | `{port: ArtifactHandle}` for an observed call. |
| `artifact_set_outputs_for(call_result)` | `{port: ArtifactSetHandle}` for computation-level sets. |
| `execution_for(call_result)` / `computation_for(call_result)` | Exact Core reference. |
| `evidence_for(call_result)` | Evidence emitted for that call. |
| `publish_artifact_set(name=..., members=..., materialize_manifest=False, manifest_name=None)` | Publish an explicitly dynamic collection. |
| `artifact_set(name)` | Retrieve a completed decorator-assembled set. |

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

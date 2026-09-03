# Instrument a computation

OCLP records durable computation facts; it does not prescribe a storage
convention or an orchestration engine. The Python SDK keeps reusable
declarations beside ordinary functions and can, when explicitly activated,
turn a declared returned value into a real Artifact and Execution.

The normative field contract is the [OCLP specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/).

## Names are application-owned labels

Every record-producing SDK declaration requires an application-supplied
`name`: Computations, Evidence evaluators, Artifact-producing decorators,
materialized computation outputs, lifecycles, ArtifactSets, and materialized
release manifests. The SDK carries that label into the resulting record; it
does not derive one from a function name, port, run ID, digest, timestamp, or
filesystem path.

Use a concise label that reads well in an explorer, such as `"Train fold"` or
`"Validation metrics"`. Keep exact instance identity in the record `id`, a
lifecycle `run_id`, input and output references, parameters, and timestamps.
An Artifact representation class used *only as an input declaration* (for
example, `inputs={"table": CsvArtifact}`) has no produced record and needs no
name.

When a local publisher sees the same logical Artifact ID and payload digest
again, it reuses the existing record only if its declared immutable metadata,
including `name`, matches. Changing a declared label therefore creates a new
record revision with the same payload digest instead of silently retaining an
old label. Retrieval locations remain catalog metadata rather than a reason to
rewrite an OCLP record.

## Declare the computation and its success check

An `@evidence` evaluator defines a reusable, source-bound validation gate. It
returns exactly one outcome—`"pass"`, `"fail"`, or `"error"`—for one
application-owned rule. The runtime creates the single core Evidence record
from that outcome. A `@computation` names an ordinary callable and can directly
require that evaluator—there is no separate ID/version reference or
requirement wrapper.

```python
from oclp import computation, evidence
from oclp.models import PortDefinition


@evidence(name="Normalized report quality")
def normalized_report_quality(report: dict[str, object]) -> str:
    title = str(report["title"])
    return "pass" if title else "fail"


@computation(
    id="urn:example:computation:normalize-report",
    name="Normalize report",
    input_ports=(PortDefinition(name="source", media_types=("application/json",)),),
    output_ports=(PortDefinition(name="report", media_types=("application/json",)),),
    requires=(normalized_report_quality,),
)
def normalize_report(source: dict[str, object]) -> dict[str, object]:
    return {"title": str(source["title"]).strip()}
```

`@evidence` preserves ordinary call semantics and attaches static metadata.
Separate checks should be separate evaluators and Evidence records—not a
second SDK-only Evidence return type.
`@computation` does the same unless it declares runtime `outputs` inside an
active `OclpRun`; that opt-in path is described below. At publication time the
source actually selected for the run binds both records to exact source
revisions.

```python
from oclp import GitSource, computation_record

source_basis = GitSource(
    repository="https://github.com/example/reports.git",
    commit="0123456789abcdef0123456789abcdef01234567",
    path="src/reports/normalize.py",
)

report_computation = computation_record(normalize_report, source=source_basis)
```

`report_computation.required_evidence` contains the exact source-bound evaluator
Implementation. The same binding appears in each resulting Evidence record;
changing the evaluator code or observed source produces a different binding.

### Declared input ports match callable parameters

When a Python Computation explicitly declares `input_ports`, each port name
must match a real function parameter (excluding `self` and `cls`). The SDK
raises an error when the decorator is applied if they differ. This keeps the
portable Computation interface and the Python call boundary honest:

```python
@computation(
    id="urn:example:computation:prepare-report",
    name="Prepare report",
    input_ports=(PortDefinition(name="source_snapshot"),),
)
def prepare_report(source_snapshot: pd.DataFrame) -> pd.DataFrame:
    return source_snapshot.dropna()
```

### Ordinary arguments become declared Execution parameters

Artifact inputs are declared at the decorator boundary. Every other ordinary,
JSON-compatible function argument is inferred as a `ParameterDefinition` on the
portable Computation record—there is no second `parameters=` block to keep in
sync. Required arguments remain required; Python defaults become JSON Schema
`default` values and are recorded as effective values on each Execution.

```python
from oclp import CsvArtifact, JsonArtifact, computation


@computation(
    id="urn:example:computation:train-fold",
    name="Train one temporal fold",
    inputs={
        "feature_table": CsvArtifact,
        "fold_definition": JsonArtifact,
    },
    outputs={"metrics": JsonArtifact(name="Fold metrics")},
)
def train_fold(
    feature_table: pd.DataFrame,
    fold_definition: dict[str, object],
    *,
    fold_number: int,
    training_window: Literal["pre-holdout"] = "pre-holdout",
) -> dict[str, object]:
    ...
```

This produces parameter definitions equivalent to:

```json
[
  {"name": "fold_number", "schema": {"type": "integer"}, "required": true},
  {
    "name": "training_window",
    "schema": {
      "type": "string",
      "enum": ["pre-holdout"],
      "default": "pre-holdout"
    },
    "required": false
  }
]
```

Within an `OclpRun`, calling `train_fold(..., fold_number=2)` publishes an
Execution with `parameters: {"fold_number": 2,
"training_window": "pre-holdout"}`. The SDK binds Python defaults before it
publishes, so an omitted optional argument is still reproducible. It infers
schemas for JSON-safe built-ins, containers, unions, and `Literal` values.
Non-JSON implementation plumbing such as `pathlib.Path` is intentionally not
captured. Make durable content an input Artifact instead; keep a local output
path as runtime-only plumbing.

An Execution then records the Artifact reference at
`inputs["source_snapshot"]`. If a function currently accepts an intermediate
object such as `frame` or `results` rather than the Artifact-oriented input it
claims to consume, it should not declare the mismatched port yet. Migrate that
stage to a real decorator-bound input first; do not use a misleading name just
to make a graph look more complete.

## Acquire an external value as an Artifact

Use `@csv_artifact` for an effectful boundary that obtains a durable tabular input instead
of deriving one. Its function body returns a `pandas.DataFrame`, while its
public decorated call returns a `CsvArtifact` handle in an active `OclpRun`.
It does **not** create a Computation, Execution, or lifecycle Event.

```python
import pandas as pd

from oclp import CsvArtifact, OclpRun, computation, csv_artifact


@csv_artifact(
    name="UCI Bike Sharing hourly source snapshot",
    index=False,
    lineterminator="\n",
)
def download_source_csv() -> pd.DataFrame:
    return fetch_the_external_table()


with OclpRun(...) as observed:
    source_snapshot = download_source_csv()
```

`@csv_artifact` validates that the function returned a `pandas.DataFrame`,
serializes it with those options, writes the CSV payload, hashes it, publishes
the Artifact, and returns its `CsvArtifact` handle. This is appropriate for a
database extract or API client that exposes a table object. If the external
system exposes original bytes, use a byte-oriented Artifact contract instead;
a CSV snapshot is not a claim to preserve the provider's response
byte-for-byte. Artifact-producing decorators require an active `OclpRun`,
because publishing a durable Artifact requires a configured store.

### Adapter-driven Artifact inputs

The OCLP protocol records an Artifact reference, while a Python function needs
an in-memory value. The SDK bridges those distinct concerns with an adapter:

```python
@computation(
    id="urn:example:computation:prepare-report",
    name="Prepare report",
    inputs={"source_snapshot": CsvArtifact},
)
def prepare_report(source_snapshot: pd.DataFrame) -> pd.DataFrame:
    # Ordinary pandas code; no OCLP loader or location logic here.
    return source_snapshot.dropna()


with OclpRun(...) as observed:
    source_snapshot = download_source_csv()  # CsvArtifact
    prepared = prepare_report(source_snapshot)  # receives pd.DataFrame
```

`inputs={"source_snapshot": CsvArtifact}` is SDK metadata, not a new protocol
record type. It derives the portable input port
`{"name": "source_snapshot", "media_types": ["text/csv"]}` and requires a
`CsvArtifact` at the runtime call boundary. The `pd.DataFrame` annotation still
describes the ordinary in-memory value delivered to the function body. The
registered `PandasCsvAdapter` verifies the payload SHA-256 digest, reads the
CSV, and supplies the resulting DataFrame. The Execution records the original
digest-bound Artifact reference in `inputs["source_snapshot"]`.

The annotation describes the value that the function actually receives; the
Artifact handle identifies the durable representation. The adapter registry
selects a loader from that pair. A representation is therefore not coupled to
one in-memory library or type.

| Durable Artifact handle | Durable representation | Requested runtime type | Built-in adapter |
| --- | --- | --- |
| `CsvArtifact` | UTF-8 `text/csv` | `pd.DataFrame` | `PandasCsvAdapter` |
| `ParquetArtifact` | `application/vnd.apache.parquet` | `pd.DataFrame` | `PandasParquetAdapter` |
| `JsonArtifact` | `application/json` | `dict[...]` or `Mapping[...]` | `JsonMappingAdapter` |
| `JsonArtifact` | pandas `orient="table"` JSON | `pd.DataFrame` | `PandasJsonTableAdapter` |

Install the optional Parquet implementation with `oclp[parquet]`. Generic JSON
loads as a mapping when the callable asks for one. Loading JSON as a DataFrame
is deliberately narrower: the adapter validates pandas' table-JSON structure
before it calls pandas. If no single registered adapter supports the Artifact
and annotated target type, the SDK fails before the function body begins. A
port's optional `media_types` constraint can restrict which durable forms a
Computation accepts.

### Exercise multiple source representations

The bike-demand reference project includes a small source-format factory to
prove that format choice remains at the Artifact boundary. All three decorated
acquisition functions fetch the same logical UCI table, but each persists a
different byte representation and therefore has its own Artifact ID and
digest:

```python
from oclp import csv_artifact, json_artifact, parquet_artifact


@csv_artifact(id="urn:example:artifact:uci-hourly:275:csv", name="UCI source (CSV)")
def download_csv() -> pd.DataFrame:
    return fetch_the_external_table()


@parquet_artifact(
    id="urn:example:artifact:uci-hourly:275:parquet",
    name="UCI source (Parquet)",
    compression="zstd",
)
def download_parquet() -> pd.DataFrame:
    return fetch_the_external_table()


@json_artifact(
    id="urn:example:artifact:uci-hourly:275:json",
    name="UCI source (pandas table JSON)",
    schema_uri="urn:example:schema:pandas-dataframe-table-json:v1",
    serialization="pandas-table",
)
def download_json() -> pd.DataFrame:
    return fetch_the_external_table()
```

All three decorated calls return handles, not DataFrames. A normal Computation
can accept all three forms while preserving a plain pandas signature:

```python
@computation(
    id="urn:example:computation:inspect-source",
    name="Inspect source",
    input_ports=(
        PortDefinition(
            name="source_snapshot",
            media_types=(
                "text/csv",
                "application/vnd.apache.parquet",
                "application/json",
            ),
        ),
    ),
)
def inspect_source(source_snapshot: pd.DataFrame) -> int:
    return len(source_snapshot)
```

The caller chooses a durable representation; the function remains ordinary
pandas code. This is the boundary we want to dogfood: storage format is
explicit in one decorator, while the computation consumes its declared
in-memory type through a verified adapter.

`@parquet_artifact` requires the SDK's `parquet` extra because it uses
`pyarrow`; its `index` and `compression` arguments are serialization policy.
`@json_artifact` defaults to ordinary JSON-compatible values. Set
`serialization="pandas-table"` explicitly for a DataFrame's
`to_json(orient="table", index=False, date_format="iso")` representation.
Both return `JsonArtifact`; the consumer annotation selects the appropriate
adapter. As with CSV, changing a representation option can change the
immutable payload digest.

### Artifact fields: declared intent and derived facts

An Artifact record intentionally contains both configuration and observations.
The decorator declares the representation the application wants. The SDK then
records facts about the exact bytes it actually persisted. Keeping these roles
separate prevents a caller from accidentally claiming a digest, size, time, or
location that does not match the materialization.

| Artifact field | Source in the current CSV decorator | Why |
| --- | --- | --- |
| `id` | Explicit `id=` string or resolver; otherwise SDK-generated from the namespace, callable, and run ID | An Artifact ID is logical identity, not its content digest. For a known immutable source, make this stable and let the digest validate its bytes. |
| `name` | Required `name=` decorator argument | Concise human-readable label for people and UIs; the ID and lifecycle profile carry run-specific identity. |
| `kind` | SDK constant: `artifact` | The decorator creates one Artifact record. |
| `media_type` | SDK constant: `text/csv` | The concrete decorator defines the persisted representation. |
| `digest` | SDK SHA-256 of the serialized CSV bytes | Integrity must describe the real payload. |
| `size` | SDK byte count | It must describe the real payload. |
| `created_at` | SDK materialization timestamp | It records when this immutable payload was created. |
| `locations` | Active store's resulting payload URI | A location is a property of a particular storage implementation. |
| `schema_uri` | Optional `schema_uri=` decorator argument | Declares the schema expected for the persisted representation. |
| `profiles`, `annotations` | Optional decorator arguments | Declare extension-profile bindings and portable application metadata. |

`index`, `lineterminator`, `na_rep`, `float_format`, `date_format`, and
`columns` are CSV serialization options, not OCLP record fields. They still
matter to the contract because changing any of them can change the persisted
bytes and therefore the digest.

```python
@csv_artifact(
    id=lambda *, dataset_id: (                       # resolved from function call
        f"urn:example:artifact:uci-hourly:{dataset_id}"
    ),
    name="UCI Bike Sharing hourly source snapshot",  # Artifact name
    index=False,                                      # do not write DataFrame index
    lineterminator="\n",                             # platform-independent bytes
    columns=("timestamp", "temperature", "demand"), # intentional CSV projection
    schema_uri="urn:example:schema:bike-demand:v1",   # representation contract
    annotations={"source": "UCI"},                   # portable application metadata
)
def download_source_csv(dataset_id: int = 275) -> pd.DataFrame:
    return fetch_the_external_table()
```

An `id=` resolver receives the decorated callable's bound named arguments.
For an immutable source, repeated calls with the same logical ID must produce
the same bytes: the local publisher reuses the first Artifact record when the
digest matches and rejects a different digest for that ID. This is why the
source's `dataset_id` belongs in its Artifact ID here, rather than in a
run-specific SDK key.

The active `OclpRun` is deliberately separate from the decorator: it supplies
the publication context—store, namespace, and run ID—not the Artifact's CSV
serialization policy. The function remains the sole place an application
declares that policy.

## Materialize a simple computation output automatically

For a straightforward durable output, declare its port and its persistence
representation directly on the callable. The port name is no longer
disconnected metadata: it names the Artifact produced from the return value.
The Python return type and the durable Artifact format are separate decisions.

```python
from pathlib import Path

import pandas as pd

from oclp import CsvArtifact, GitSource, OclpRun, computation
from oclp.publishing import LocalArtifactPublisher


@computation(
    id="urn:example:computation:fetch-report",
    name="Fetch report",
    outputs={
        "source_snapshot": CsvArtifact(
            name="Fetched report snapshot",
            key="report-source",
            path="reports/source.csv",
            schema_uri="urn:example:schema:report:v1",
        ),
    },
)
def fetch_report() -> pd.DataFrame:
    return pd.DataFrame({"value": [1, 2, 3]})


with LocalArtifactPublisher(
    catalog_path=Path("data/oclp/catalog.duckdb"),
    record_root=Path("data/oclp"),
    payload_root=Path("data/runs/example-run"),
) as publisher:
    with OclpRun(
        publisher=publisher,
        namespace="urn:example",
        run_id="example-run",
        source=source_basis,
    ) as observed:
        dataframe = fetch_report()
        source_snapshot = observed.outputs_for(dataframe)["source_snapshot"]
```

The wrapper returns the original `dataframe`; it does not replace application
objects with an OCLP proxy. `outputs_for(...)` returns an internal resolved
Artifact handle for the materialized `source_snapshot`, suitable for passing to
a later Computation. The `CsvArtifact(...)` declaration is the explicit
decision to persist a CSV snapshot, including its name, logical ID component,
payload path, schema, and pandas serialization options. The SDK publishes the source-bound Computation if
needed, that Artifact, an Execution whose
`outputs["source_snapshot"]` references it, and start/publication/terminal
Events. Passing the returned output handle to another declared Computation
records the exact Artifact reference and loads its verified bytes through the
adapter selected by that function's parameter annotation.

An output mapping always needs an explicit concrete Artifact type. `CsvArtifact`
persists a table snapshot, `JsonArtifact` persists JSON-compatible data,
`BytesArtifact` persists bytes, and `FileArtifact` copies an existing file. A
domain type can own both sides of the boundary: `CatBoostModelArtifact`, for
example, persists a returned fitted CatBoost model as native `.cbm` bytes and
loads a verified Artifact into a later `CatBoostRegressor` parameter. For more
than one direct output, each port is a key in the returned mapping. There is no
separate `attribute=` selector to keep in sync.

```python
@computation(
    id="urn:example:computation:prepare-report",
    name="Prepare report",
    outputs={
        "table": CsvArtifact(name="Prepared report"),
        "report_contract": JsonArtifact(name="Report contract"),
    },
)
def prepare_report() -> dict[str, object]:
    return {
        "table": ...,
        "report_contract": ...,
    }
```

The same Artifact type names describe both ends of a specialized boundary. A
fitted CatBoost model needs no runner-provided path or application-owned
serializer:

```python
import pandas as pd
from catboost import CatBoostRegressor
from oclp import CatBoostModelArtifact, CsvArtifact, computation


@computation(
    id="urn:example:computation:train-model",
    name="Train model",
    inputs={"features": CsvArtifact},
    outputs={"model": CatBoostModelArtifact(name="Candidate model")},
)
def train_model(features: pd.DataFrame) -> CatBoostRegressor:
    model = CatBoostRegressor(verbose=False)
    return model.fit(features[["temperature"]], features["demand"])
```

`CatBoostModelArtifact` saves the returned model to temporary native `.cbm`
bytes and publishes those bytes as the Artifact payload. A later computation
can declare `inputs={"model": CatBoostModelArtifact}` and annotate its normal
Python parameter as `CatBoostRegressor`; the SDK verifies the digest and loads
the model before it calls the function.

For a port that consumes several Artifacts of one representation, use
`many(...)`. The OCLP record gets `cardinality: "many"`, while the function
receives its normal typed collection after every member is verified and loaded.

```python
import pandas as pd

from oclp import CsvArtifact, JsonArtifact, computation, many


@computation(
    id="urn:example:computation:aggregate-folds",
    name="Aggregate fold predictions",
    inputs={"fold_predictions": many(CsvArtifact)},
    outputs={"evaluation": JsonArtifact(name="Fold evaluation")},
)
def aggregate_folds(
    fold_predictions: tuple[pd.DataFrame, ...],
) -> dict[str, object]:
    return {"evaluation": summarize(fold_predictions)}
```

Required Evidence can remain equally local. An evaluator must accept exactly
one parameter; for a multi-output result that parameter name selects the
returned output it evaluates. `OclpRun` publishes its Evidence before the
terminal Event and uses a failed terminal status when any required evaluator
fails.

```python
@evidence(name="Quality")
def quality(evaluation: dict[str, float]) -> str:
    return "pass" if evaluation["rmse"] <= 1.0 else "fail"


@computation(
    id="urn:example:computation:evaluate",
    name="Evaluate",
    outputs={"evaluation": JsonArtifact(name="Evaluation")},
    requires=(quality,),
)
def evaluate(...) -> dict[str, object]:
    return {"evaluation": {"rmse": 0.8}}
```

`CsvArtifact` writes deterministic CSV deliberately, while `ParquetArtifact`
is available as an optional `oclp[parquet]` integration. Both
`CatBoostModelArtifact` and `XGBoostModelArtifact` are SDK-owned model
integrations rather than application boilerplate. See [Artifact formats and
library integrations](integrations.md) for their exact native persistence
formats and compatibility rules.

### Publish a release ArtifactSet from exact handles

An ArtifactSet is a Core collection record, not a storage format with bytes to
serialize and not necessarily the output of a Computation. Publishing a named
collection of already-materialized Artifacts is an SDK operation, so it must
not fabricate a function, Computation, Execution, or lifecycle Event merely to
make the graph look connected.

```python
from oclp import OclpRun


with OclpRun(...) as observed:
    model_release = observed.publish_artifact_set(
        key="candidate-model",
        name="Validated candidate model release",
        members={
            "model": (model, "model"),
            "evaluation": (evaluation, "validation-report"),
            "features": (features, "training-data"),
        },
        materialize_manifest=True,
        manifest_name="Validated candidate release manifest",
    )
```

Each dictionary key is the stable ArtifactSet member name. Its value is a
two-item tuple of an exact `ArtifactHandle` and an optional semantic role:
`dict[str, tuple[ArtifactHandle, str | None]]`. The SDK publishes an immutable
ArtifactSet whose members contain those handles' digest-bound references and
returns an `ArtifactSetHandle`. When the `OclpRun` carries profiles such as a
lifecycle profile, the ArtifactSet carries those same bindings so a viewer can
associate this direct publication with the run. No member helper, record-ID
construction, or application-owned publisher call is required.

`materialize_manifest=True` asks the SDK to persist a deterministic
`release-manifest.json` sidecar Artifact. The application must supply its
`manifest_name`; the SDK returns its handle as `model_release.manifest`. It is
not an ArtifactSet member: the sidecar records the exact digest-bound set
reference, and making it a member would create a self-content cycle. The JSON
snapshot contains the exact ArtifactSet record plus
the resolved upstream OCLP record closure available in the local publisher:
their exact canonical Artifact, Execution, Computation, Evidence, Event, and
nested ArtifactSet records as applicable. It does not duplicate model or
dataset bytes.

Because the sidecar is outside the collection, it records the final
ArtifactSet ID *and* digest. `load_release_manifest(path)` verifies that exact
binding and returns an `ArtifactSetHandle` with verified local handles for its
members. The local publisher stores the sidecar under a collision-safe
`release/.../release-manifest.json` payload path.

### Consume a release as one real ArtifactSet input

An inference or deployment Computation should consume the release collection,
not smuggle its identity through a string parameter while separately accepting
only the model file. `artifact_set_input(...)` records the ArtifactSet's exact
reference in `Execution.inputs`, validates the named members a callable needs,
and leaves the function in control of which verified members it materializes.

```python
from catboost import CatBoostRegressor

from oclp import (
    ArtifactSetHandle,
    CatBoostModelArtifact,
    JsonArtifact,
    artifact_set_input,
    computation,
)


@computation(
    id="urn:example:computation:predict",
    name="Predict demand",
    inputs={
        "model_release": artifact_set_input(
            {
                "model": CatBoostModelArtifact,
                "feature-contract": JsonArtifact,
            }
        ),
        "request": JsonArtifact,
    },
)
def predict(model_release: ArtifactSetHandle, request: dict[str, object]) -> float:
    model = model_release.load_member("model", CatBoostRegressor)
    contract = model_release.load_member("feature-contract", dict[str, object])
    return float(model.predict(build_frame(request, contract))[0])
```

The produced `Execution` has a real `model_release` input pointing to the
immutable ArtifactSet. A graph viewer can therefore show the release handoff
without an application-specific naming convention.

Use a real `@computation` only when release creation itself does work—for
example, evaluating a promotion policy, signing a manifest, or publishing to a
model registry. That Computation may consume or produce other Artifacts; the
final named ArtifactSet remains a direct collection-publication operation.

## Publish one execution and its Evidence

The optional `LocalArtifactPublisher` writes immutable payloads, records, and a
DuckDB catalog. Another store can implement the same record contract. Use its
explicit methods when the output has domain-specific naming, profiles,
membership, or serialization that should not be inferred from a function
return value.

```python
from datetime import UTC, datetime
from pathlib import Path

from oclp import Execution, evaluate_evidence
from oclp.publishing import LocalArtifactPublisher


with LocalArtifactPublisher(
    catalog_path=Path("data/oclp/catalog.duckdb"),
    record_root=Path("data/oclp"),
    payload_root=Path("data/runs/example-run"),
) as publisher:
    computation_ref = publisher.publish(report_computation)

    source = publisher.json_artifact(
        artifact_id="urn:example:artifact:source:example-run",
        name="Source document",
        relative_path="source.json",
        value={"title": "  Report  "},
        created_at=datetime.now(UTC),
    )
    report_value = normalize_report({"title": "  Report  "})
    result = publisher.json_artifact(
        artifact_id="urn:example:artifact:result:example-run",
        name="Normalized report",
        relative_path="result.json",
        value=report_value,
        created_at=datetime.now(UTC),
    )
    execution = Execution(
        id="urn:example:execution:normalize-report:example-run",
        computation=computation_ref,
        inputs={"source": (source.reference,)},
        outputs={"report": (result.reference,)},
    )
    execution_ref = publisher.publish(execution)
    publisher.publish(
        evaluate_evidence(
            normalized_report_quality,
            report_value,
            id="urn:example:evidence:normalize-report:example-run:quality",
            subject=execution_ref,
            source=source_basis,
            observed_at=datetime.now(UTC),
        )
    )
```

`evaluate_evidence()` calls the evaluator and records its exact source-bound
Implementation in the resulting Evidence. A terminal Event can claim
`status="succeeded"` only after every required evaluator has Evidence about the
exact Execution with `outcome="pass"`.

The SDK evaluates every required evaluator by default, even when an earlier
gate returns `"fail"` or raises. An evaluator exception is published as
`Evidence(outcome="error")` with a portable Diagnostic; remaining gates still
run, and the terminal Execution is `failed`. This gives operators the complete
set of gate results in a single run. The protocol permits other implementation
strategies, but never permits a successful terminal status until all required
evaluators have passed.

## Record lifecycle and validate

Publish lifecycle Events explicitly. The optional lifecycle profile convention
uses `execution-started` at sequence 0 and an optional terminal
`execution-terminal`; it has no request event or nested attempt identity.

```python
from oclp import Event, validate_execution_acceptance

publisher.publish(
    Event(
        id="urn:example:event:normalize-report:example-run:terminal",
        execution=execution_ref,
        event_type="execution-terminal",
        occurred_at=datetime.now(UTC),
        sequence=1,
        status="succeeded",
    )
)
validate_execution_acceptance(publisher.records())
```

For complete stores, also call `validate_derivation_graph()` and
`validate_execution_hierarchy()`. See the [bike-demand example](bike-demand-example.md)
for a multi-stage CatBoost run that uses these decorators, local Artifacts,
Evidence, lifecycle Events, and MLflow correlation.

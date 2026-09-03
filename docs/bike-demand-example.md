# Bike-demand reference project

The SDK repository includes a self-contained consumer project at
[`examples/bike-demand-service`](https://github.com/EvanZ/oclp-python/tree/main/examples/bike-demand-service).
It dogfoods OCLP on a public UCI Bike Sharing dataset without adding
data-science or MLflow dependencies to the `oclp` package itself.

It is a reference project, not an SDK feature or a prescribed architecture.
The application owns its computation boundaries, storage, ML workflow, and
contracts. OCLP makes its durable observations interoperable.

## Project layout

| Area | Responsibility | OCLP role |
| --- | --- | --- |
| `data.py` | Downloads the UCI source and prepares leakage-safe temporal features. | Declares a CSV Artifact acquisition and the feature-preparation Computation. |
| `modeling.py` | Declares the training-plan Artifact, trains CatBoost folds and final model, evaluates, and scores holdout data. | Declares reusable model boundaries. |
| `environment.py` | Resolves local OCLP, MLflow, and payload directories. | Local-only execution environment; not a durable run input. |
| `runner.py` | Declares and coordinates the real model lifecycle. | Uses SDK `@lifecycle` / `observe_lifecycle(...)`, passes persisted outputs into training, and directly publishes the final ArtifactSet from exact handles. |
| `oclp.publishing` | Writes immutable payload bytes, hashes them, and persists canonical records. | Generic local persistence; no bike-specific policy. |
| `mlflow.py` | Owns all interaction with local MLflow. | Opens MLflow runs, logs application-selected metrics/parameters, links OCLP references, and mirrors immutable payloads. |

All generated data is local and ignored by Git:

```text
data/
  runs/<run-id>/       # payload bytes: CSV, CatBoost models, JSON reports
  oclp-0.2-evidence/   # canonical OCLP records and producer catalog
  mlflow/              # local MLflow SQLite metadata and its own artifacts
```

## What it does

The batch milestone executes a time-ordered CatBoost regression workflow:

```text
UCI source CSV
  -> feature table + temporal-fold definition
  -> three fold-training Executions
  -> candidate evaluation and Evidence
  -> final model
  -> model-release ArtifactSet
  -> offline holdout predictions and Evidence
  -> release-inference smoke-test lifecycle
```

The `@lifecycle` declaration on `run_bike_lifecycle` lets the SDK derive the
same `profiles.lifecycle.run_id` for every real Execution in the batch.
Cyclops uses that explicit profile identity to group the batch as one
lifecycle run; it does **not** add a synthetic root Execution or an
orchestration edge. The data-derivation graph remains explicit: Artifacts and
ArtifactSets flow into Executions, which produce new Artifacts or ArtifactSets.
OCLP's graph and execution-acceptance validators run at the end of the demo.

After the batch lifecycle publishes its model-release ArtifactSet and manifest,
`run_demo` opens a second lifecycle, **Release inference smoke test**, with run
ID `<batch-run-id>-release-smoke`. It resolves the released model only through
that manifest, persists one fixed request and response, and requires
`Prediction response validation` Evidence to pass. This is deliberately a
linked sibling lifecycle: the released model Artifact connects it to training,
but it is not a synthetic child Execution of the training workflow. The smoke
test calls the same decorated prediction callable that FastAPI uses; it does
not start an HTTP server.

Each reusable computation is declared beside its actual Python callable with
`@oclp.computation`. At run time the demo adds its observed Git source via
`computation_record`, so Computation locators are derived from the functions
rather than copied as hand-maintained strings in the runner.

## How this project uses OCLP

### 1. Source acquisition declares an Artifact boundary

Most of the example still uses explicit Artifact materialization for rich
outputs. The UCI fetch is not a derived Computation: it is a network
acquisition boundary. Its decorator declares the source Artifact's CSV
representation, while the SDK derives facts about the materialized bytes.
`@csv_artifact` makes the returned DataFrame a persisted CSV source snapshot in an
active `OclpRun`, then returns a `CsvArtifact` handle. The downstream
feature-preparation Computation receives a verified pandas DataFrame loaded by
the SDK's CSV-to-pandas adapter.

```python
from oclp import csv_artifact


@csv_artifact(
    id=lambda *, dataset_id: (
        f"urn:oclp-bike-demand:artifact:uci-bike-sharing-hourly:{dataset_id}:csv"
    ),
    name="UCI Bike Sharing hourly source snapshot (CSV)",
    index=False,
    lineterminator="\n",
)
def download_source_csv(dataset_id: int = UCI_BIKE_SHARING_DATASET_ID) -> pd.DataFrame:
    dataset = fetch_ucirepo(id=dataset_id)
    ...
    return frame
```

The decorated call returns a `CsvArtifact` handle, not an OCLP proxy. The
function body remains ordinary pandas code and runs only inside an active
`OclpRun`, where a store exists to persist its immutable payload. It creates no
Computation, Execution, or lifecycle Event.

The next boundary retains its `pd.DataFrame` parameter and ordinary feature
logic. The runtime receives the resolved CSV Artifact handle, verifies its digest,
loads it through `PandasCsvAdapter`, and records the exact reference on the
Execution input port before calling the function body.

Its ordinary `fold_count: int = 3` argument is not an input Artifact. The
`@computation` decorator infers it as an optional Computation parameter with
JSON Schema `{"type": "integer", "default": 3}` and records the effective
value on the feature-preparation Execution. The same rule captures
`train_fold(..., fold_number: int)` and
`train_final_model(..., training_window: Literal[...])`. The SDK itself stages
the returned CatBoost model while materializing `CatBoostModelArtifact`, so no
local model path belongs in the portable computation contract.
The UCI client returns tabular Python objects rather than the original remote
response bytes, so this Artifact is intentionally a reproducible CSV snapshot
of the fetched table—not a claim to preserve the provider's exact download.
The feature-preparation boundary remains a useful example of an explicit
multi-output contract:

```python
@computation(
    id="urn:oclp-bike-demand:computation:prepare-features",
    name="Prepare bike demand features",
    inputs={
        "source_snapshot": CsvArtifact,
        "training_plan": JsonArtifact,
    },
    outputs={
        "features": CsvArtifact(name="Bike demand features"),
        "fold_definition": JsonArtifact(name="Temporal fold definition"),
        "feature_contract": JsonArtifact(name="Feature contract"),
    },
)
def prepare_features(
    source_snapshot: pd.DataFrame,
    training_plan: dict[str, object],
) -> dict[str, object]:
    # Ordinary domain logic: normalize, remove leakage, and create time folds.
    ...
```

This static-declaration pattern is used by
[`data.py`](https://github.com/EvanZ/oclp-python/blob/main/examples/bike-demand-service/src/bike_demand_service/data.py),
[`modeling.py`](https://github.com/EvanZ/oclp-python/blob/main/examples/bike-demand-service/src/bike_demand_service/modeling.py),
and `runner.py`. It avoids the drift-prone alternative of maintaining a
separate table of string locators.

### Source-format factory: CSV, Parquet, and table JSON

The default batch pipeline stays on CSV. Separately, `data.py` includes a
small `download_source_artifact()` factory that dogfoods three SDK-owned
representation decorators against the same UCI fetch:

| Factory choice | Decorator | Persisted format | Returned handle | pandas adapter |
| --- | --- | --- | --- | --- |
| `"csv"` | `@csv_artifact` | `text/csv` | `CsvArtifact` | `PandasCsvAdapter` |
| `"parquet"` | `@parquet_artifact` | `application/vnd.apache.parquet` | `ParquetArtifact` | `PandasParquetAdapter` |
| `"json"` | `@json_artifact(serialization="pandas-table")` | pandas `orient="table"` `application/json` | `JsonArtifact` | `PandasJsonTableAdapter` |

The format suffix belongs in each Artifact's logical ID—for example,
`...:uci-bike-sharing-hourly:275:parquet`. These are different immutable
representations with different bytes, so sharing an ID would incorrectly claim
that they are the same Artifact. `dataset_id` still comes directly from the
acquisition callable's parameter through its `id=` resolver.

The source-format contract test defines its own test-local input-only
Computation. Passing it each of the three handles proves that the SDK verifies
the payload digest, deserializes it using the registered adapter, and passes an
equivalent normalized DataFrame to a normal Python function. That probe is not
part of the bike-demand application or any production run graph.

### 2. The SDK observes one declared lifecycle and materializes its source-bound Computations

`runner.py` declares its actual workflow with `@lifecycle`, resolves one Git
source basis for the checkout, and activates it once with
`observe_lifecycle(...)`. Each observed decorated function then materializes
and publishes its source-bound Computation record itself. The SDK derives
`implementation.locator` directly from the function—for example,
`bike_demand_service.data.prepare_features`—and the runner can retrieve the
resulting reference from the observed result for the optional MLflow bridge.

```python
environment = DemoEnvironment.default()
source = source_from_git_checkout(
    environment.project_root,
    path="examples/bike-demand-service/src/bike_demand_service",
)

with observe_lifecycle(
    run_bike_lifecycle,
    publisher=publisher,
    run_id=run_id,
    source=source,
) as observed:
    lifecycle_result = run_bike_lifecycle(
        observed=observed,
        tracker=tracker,
        run_id=run_id,
        fold_count=3,
        temporal_validation_rmse_max=250,
    )
```

There is no bike-specific Computation registry and no pre-publication pass.
The runtime records exactly the Computations that actually execute.

### 3. Decorated feature preparation owns its ordinary outputs

The SDK's `LocalArtifactPublisher` is intentionally generic. It writes bytes
to a configured payload root, computes their content digest, writes canonical
record JSON, and indexes records in the producer-owned local catalog. It does
not choose a project's Artifact ID, display name, path, schema, or profile.

`prepare_features` returns a plain mapping with `features`, `fold_definition`,
and `feature_contract` keys matching its three output ports. The feature table
is a single immutable CSV Artifact, so it does not pretend to be a
DatasetSnapshot. Each declaration specifies its display name, logical key,
payload path, representation, and any schema metadata beside the function that
produced it. The SDK writes and binds all three Artifacts; the runner does not
construct them.

The source DataFrame is similarly persisted by its Artifact-decorated
acquisition function. The runner only reads resulting references to connect
later stages:

```python
with LocalArtifactPublisher(...) as publisher:
    with observe_lifecycle(
        run_bike_lifecycle,
        publisher=publisher,
        run_id=run_id,
        source=source,
    ) as observed:
        training_plan = create_training_plan(run_id=run_id, fold_count=3)
        source_snapshot = download_source_csv()
        prepared = prepare_features(source_snapshot, training_plan)
        prepare_outputs = observed.outputs_for(prepared)
        feature_table = prepare_outputs["features"]       # CsvArtifact
        folds = prepare_outputs["fold_definition"]        # JsonArtifact
        # The remaining workflow calls run while this same lifecycle is active.
```

Each temporal-fold training call receives `feature_table` and `folds`, not
`prepared["features"]` or an in-memory fold dictionary. The runtime reloads the CSV
into the function's `pd.DataFrame` parameter and reloads the JSON fold document
into its `dict[str, object]` parameter, while the Execution records those two
exact Artifact references. Candidate evaluation receives
`many(CsvArtifact)` prediction handles and its ordinary function parameter is
a `tuple[pd.DataFrame, ...]`. It directly returns `evaluation` and
`training_config`; holdout scoring similarly receives the published CatBoost
model file and feature CSV through adapters, then returns `predictions` and
`metrics`. Final-model training likewise receives the feature CSV and its JSON
training configuration as typed handles. The runner then calls
`observed.publish_artifact_set(..., materialize_manifest=True,
manifest_name="Bike demand release manifest")` with the five exact output
handles. The SDK materializes a separate `release-manifest.json` sidecar from
those handles and their available upstream OCLP record closure. It carries the
exact digest-bound ArtifactSet reference and therefore is not a sixth member:
including it in the set would create a self-content cycle. This remains direct
collection publication, not a fake package Computation: it has no locator,
Execution, or lifecycle Events.

`training_plan` is an input Artifact rather than a fake lifecycle output. Its
decorator persists the configuration as JSON and its exact reference is bound
to `prepare_features`. This makes the fold-count choice a real, portable input
to the computation that uses it.

### 4. The runner bridges automatic observations to MLflow

The source snapshot is an external input Artifact; it is not an output of a
fabricated ingest Execution. Feature preparation is a decorated multi-output
Computation: the SDK creates its Execution, lifecycle Events, and output
bindings automatically. The runner forwards its exact record references to
MLflow and mirrors the resulting OCLP payload files into that nested MLflow
run for convenient experiment inspection.

```python
prepare_ref = observed.execution_for(prepared)
prepare_computation = observed.computation_for(prepared)
tracker.attach_execution(
    execution=prepare_ref,
    computation=prepare_computation,
    inputs={
        "source_snapshot": (source_snapshot.reference,),
        "training_plan": (training_plan.reference,),
    },
    outputs={port: (artifact.reference,) for port, artifact in prepare_outputs.items()},
    artifacts=prepare_outputs,
)
```

The root MLflow run mirrors acquired source/configuration Artifacts, each
computation child run mirrors the Artifacts it produced, and the release child
run mirrors every release member plus the materialized release-manifest sidecar.
Each mirror has an `artifact-manifest.json` with the OCLP ID, digest, media
type, and MLflow destination. OCLP remains the immutable source of truth;
MLflow deliberately holds convenient copies for its experiment UI.

The SDK observes temporal-fold training, candidate evaluation, final-model
training, and holdout scoring and materializes their declared outputs.
`@evidence` evaluators run automatically against a same-named returned output:
`temporal_validation_quality(evaluation)`, for example, evaluates the
`"evaluation"` entry. The runtime publishes Evidence before the terminal Event
and marks the Execution failed when a required evaluator fails.
Release publication is deliberately outside the automatic Computation path.
`observed.publish_artifact_set(...)` publishes a named, immutable collection
from exact handles but does not invent an Execution or lifecycle Events. A
retry that represents a new release should use a new lifecycle run ID, which
creates a distinct ArtifactSet ID, digest, and manifest Artifact. The runner
supplies the concise manifest name; the SDK does not derive it from the run.

After that collection exists, `run_demo` opens a separate observed lifecycle
for the release inference smoke test. It calls
`load_release_manifest(release_manifest_path)`, persists a deterministic
request Artifact, and invokes `predict_bike_demand(...)` using the resolved
`ArtifactSetHandle`. Its required `Prediction response validation`
Evidence ensures a finite prediction with a request and release identity.
Failure raises from the smoke lifecycle without changing the already immutable
training release.

### 5. Dataset, release, and quality concepts use the records that fit them

The feature table is a normal immutable CSV Artifact. A fold-training
Execution directly accepts that Artifact and the temporal-fold JSON document;
final training directly accepts the same feature table and its JSON
configuration.

The selected model release is an ArtifactSet, not another opaque model file.
It has named members for the final CatBoost model, feature contract, temporal
evaluation report, training configuration, and input feature table. The runner
publishes that ArtifactSet for release consumers directly from the five exact
digest-bound Artifact handles, and the SDK writes a `release-manifest.json`
sidecar. That sidecar carries the exact ArtifactSet reference, record body, and
resolved upstream provenance closure without copying model or dataset bytes.
Its set members remain individually addressable immutable Artifacts. The
offline holdout scorer dogfoods the lower-level inputs directly—the published
model file Artifact and the feature-table CSV—and publishes prediction and
metrics Artifacts through its declared output mapping.

The evaluation Computation directly requires the decorated
`temporal_validation_quality` evaluator. The SDK binds that evaluator to the
source observed for the Execution and records the same exact binding in its
Evidence. Its quality gate is therefore both Evidence and a success condition
for that Execution; a failed gate produces a terminal failed status rather than
a misleading successful completion. The detailed numeric report remains a
separate Artifact. The evaluator binding is inferred from its parameter name,
so the runner does not call `evaluate_evidence()`:

```python
@evidence(name="Temporal validation quality")
def temporal_validation_quality(evaluation) -> str:
    return "pass" if evaluation["rmse"] <= 250 else "fail"


@computation(
    ...,
    outputs={"evaluation": JsonArtifact(name="Candidate evaluation")},
    requires=(temporal_validation_quality,),
)
def evaluate_candidate(...) -> dict[str, object]:
    return {"evaluation": evaluation}
```

The detailed fold and holdout metrics remain JSON Artifact payloads, where
they can be inspected or used by later computations. Evidence makes the
specific pass/fail decision visible in a generic way. At the end of a run, the
demo also calls `validate_execution_acceptance()` alongside the derivation and
execution-hierarchy validators.

## Run it locally

The example has an isolated dependency environment and uses the adjacent SDK
checkout by editable path.

```bash
cd examples/bike-demand-service
uv sync
uv run bike-demand run --run-id bike-demand-first-run
```

All downloaded data, immutable payloads, OCLP records, the DuckDB catalog, and
MLflow's local data stay under the example's ignored `data/` directory. A run
prints the OCLP record directory, model-release ArtifactSet ID, release-smoke
Execution and response Artifact IDs, and MLflow tracking URI.

To inspect the durable records with Cyclops, point its API at the generated
directory:

```bash
oclp-explorer --oclp-dir "$(pwd)/data/oclp-0.2-evidence"
```

## OCLP and MLflow have separate responsibilities

MLflow is intentionally a parallel experiment-tracking view, not an OCLP
record store or Artifact registry.

| Concern | OCLP | MLflow in this demo |
| --- | --- | --- |
| Immutable model/data/prediction bytes | Canonical Artifacts at local file locations | Mirrored into the owning MLflow run for inspection |
| Exact inputs and outputs | Digest-bound Execution references | Linked through tags and a small bridge manifest |
| Quality gates | Evidence records | Metric comparison and inspection |
| Batch grouping | Shared lifecycle-profile `run_id` across real Executions | Parent and nested MLflow runs |
| Parameters and scalar metrics | Durable Execution/Evidence details where meaningful | Experiment-comparison UI |

Every MLflow run is tagged with its OCLP Execution and Computation IDs and
record digests. MLflow logs `oclp/record-links.json` plus a local
`artifact-manifest.json` beside each mirrored OCLP payload. The copies are for
MLflow inspection; their identity always comes from the OCLP reference and
digest.

Start the local MLflow UI with:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/data/mlflow/mlflow.db"
```

## OCLP records to look for

- The feature table is an immutable CSV Artifact used for training.
- Each fold child Execution accepts the prepared feature-table CSV and a
  temporal-fold JSON Artifact, then outputs a CatBoost model, validation
  predictions, and metrics.
- The candidate evaluation publishes a detailed metrics Artifact and quality
  gate Evidence. The Evidence records the check, not every prediction value.
- The final model, feature contract, evaluation report, training configuration,
  and feature table are named members of the release ArtifactSet.
- The holdout scorer consumes the final model and feature-table member
  Artifacts directly, then produces prediction and metrics Artifacts plus a
  response-contract Evidence record.

## Release-backed FastAPI inference

The example now includes a deliberately small local FastAPI service. It is
started with an SDK-created `release-manifest.json`, not an arbitrary model
path:

```bash
uv run bike-demand serve --release-manifest \
  data/runs/<run-id>/release/<release-key>/release-manifest.json
```

At application startup, `oclp.load_release_manifest()` verifies the manifest's
exact ArtifactSet reference and resolves its locally available members. Each
`POST /predict` request is persisted through `@json_artifact` as an external
request Artifact. The `predict_bike_demand` Computation accepts that request
Artifact and the entire `ArtifactSetHandle` as a `model_release` input. It
explicitly materializes the release's model and feature-contract members, so
the emitted Execution records a real ArtifactSet → Execution edge rather than
an untracked release ID parameter. The SDK persists the JSON prediction
response and emits the normal `execution-started`, `artifacts-published`, and
terminal `Event` records. `Prediction response validation` is the same required
Evidence gate used by the release inference smoke test.

This is intentionally correctness-first: every request reloads the verified
model Artifact and records both payloads under `data/inference/`. It proves the
release-to-serving contract without introducing production concerns such as
sampling, redaction, asynchronous publication, caching, OpenTelemetry, or a
metrics backend. Those are follow-on service concerns rather than requirements
for the OCLP contract.

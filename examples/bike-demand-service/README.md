# OCLP bike-demand service demo

This project is a deliberately staged, end-to-end OCLP example. It will train
a bike-demand model from an open dataset, package the selected model as a
release, score a holdout set, and serve release-pinned predictions through
FastAPI.

It lives beside the Python SDK so it can dogfood the SDK as a real consumer,
without making CatBoost, FastAPI, or data-science dependencies part of the
SDK's core installation.

The published SDK documentation includes a full [bike-demand reference-project
guide](https://evanz.github.io/oclp-python/bike-demand-example/)
that explains the computation boundaries and the OCLP records this project
publishes.

## Status

**Batch and local inference milestones implemented.** The `run` command
downloads the UCI source data, builds leakage-safe time-ordered folds, trains
CatBoost models, evaluates and packages a release, and scores an untouched
holdout. It then opens a separate **Release inference smoke test** run:
it resolves the just-published `release-manifest.json`, submits one fixed
request to the same decorated prediction callable used by FastAPI, and requires
response-validation Evidence to pass. The `serve` command requires that
SDK-created release manifest and
will load the exact release ArtifactSet, then verify and load the CatBoost
model and serving feature contract named by its members.
Each accepted `/predict` request becomes a local JSON Artifact, and its response
is the output Artifact of a request-scoped OCLP Execution.

Every reusable transformation declares an OCLP Computation beside its real
Python function with `@oclp.computation`. The UCI ingest boundary instead uses
`@oclp.csv_artifact`: its decorator-owned CSV policy persists the function
body's returned `pandas.DataFrame` as a source snapshot Artifact and returns a
resolved Artifact handle. The SDK adapts that handle back to pandas for the next
Computation according to its parameter annotation. It creates no child
Execution because data acquisition is not a derived computation. Feature
preparation, temporal-fold training, candidate evaluation, final-model
training, and holdout scoring all persist their declared outputs and pass typed
Artifact handles between them. Candidate evaluation uses a `many(CsvArtifact)`
input that the SDK reloads as a tuple of pandas DataFrames; holdout scoring
reloads the published CatBoost model Artifact through the SDK-owned
`CatBoostModelArtifact` adapter.
The runner retains only root orchestration and directly publishes the
model-release ArtifactSet from its exact input Artifact handles. With the SDK
release-manifest option it also materializes `release-manifest.json` as a
sidecar containing the exact ArtifactSet reference and available upstream OCLP
record closure without duplicating payload bytes. That is a
collection-publication operation, not a synthetic Computation or Execution.
The candidate evaluation and holdout scorer declare their quality evaluators with
`@oclp.evidence` and require them directly from `@oclp.computation`, so a
terminal successful status is only valid when every corresponding required
Evidence evaluator has passed.

The demo also contains a deliberately isolated source-format factory for SDK
dogfooding. It can persist that same source table through `@csv_artifact`,
`@parquet_artifact`, or `@json_artifact(serialization="pandas-table")` and pass every
result to a typed `pd.DataFrame` consumer through the matching verified pandas
adapter. The normal batch pipeline deliberately remains on CSV.

## Dataset

The planned dataset is the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset).
It contains hourly bike-rental counts with weather, seasonal, and calendar
inputs. The example will predict demand with CatBoost regression using
time-ordered folds. Target-derived fields must be excluded from features to
avoid leakage.

## Intended run topology

```text
source snapshot Artifact (CSV)
  -> feature-table + fold-definition Artifacts
  -> prepare, fold-training, evaluation, final-training, and holdout-scoring Executions
       (all share one UUID-based run-profile `run_id`)
  -> evaluation Evidence and metrics Artifact
  -> model-release ArtifactSet
  -> offline holdout inference
  -> separate release-inference smoke-test run
  -> FastAPI request-scoped inference
```

The example will use OCLP Core records as follows:

| Boundary | OCLP representation |
| --- | --- |
| Download UCI source table | Persisted CSV source snapshot of the fetched table. |
| Prepare features | Execution that produces feature-table and fold-definition Artifacts. |
| Train a candidate | Separate fold-training Executions joined with the batch by a shared UUID-based run-profile `run_id`. |
| Evaluate candidate | Predictions and detailed metrics as Artifacts; aggregate quality checks as Evidence. |
| Publish release | ArtifactSet containing the selected model, feature contract, configuration, validation report, and training data; an SDK-owned manifest sidecar identifies the exact set. |
| Release inference smoke test | A linked sibling run that resolves the release manifest into the exact ArtifactSet, records a deterministic request and response, and requires prediction-response Evidence to pass. |
| Score a request | Execution from the release ArtifactSet and a request Artifact to a response Artifact, with OCLP Events. |

The FastAPI demo retains all request/response Artifacts locally. A production
deployment would generally use asynchronous publication, sampling, redaction,
and an operational export rather than synchronously persisting every request.

## MLflow tracking

The batch milestone also uses MLflow as a parallel experiment-tracking view.
It is not the source of truth for OCLP records or Artifact identity.

- One MLflow run mirrors one OCLP model-training run.
- The SDK adapter logs canonical OCLP record JSON, UUID cross-links, typed
  Execution parameters, and numeric Evidence details.
- Model payloads are mirrored by default; other payloads are opt-in, avoiding
  accidental copies of large datasets.
- OCLP remains authoritative. By default an MLflow failure becomes a durable
  OCLP `adapter-failed` Event with a Diagnostic instead of losing OCLP records.

The initial demo will use local MLflow metadata and artifacts under
`data/mlflow/`. This makes the MLflow UI easy to start without a service, while
leaving the OCLP DuckDB store independently inspectable in Cyclops.

MLflow is useful here for experiment comparison. The FastAPI service does not
write to MLflow: it uses the local OCLP store directly so the release, request,
response, Execution, and Events can be inspected together. Operational export
such as OpenTelemetry remains a later concern.

## Layout

```text
examples/bike-demand-service/
  data/                       # local downloads and generated records; ignored
  src/bike_demand_service/
    data.py                   # UCI access and time-ordered feature preparation
    modeling.py               # training plan, CatBoost training, evaluation, scoring
    environment.py            # local OCLP, payload, and MLflow storage locations
    runner.py                 # declared run, bootstrap, and SDK MLflow adapter configuration
    cli.py                    # executable model-training command
    service.py                # release-backed FastAPI application factory
  tests/                      # preparation, tracking, and FastAPI contract tests
  pyproject.toml              # demo-only dependencies and local SDK binding
```

## Setup

The example has its own environment and depends on this checkout of `oclp` by
path. From this directory:

```bash
uv sync
uv run bike-demand run --materialization-id bike-demand-first-run
```

The first command creates an isolated demo environment. `run` performs the
complete model-training run, publishes its release, then performs one
release-backed inference smoke test in a separate run. The SDK assigns each
run a fresh UUID, so Cyclops shows the smoke test as a sibling run
linked to training through the released model Artifact rather than as a fake
training step. It writes only ignored local data beneath `data/`.

The command prints the resulting release manifest plus the smoke Execution and
response Artifact IDs. The smoke test exercises the same OCLP-decorated
prediction callable as FastAPI; it intentionally does not start an HTTP server.

After a run, inspect the resulting records in Cyclops:

```bash
oclp-explorer --oclp-dir "$(pwd)/data/oclp-0.3"
```

Start a request-scoped service with the `release-manifest.json` path printed by
the batch command:

```bash
uv run bike-demand serve --release-manifest \
  data/runs/bike-demand-first-run/release/<release-key>/release-manifest.json
```

The health endpoint reports the pinned release ID. `POST /predict` accepts the
twelve model features (`season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`,
`workingday`, `weathersit`, `temp`, `atemp`, `hum`, and `windspeed`). Its JSON
response identifies the exact request Artifact UUID, response Artifact UUID,
release ID, and OCLP Execution ID. Request and response payloads remain under
`data/inference/`; their canonical records appear in `data/oclp-0.3/`.

Or start MLflow's local UI against the independent SQLite tracking database:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/data/mlflow/mlflow.db"
```

MLflow receives canonical OCLP record JSON and a convenient copy of the
CatBoost model under `oclp/`. Non-model payloads are not copied unless the
application explicitly selects them. The canonical OCLP Artifact remains the
source of truth.

## Dagster UI dogfood

The optional Dagster definition runs the existing OCLP- and MLflow-instrumented
training workflow as one local Dagster asset. It does not duplicate the
workflow's OCLP declarations or turn the Dagster asset into a synthetic OCLP
Computation. After a materialization, Dagster displays the OCLP run ID, record
directory, and selected Dagster context in the asset metadata panel.

```bash
uv sync --group dev --extra dagster
uv run dagster dev -m bike_demand_service.dagster_defs --port 3000
```

Open <http://127.0.0.1:3000>, select **bike_demand_training**, and materialize
the **bike_demand_model_release** asset. The job uses Dagster's in-process
executor because the dogfood store has a local single-writer DuckDB catalog.
It writes to the same ignored `data/` area used by the ordinary demo: OCLP
records under `data/oclp-0.3`, immutable payloads under `data/runs`, and the
MLflow mirror under `data/mlflow`.

## Implementation sequence

1. Add request-volume sampling and redaction policy for the service boundary.
2. Add asynchronous OCLP publication and release-model caching without losing
   the exact release-to-Execution binding.
3. Add operational export, such as OpenTelemetry, after the local record
   contract proves useful.

No new OCLP Core model profile is required for the first implementation. Any
model-specific validation should begin as an example-owned, versioned profile
or Evidence evaluator and be generalized only after it proves reusable.

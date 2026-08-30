# OCLP bike-demand service demo

This project is a deliberately staged, end-to-end OCLP example. It will train
a bike-demand model from an open dataset, package the selected model as a
release, score a holdout set, and eventually serve predictions through FastAPI.

It lives beside the Python SDK so it can dogfood the SDK as a real consumer,
without making CatBoost, FastAPI, or data-science dependencies part of the
SDK's core installation.

The published SDK documentation includes a full [bike-demand reference-project
guide](https://evanz.github.io/oclp-python/bike-demand-example/)
that explains the computation boundaries and the OCLP records this project
publishes.

## Status

**Batch milestone implemented.** The `run` command downloads the UCI source
data, builds leakage-safe time-ordered folds, trains CatBoost models, evaluates
and packages a release, and scores an untouched holdout. The FastAPI inference
service remains a later milestone.

Every reusable computation declares an OCLP Definition beside its real Python
function with `@oclp.definition`. The runner keeps runtime observation explicit:
it materializes payload bytes as Artifacts, binds them into Invocations, and
publishes lifecycle Events and contract Evidence. The decorator is therefore
helpful metadata—not an opaque pipeline framework or automatic tracer.

## Dataset

The planned dataset is the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset).
It contains hourly bike-rental counts with weather, seasonal, and calendar
inputs. The example will predict demand with CatBoost regression using
time-ordered folds. Target-derived fields must be excluded from features to
avoid leakage.

## Intended lifecycle

```text
raw dataset Artifact
  -> feature DatasetSnapshot + fold-definition Artifact
  -> parent training-lifecycle Invocation
       -> child fold-training Invocations
       -> evaluation, final-training, packaging, and holdout-scoring Invocations
  -> evaluation Evidence and metrics Artifact
  -> model-release ArtifactSet
  -> offline holdout inference
  -> FastAPI request-scoped inference
```

The example will use OCLP Core records as follows:

| Boundary | OCLP representation |
| --- | --- |
| Download source CSV | Raw immutable Artifact with source metadata. |
| Prepare features | Invocation that produces a feature DatasetSnapshot and fold-definition Artifact. |
| Train a candidate | Parent Invocation with child fold-training Invocations. |
| Evaluate candidate | Predictions and detailed metrics as Artifacts; aggregate quality checks as Evidence. |
| Publish release | ArtifactSet containing the selected model, feature contract, configuration, metrics, and training manifest. |
| Score a request | Invocation from a model-release input to a response Artifact, with lifecycle Event and response-contract Evidence. |

The FastAPI phase will retain all request/response artifacts for this demo. A
production deployment would generally use asynchronous publication, sampling,
and redaction rather than synchronously persisting every request.

## MLflow tracking

The batch milestone also uses MLflow as a parallel experiment-tracking view.
It is not the source of truth for OCLP records or Artifact identity.

- One parent MLflow run mirrors the root model-lifecycle Invocation.
- Nested MLflow runs mirror the child Invocations, including every fold.
- MLflow records tunable parameters, per-fold and aggregate metrics, and
  human-oriented charts or reports.
- Every MLflow run is tagged with the corresponding OCLP Invocation and
  Definition identities and digests.
- MLflow will log a small OCLP reference manifest, rather than duplicate model
  or dataset bytes that are already published as canonical OCLP Artifacts.

The initial demo will use local MLflow metadata and artifacts under
`data/mlflow/`. This makes the MLflow UI easy to start without a service, while
leaving the OCLP DuckDB store independently inspectable in Cyclops.

MLflow is useful here for experiment comparison. FastAPI request monitoring is
a later concern and is better demonstrated with request-scoped OCLP records
and an operational exporter such as OpenTelemetry.

## Layout

```text
examples/bike-demand-service/
  data/                       # local downloads and generated records; ignored
  src/bike_demand_service/
    data.py                   # UCI access and time-ordered feature preparation
    modeling.py               # CatBoost training, evaluation, and holdout scoring
    runner.py                 # OCLP publication and nested MLflow instrumentation
    tracking.py               # local MLflow settings and OCLP run correlation
    cli.py                    # `status` and executable `run` commands
    service.py                # future FastAPI application factory
  tests/                      # future contract and integration tests
  pyproject.toml              # demo-only dependencies and local SDK binding
```

## Setup

The example has its own environment and depends on this checkout of `oclp` by
path. From this directory:

```bash
uv sync
uv run bike-demand status
uv run bike-demand run --run-id bike-demand-first-run
```

The first command creates an isolated demo environment. The `status` command
prints the computation boundaries; `run` performs the complete batch lifecycle
and writes only ignored local data beneath `data/`.

After a run, inspect the resulting records in Cyclops:

```bash
oclp-explorer --oclp-dir "$(pwd)/data/oclp"
```

Or start MLflow's local UI against the independent SQLite tracking database:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/data/mlflow/mlflow.db"
```

MLflow does not receive the CatBoost model, dataset, or prediction payload
bytes. It receives parameters, metrics, and `oclp/record-links.json`, which
links each MLflow run to exact OCLP record IDs and digests.

## Implementation sequence

1. Add a FastAPI prediction endpoint with asynchronous OCLP instrumentation.
2. Add request-volume sampling, redaction, and OpenTelemetry delivery for the
   service boundary.
3. Point Cyclops at the local store and document the complete graph.

No new OCLP Core model profile is required for the first implementation. Any
model-specific contract should begin as an example-owned, versioned profile or
Evidence contract and be generalized only after it proves reusable.

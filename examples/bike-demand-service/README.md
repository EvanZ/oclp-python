# OCLP bike-demand service demo

This project is a deliberately staged, end-to-end OCLP example. It will train
a bike-demand model from an open dataset, package the selected model as a
release, score a holdout set, and eventually serve predictions through FastAPI.

It lives beside the Python SDK so it can dogfood the SDK as a real consumer,
without making CatBoost, FastAPI, or data-science dependencies part of the
SDK's core installation.

## Status

**Scaffold only.** The project structure, dependencies, intended OCLP
boundaries, and command entry point are in place. It intentionally does not
download data, train a model, or start a service yet.

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
  -> parent training run
       -> child fold-training Invocations
       -> fold models and holdout predictions
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

## Layout

```text
examples/bike-demand-service/
  data/                       # local downloads and generated records; ignored
  src/bike_demand_service/
    pipeline.py               # declared computation boundaries
    cli.py                    # scaffolding command
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
```

The second command currently prints the planned stages. Later commands will
run the batch pipeline, score the holdout, start the FastAPI server, and open
the generated OCLP store in Cyclops.

## Implementation sequence

1. Ingest the dataset and publish the raw Artifact.
2. Build leakage-safe features, a DatasetSnapshot, and time-ordered fold
   definition.
3. Train CatBoost models in child fold Invocations and evaluate the candidate.
4. Publish a model-release ArtifactSet and score an offline holdout set.
5. Add a FastAPI prediction endpoint with asynchronous OCLP instrumentation.
6. Point Cyclops at the local store and document the complete graph.

No new OCLP Core model profile is required for the first implementation. Any
model-specific contract should begin as an example-owned, versioned profile or
Evidence contract and be generalized only after it proves reusable.

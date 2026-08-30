# End-to-end bike-demand example

The SDK repository includes a self-contained consumer project at
[`examples/bike-demand-service`](https://github.com/EvanZ/oclp-python/tree/main/examples/bike-demand-service).
It dogfoods OCLP on a public UCI Bike Sharing dataset without adding
data-science or MLflow dependencies to the `oclp` package itself.

## What it runs

The batch milestone executes a time-ordered CatBoost regression workflow:

```text
UCI source CSV
  -> feature table + DatasetSnapshot + temporal-fold definition
  -> three fold-training Invocations
  -> candidate evaluation and Evidence
  -> final model
  -> model-release ArtifactSet
  -> offline holdout predictions and Evidence
```

The root lifecycle Invocation is the parent of each child computation. The
data-derivation graph remains explicit: Artifacts and ArtifactSets flow into
Invocations, which produce new Artifacts or ArtifactSets. OCLP's graph and
Invocation-hierarchy validators run at the end of the demo.

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
prints the OCLP record directory, root Invocation ID, model-release ArtifactSet
ID, and MLflow tracking URI.

To inspect the durable records with Cyclops, point its API at the generated
directory:

```bash
oclp-explorer --oclp-dir "$(pwd)/data/oclp"
```

## OCLP and MLflow have separate responsibilities

MLflow is intentionally a parallel experiment-tracking view, not an OCLP
record store or Artifact registry.

| Concern | OCLP | MLflow in this demo |
| --- | --- | --- |
| Immutable model/data/prediction bytes | Canonical Artifacts at local file locations | Not copied |
| Exact inputs and outputs | Digest-bound Invocation references | Linked through tags and a small bridge manifest |
| Contracts | Evidence records | Metric comparison and inspection |
| Run hierarchy | Parent and child Invocations | Parent and nested MLflow runs |
| Parameters and scalar metrics | Durable Invocation/Evidence details where meaningful | Experiment-comparison UI |

Every MLflow run is tagged with its OCLP Invocation and Definition IDs and
record digests. MLflow logs `oclp/record-links.json`, a small reference
manifest. It deliberately does **not** log the CatBoost model, dataset, or
prediction files that OCLP already identifies as immutable Artifacts.

Start the local MLflow UI with:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/data/mlflow/mlflow.db"
```

## OCLP records to look for

- The feature DatasetSnapshot is an Artifact whose `dataset-snapshot` profile
  value describes its exact feature-table partition.
- Each fold child Invocation accepts that DatasetSnapshot and a temporal-fold
  Artifact, then outputs a CatBoost model, validation predictions, and metrics.
- The candidate evaluation publishes a detailed metrics Artifact and quality
  gate Evidence. The Evidence records the check, not every prediction value.
- The final model, feature contract, evaluation report, training configuration,
  and DatasetSnapshot are named members of the release ArtifactSet.
- The holdout scorer consumes the release ArtifactSet and DatasetSnapshot and
  produces prediction and metrics Artifacts plus a response-contract Evidence
  record.

FastAPI request-scoped inference, sampling, redaction, and operational export
are deliberately deferred to the next milestone.

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
| `data.py` | Downloads the UCI source and prepares leakage-safe temporal features. | Declares ingest and feature-preparation Definitions. |
| `modeling.py` | Trains CatBoost folds and final model, evaluates, and scores holdout data. | Declares reusable model-computation Definitions. |
| `runner.py` | Coordinates the real model work and explicitly publishes observations. | Binds Artifacts, Invocations, Evidence, and Events. |
| `publication.py` | Writes immutable payload bytes, hashes them, and persists canonical records. | A small local OCLP publisher, not a general service. |
| `lifecycle.py` | Publishes standard lifecycle Events around each observed Invocation. | Application-owned lifecycle adapter. |
| `tracking.py` | Sends experiment-oriented metadata to local MLflow. | Bridges MLflow runs to OCLP record references without copying payloads. |

All generated data is local and ignored by Git:

```text
data/
  runs/<run-id>/       # payload bytes: CSV, CatBoost models, JSON reports
  oclp/                # canonical OCLP record documents and producer catalog
  mlflow/              # local MLflow SQLite metadata and its own artifacts
```

## What it does

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

Each reusable computation is declared beside its actual Python callable with
`@oclp.definition`. At run time the demo adds its observed Git source via
`definition_record`, so Definition locators are derived from the functions
rather than copied as hand-maintained strings in the runner.

## How this project uses OCLP

### 1. A callable declares its static Definition contract

The decorator expresses only static metadata: a logical Definition ID, a
human-readable name, and its named input and output ports. The original
function remains a normal `pandas`/scikit-learn function; the decorator does
not intercept its arguments, return values, or errors.

```python
@definition(
    id="urn:oclp-bike-demand:definition:prepare-features",
    name="Prepare bike-demand features and folds",
    input_ports=(PortDefinition(name="raw_dataset"),),
    output_ports=(
        PortDefinition(name="features"),
        PortDefinition(name="dataset_snapshot", media_types=("application/json",)),
        PortDefinition(name="fold_definition", media_types=("application/json",)),
    ),
)
def prepare_features(source: pd.DataFrame, *, fold_count: int = 3) -> PreparedFeatures:
    # Ordinary domain logic: normalize, remove leakage, and create time folds.
    ...
```

This is the actual pattern used by
[`data.py`](https://github.com/EvanZ/oclp-python/blob/main/examples/bike-demand-service/src/bike_demand_service/data.py),
[`modeling.py`](https://github.com/EvanZ/oclp-python/blob/main/examples/bike-demand-service/src/bike_demand_service/modeling.py),
and `runner.py`. It avoids the drift-prone alternative of maintaining a
separate table of string locators.

### 2. The runner materializes source-bound Definition records

At the start of a run, the application gathers its decorated functions and
binds every template to the Git source observed for this checkout. The SDK
derives `implementation.locator` directly from each function, for example
`bike_demand_service.data.prepare_features`.

```python
functions = {
    "ingest": download_source_data,
    "prepare": prepare_features,
    "train_fold": train_fold,
    "evaluate": evaluate_folds,
    "train_final": train_final_model,
    "package": package_model_release,
    "score": score_holdout,
}
definitions_by_key = definitions(project_root, functions=functions)
definition_refs = {
    key: publisher.publish(record)
    for key, record in definitions_by_key.items()
}
```

`definitions()` is a thin application helper around `definition_record()`.
Its only policy is how this project identifies the source checkout; it does
not infer a pipeline or execute a decorated function.

### 3. Domain bytes become Artifacts; the runner binds them to an Invocation

After the actual function returns, `LocalPublisher` writes the exact bytes,
computes their content digest, and publishes the matching Artifact. The runner
then explicitly identifies the invocation inputs and outputs.

```python
prepared = prepare_features(source)
features = publisher.artifact_for_bytes(
    artifact_id=_artifact_id(run_id, "feature-table"),
    name=f"Leakage-safe bike-demand features — {run_id}",
    relative_path="prepared/features.csv",
    content=_csv_bytes(prepared.frame),
    media_type="text/csv",
    created_at=utc_now(),
)

prepare_ref = _publish_stage(
    publisher=publisher,
    run_id=run_id,
    stage="prepare",
    definition=definition_refs["prepare"],
    parent=root_ref,
    inputs={"raw_dataset": (source_artifact.reference,)},
    outputs={"features": (features.reference,)},
    # parameters, MLflow bridge, and lifecycle timestamps omitted here
)
```

`_publish_stage()` constructs the Invocation and records
`invocation-requested`, `attempt-started`, `artifacts-published`, and terminal
lifecycle Events. That helper is deliberately application code: different
projects will have different retry, failure, orchestration, and privacy
policies.

### 4. Dataset, release, and quality concepts use the records that fit them

The feature table is a normal file Artifact. Its DatasetSnapshot manifest is a
separate JSON Artifact carrying the `dataset-snapshot` profile and naming the
exact partition Artifact it describes. A fold-training Invocation accepts that
snapshot and its temporal-fold definition as inputs.

The selected model release is an ArtifactSet, not another opaque model file.
It has named members for the final CatBoost model, feature contract, temporal
evaluation report, training configuration, and input DatasetSnapshot. The
holdout scorer consumes this ArtifactSet as `model_release` and publishes
prediction and metrics Artifacts.

The quality gate is Evidence because it answers a contract question about the
evaluation Invocation, rather than serving as the full numeric report:

```python
Evidence(
    id=f"urn:oclp-bike-demand:evidence:temporal-quality:{run_id}",
    subject=invocation_reference,
    contract={
        "id": "urn:oclp-bike-demand:contract:temporal-validation-quality",
        "version": "1",
    },
    outcome=quality_gate,
    observed_at=utc_now(),
    details={
        "checks": [{
            "id": "temporal-validation-rmse-below-threshold",
            "expectation": "rmse-less-than-or-equal-to-250",
            "observed": evaluation["rmse"],
            "outcome": quality_gate,
        }],
    },
)
```

The detailed fold and holdout metrics remain JSON Artifact payloads, where
they can be inspected or used by later computations. Evidence makes the
specific pass/fail decision visible in a generic way.

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

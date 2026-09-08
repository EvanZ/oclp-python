# Observe a run

`@run` declares an application workflow that coordinates several real
Computations. It does not create a root Execution or synthetic flow edges.
`observe_run(...)` creates a fresh UUID and applies the same `profiles.run`
binding to every real Execution observed inside that context. Optional SDK
integration adapters may mirror those already-published records elsewhere.

```python
from oclp import MlflowAdapter, RunArtifactSet, observe_run, run


@run(
    name="Daily demand model training",
    artifact_sets=(
        RunArtifactSet(
            name="Demand model release",
            members={
                "model": (train_model.output("model"), "model"),
                "features": (prepare_features.output("features"), "training-data"),
            },
            materialize_manifest=True,
            manifest_name="Demand model release manifest",
        ),
    ),
    adapters=(MlflowAdapter(experiment_name="daily-training"),),
)
def train_demand_model(*, fold_count: int):
    source = acquire_source_snapshot()
    prepared = prepare_features(source, fold_count=fold_count)
    train_model(prepared["features"])
```

At the application bootstrap boundary, select the publisher and the exact
source basis. The SDK generates the UUID automatically:

```python
with observe_run(
    train_demand_model,
    publisher=publisher,
    source=source,
) as observed:
    train_demand_model(fold_count=3)

release = observed.artifact_set("Demand model release")
```

When the destination is bootstrap configuration rather than a static workflow
choice, pass `adapters=(...)` to `observe_run(...)` instead. Adapters are
optional integration mirrors; OCLP publication remains authoritative.

## Integration adapters

Adapters are optional SDK extensions that observe records after the runtime has
published them. They can mirror those records to an external system, but do not
define OCLP dataflow, create Core records, or become an authority for
provenance. Declare stable workflow integrations on `@run`:

```python
@run(
    name="Daily demand model training",
    adapters=(MlflowAdapter(experiment_name="daily-training"),),
)
def train_demand_model(...): ...
```

The declaration is a template. When `observe_run(...)` begins, the runtime
creates a fresh active adapter instance for that one observation (by calling
`for_run()` when the adapter provides it). This prevents connection and session
state from leaking across runs.

Inside the active context, retrieve one configured adapter by its *class*, not
by a variable captured from the declaration:

```python
with observe_run(train_demand_model, publisher=publisher, source=source) as observed:
    train_demand_model(...)

    mlflow = observed.adapter(MlflowAdapter)
    mlflow.log_metrics({"validation_rmse": 42.1})
```

`observed.adapter(MlflowAdapter)` returns the active per-run `MlflowAdapter`
instance—the runtime selects it with the equivalent of
`isinstance(adapter, MlflowAdapter)`. It requires exactly one matching
adapter: zero or multiple matches raise a clear error rather than silently
choosing one. Use this escape hatch only for application-selected integration
context. Automatic OCLP record and payload mirroring stays inside the adapter.

`RunArtifactSet` declarations are resolved only when the `observe_run(...)`
context completes successfully. Each member references a persisted output port
on a real `@computation` callable via `.output("port")`; the SDK resolves it
to the one exact Artifact emitted in that run. A missing member or a callable
that emitted the referenced port more than once fails clearly rather than
guessing. The direct collection publication creates no synthetic Computation,
Execution, or Event.

Use this for a run-local release assembled from child Computations. Keep
`observed.publish_artifact_set(...)` for genuinely dynamic collections whose
members cannot be declared before the workflow runs.

### Required Evidence policy

Required Evidence always determines the terminal status of its own Execution.
By default, a failed gate does **not** control the surrounding workflow: its
outputs, Evidence, and failed terminal Event are published, then the workflow
may deliberately inspect the result and continue along an independent branch.

When a run should stop before downstream code can consume a rejected result,
declare that policy once on the workflow:

```python
@run(
    name="Daily demand model training",
    required_evidence_policy="raise",
)
def train_demand_model(...): ...
```

After the SDK has materialized the outputs, evaluated every required evaluator,
and published the failed terminal Execution, it raises
`RequiredEvidenceFailedError`. The exception exposes the exact Execution
reference and the complete tuple of Evidence outcomes. This is SDK workflow
control flow, not a new Core record or a replacement for Evidence.

### MLflow metric outputs

`MlflowAdapter` can project numeric fields from explicitly selected JSON
Computation outputs. This avoids application calls to `log_metrics()` for
ordinary model-comparison outputs while keeping the selection in the MLflow
integration declaration:

```python
from oclp import MlflowAdapter, MlflowMetricOutput

@run(
    name="Daily demand model training",
    adapters=(
        MlflowAdapter(
            experiment_name="daily-training",
            metric_outputs=(
                MlflowMetricOutput(
                    output=evaluate_candidate.output("metrics"),
                    prefix="candidate",
                ),
                MlflowMetricOutput(
                    output=train_fold.output("metrics"),
                    prefix="fold",
                    dimensions=("fold_number",),
                ),
            ),
        ),
    ),
)
def train_demand_model(...): ...
```

Each declaration targets one exact decorated output, rather than guessing from
an output named `"metrics"`. The adapter reads the verified persisted JSON
Artifact and exports only top-level numeric scalar fields. `dimensions` add
declared scalar Execution parameters to the MLflow key—for example,
`fold.fold_number-2.rmse`. Repeated selected calls without distinct dimensions
are rejected rather than silently colliding. No output is selected by default,
and this remains an MLflow projection rather than a Core OCLP `Metric` type.

Within one active `OclpRun`, an exact raw value returned from a decorated
Computation can be supplied directly to another decorated Computation. The SDK
reuses the already-materialized Artifact binding by object identity and records
the correct input reference. Pass an `ArtifactHandle` instead only when the
consumer should reload verified persisted bytes through an adapter, or when
crossing a process/run boundary.

Every real Execution receives a binding like:

```json
{
  "profiles": {
    "run": {
      "version": "0.3.0-draft",
      "run_id": "2ba2c124-bcc8-4ac4-a3d4-b4fdd9aa8fb0",
      "run_name": "Daily demand model training"
    }
  }
}
```

The UUID identifies one concrete invocation. `run_name` is its concise display
label. The profile groups Executions for navigation only; actual dataflow
continues to be the explicit Artifact → Execution → Artifact graph.

`OclpRun` remains available when an application wants scoped automatic
observation without claiming a batch run—for example, a request-scoped
inference service that is represented through its service-level projection.

“Lifecycle” is intentionally not used for the per-invocation profile. A
future persistent lifecycle identifier may associate several UUID-identified
runs, but it must be explicitly supplied by an application rather than created
implicitly by the SDK.

## Dirty Git source

`source_from_git_checkout(...)` always preserves a usable Git basis when the
worktree is dirty by setting `GitSource.dirty` to `true`. When reproducibility
matters, capture the exact working-tree changes before opening the run:

```python
from oclp import (
    GitSource,
    capture_git_source_overlay,
    source_from_git_checkout,
)

source = source_from_git_checkout(project_root, path="src/demand_model")
if isinstance(source, GitSource) and source.dirty:
    source = capture_git_source_overlay(
        project_root,
        source=source,
        publisher=publisher,
        name="Demand-model training source overlay",
        relative_path="source-overlays/2026-09-05T120000Z",
    )
```

The helper publishes a binary `git diff HEAD` as an Artifact and binds its
ArtifactSet through `GitSource.overlay`. This makes the selected source basis
`commit + overlay`, rather than merely claiming that it was dirty.

Untracked files require explicit selection because the SDK must not silently
capture generated files or secrets. If any are present, pass every selected
path in `untracked_files=("src/local_rules.py",)`; otherwise the helper raises
instead of recording an incomplete overlay.

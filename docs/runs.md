# Observe a run

`@run` declares an application workflow that coordinates several real
Computations. It does not create a root Execution or synthetic flow edges.
`observe_run(...)` creates a fresh UUID and applies the same `profiles.run`
binding to every real Execution observed inside that context.

```python
from oclp import RunArtifactSet, observe_run, run


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

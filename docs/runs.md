# Observe a run

`@run` declares an application workflow that coordinates several real
Computations. It does not create a root Execution or synthetic flow edges.
`observe_run(...)` creates a fresh UUID and applies the same `profiles.run`
binding to every real Execution observed inside that context.

```python
from oclp import observe_run, run


@run(name="Daily demand model training")
def train_demand_model(*, observed, fold_count: int):
    source = acquire_source_snapshot()
    prepared = prepare_features(source, fold_count=fold_count)
    model = train_model(observed.outputs_for(prepared)["features"])
    return observed.publish_artifact_set(
        name="Demand model release",
        members={"model": (observed.outputs_for(model)["model"], "model")},
    )
```

At the application bootstrap boundary, select the publisher and the exact
source basis. The SDK generates the UUID automatically:

```python
with observe_run(
    train_demand_model,
    publisher=publisher,
    source=source,
) as observed:
    release = train_demand_model(observed=observed, fold_count=3)
```

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

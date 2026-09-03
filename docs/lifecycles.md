# Observe a lifecycle

`@lifecycle` is the Python SDK's declaration for an application workflow that
contains several real computations. It does not create a Core record, a root
Execution, or synthetic flow edges. Instead, the SDK derives one shared
`profiles.lifecycle` binding for the real Executions created inside that
workflow.

This keeps two facts separate:

- The application owns what it does and in what order.
- The SDK owns how decorated artifact acquisitions and computations become
  immutable OCLP Artifacts, Executions, Events, and Evidence.

## Declare the workflow

Put the declaration on the function that actually coordinates the domain
steps. Its body remains ordinary Python; it calls Artifact- and
Computation-decorated functions directly.

```python
from oclp import lifecycle


@lifecycle(
    namespace="urn:example",
    name="Daily demand model lifecycle",
)
def train_demand_model(*, observed, run_id: str, fold_count: int):
    source = acquire_source_snapshot()
    prepared = prepare_features(source, fold_count=fold_count)
    model = train_model(observed.outputs_for(prepared)["features"])
    return observed.publish_artifact_set(
        key="demand-model-release",
        name="Demand model release",
        members={"model": (observed.outputs_for(model)["model"], "model")},
        materialize_manifest=True,
        manifest_name="Demand model release manifest",
    )
```

The `observed` value above is only needed for the SDK's explicit inspection and
collection helpers such as `outputs_for(...)` and `publish_artifact_set(...)`.
It does not publish individual Execution or Event records: the active runtime
does that automatically when a decorated boundary runs.

## Activate the lifecycle once

At the application's bootstrap boundary, choose the local or remote store and
the source basis actually used for this run. `observe_lifecycle(...)` creates
the active `OclpRun` and derives the lifecycle profile from the decorated
workflow and concrete `run_id`.

```python
from pathlib import Path

from oclp import observe_lifecycle, source_from_git_checkout
from oclp.publishing import LocalArtifactPublisher


source = source_from_git_checkout(Path.cwd())
with LocalArtifactPublisher(
    catalog_path=Path("data/oclp/catalog.duckdb"),
    record_root=Path("data/oclp"),
    payload_root=Path("data/runs/demand-20260902"),
) as publisher:
    with observe_lifecycle(
        train_demand_model,
        publisher=publisher,
        run_id="demand-20260902",
        source=source,
    ) as observed:
        release = train_demand_model(
            observed=observed,
            run_id="demand-20260902",
            fold_count=3,
        )
```

`source_from_git_checkout()` does not reject a development checkout with local
changes. It returns a `GitSource` with `dirty: true`, retaining the current
commit as the reviewed base revision. That makes the observation useful and
honest, while signalling that the source cannot be reproduced from the commit
alone unless the application also publishes an explicit source overlay.

Every real Execution created by the active runtime receives:

```json
{
  "profiles": {
    "lifecycle": {
      "version": "0.2.0-draft",
      "run_id": "urn:example:lifecycle:demand-20260902",
      "run_name": "Daily demand model lifecycle"
    }
  }
}
```

The call does not invent a parent Execution for `train_demand_model`. The
derivation graph still contains only actual Artifact → Execution → Artifact
relationships. Explorers can use the profile as a navigation and grouping
boundary without mistaking orchestration for data flow.

## Link lifecycles through real Artifacts

One application can start a second lifecycle after the first publishes an
ArtifactSet or Artifact. The second lifecycle should consume the explicit
Artifact reference it needs; it does not become a child Execution of the first
lifecycle merely because it was launched afterwards. For example, a release
inference smoke test can resolve an ArtifactSet from a release manifest and
record its own request → Execution → response graph. An explorer can then show
the two lifecycle boundaries as siblings connected by that real released
ArtifactSet input.

## What remains application-owned

`observe_lifecycle(...)` deliberately does not guess where an application
wants to store records, what source revision it considers authoritative, or
which functions belong in its workflow. Those are deployment and domain
choices. The application supplies the publisher and selected source once; the
SDK handles the repeated observation mechanics inside the workflow.

Integrations such as MLflow are intentionally separate. They can consume or
mirror the OCLP records produced by this lifecycle, but they are not required
for lifecycle observation and are not a dependency of the core SDK.

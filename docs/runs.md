# Runs

`@run` declares an application workflow that coordinates real Computations.
It is not a Core record, a parent Execution, or a replacement for explicit
Artifact → Execution → Artifact lineage. `observe_run(...)` activates the
runtime for one concrete workflow invocation.

## Declare and observe a workflow

```python
from oclp import observe_run, run


@run(name="Daily demand model training")
def train_demand_model(*, fold_count: int) -> None:
    source = acquire_source_snapshot()
    features = prepare_features(source, fold_count=fold_count)
    train_model(features)


with observe_run(
    train_demand_model,
    publisher=publisher,
    source=source,
) as observed:
    train_demand_model(fold_count=3)
```

The SDK generates a fresh run UUID and applies the same `profiles.run` binding
to every real Execution observed inside the context. The run UUID groups work
for navigation; it does not create dataflow edges or a synthetic parent node.
Supply `run_id=` only when the application already owns that concrete UUID.

The decorated workflow is still ordinary application code. It can call the
same Computation several times, branch on domain logic, and choose how to
handle an accepted or rejected result. Every decorated call records its own
Execution.

## Retrieve run results

`OclpRun` exposes exact references created during the active observation:

| API | Result |
| --- | --- |
| `outputs_for(call_result)` | Materialized output `ArtifactHandle` values. |
| `execution_for(call_result)` | The exact Execution reference for one call. |
| `computation_for(call_result)` | The source-bound Computation reference. |
| `evidence_for(call_result)` | Evidence emitted for one call. |
| `artifact_set(name)` | A completed declared ArtifactSet. |

See [Artifacts](artifacts.md) for output handles and [Artifact sets](artifact-sets.md)
for run-local release assembly.

## Configure integrations

Adapters are optional extensions that observe records after the runtime has
published them. Declare stable workflow integrations with `adapters=(...)` on
`@run`, or pass them to `observe_run(...)` when they are bootstrap
configuration. An adapter declaration is a template; the runtime creates a
fresh active instance for each observed run.

See [MLflow](mlflow.md) for experiment tracking and [Dagster](dagster.md) for
native scheduler assets. OCLP remains authoritative: adapters neither define
OCLP dataflow nor create Core records.

## Stop a workflow after failed required Evidence

Required Evidence determines the terminal status of its own Execution. By
default, a failed gate publishes its outputs, Evidence, and failed terminal
Event, then leaves application control flow unchanged.

Set `required_evidence_policy="raise"` on the run when downstream work must
not consume a rejected result:

```python
@run(
    name="Daily demand model training",
    required_evidence_policy="raise",
)
def train_demand_model() -> None: ...
```

After publishing all outputs, Evidence, and the failed terminal Event, the
runtime raises `RequiredEvidenceFailedError`. Its `execution` and `evidence`
attributes identify the exact records. See [Evidence](evidence.md) for
declaring the gates.

## Group separately scheduled work with a lifecycle

`new_lifecycle()` creates an application-selected UUID that can group
Artifacts, ArtifactSets, and Executions across independently scheduled runs.
It is opt-in and creates no dataflow or scheduler dependency:

```python
from oclp import lifecycle_from_profiles, new_lifecycle, observe_run


lifecycle = new_lifecycle()
with observe_run(train, publisher=publisher, source=source, lifecycle=lifecycle):
    train()

# A later service can recover the release's durable lifecycle identity.
lifecycle = lifecycle_from_profiles(release.artifact_set.profiles)
with observe_run(serve, publisher=publisher, source=service_source, lifecycle=lifecycle):
    serve()
```

The portable `profiles.lifecycle` binding is useful for a release and its later
inference observations, but those records still require explicit Artifact and
ArtifactSet relationships.

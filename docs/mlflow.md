# MLflow integration

Install the optional integration first:

```bash
pip install 'oclp[mlflow]'
```

OCLP remains the authoritative provenance store. The MLflow adapter mirrors
already-published OCLP records and selected payloads for experiment tracking;
it never reads MLflow to create, validate, or query OCLP provenance.

## Configure an MLflow mirror for a run

Attach `MlflowAdapter` to the workflow's `@run` declaration:

```python
from oclp import MlflowAdapter, observe_run, run


@run(
    name="Daily training",
    adapters=(
        MlflowAdapter(
            experiment_name="daily-training",
            tracking_uri="sqlite:///mlflow.db",
        ),
    ),
)
def workflow(): ...


with observe_run(workflow, publisher=publisher, source=source):
    workflow()
```

The declaration is a reusable template. The runtime calls `for_run()` when
available, creating a fresh active mirror for each observation. When neither
`tracking_uri` nor `artifact_location` is set, the local publisher defaults to
a SQLite MLflow store and artifact directory beside the OCLP record directory.

The adapter mirrors canonical OCLP record JSON, UUID tags, typed Execution
parameters, and top-level numeric Evidence details. Model payloads mirror by
default. Each mirrored payload is stored beneath its OCLP Artifact UUID, so
repeated same-named outputs, such as temporal-fold models, remain distinct in
the MLflow artifact store.

Execution parameter keys include their OCLP Execution UUID because MLflow
parameters are immutable within one MLflow run while a Computation can run
repeatedly with different arguments.

## Select Computation metrics and additional payloads

Use `@mlflow` outside `@computation` to keep MLflow presentation policy beside
the output it projects:

```python
from oclp import JsonArtifact, MlflowMetrics, computation, mlflow


@mlflow(
    metrics=(
        MlflowMetrics(
            output_port="metrics",
            prefix="fold",
            dimensions=("fold_number",),
        ),
    ),
)
@computation(
    name="Train fold",
    outputs={"metrics": JsonArtifact(name="Fold metrics")},
)
def train_fold(*, fold_number: int) -> dict[str, object]:
    return {"metrics": {"rmse": 42.1}}
```

The selected output must be an `application/json` Artifact. The adapter reads
its verified bytes and logs only top-level numeric scalar fields. `dimensions`
must name declared scalar Execution parameters and prevent repeated
invocations from producing the same MLflow key. A collision is an adapter
failure: it follows the normal Diagnostic behavior unless `strict=True`.

`@mlflow(payloads=("report",))` similarly mirrors a named non-model output
payload. Payload names are exact output ports, not Artifact display names.
Model payloads and canonical OCLP record JSON always mirror. The declaration
is inert when the active run has no `MlflowAdapter`, so the same Computation
works in environments without MLflow. Execution parameters already mirror
automatically, so the decorator is needed only for metric projections and
additional payloads.

`@computation` must be closest to the function. `@mlflow` must be outside it
so it can validate declared output ports. When a Computation also has a
run-local `@artifact_set` declaration, that decorator is also outside
`@computation`; their relative order does not matter.

## Select workflow parameters

Use `@mlflow` outside `@run` when a small, explicit set of workflow arguments
should also appear in MLflow. The mapping is **MLflow parameter name to
workflow argument name**:

```python
from oclp import MlflowAdapter, mlflow, run


@mlflow(
    run_parameters={
        "release_id": "release_id",
        "temporal_fold_count": "fold_count",
    },
)
@run(
    name="Daily training",
    adapters=(MlflowAdapter(experiment_name="daily-training"),),
)
def train(*, release_id: str, fold_count: int) -> None: ...
```

The adapter logs these as `workflow.release_id` and
`workflow.temporal_fold_count`, with matching
`oclp.mlflow.workflow_parameter.<name>=true` tags. This keeps them separate
from automatically mirrored, Execution-UUID-scoped parameters and makes their
provenance clear: they are application-chosen workflow context, not Core
Execution parameters.

Every selected argument must be JSON-compatible. The decorator rejects unknown
argument names at module load. A conflicting value for the same `workflow.*`
key within an MLflow run becomes an integration Diagnostic by default and
raises with `MlflowAdapter(strict=True)`. The declaration is inert without an
MLflow adapter and never creates an OCLP record, Execution, or derivation edge.

`run_parameters` belongs only on `@mlflow` outside `@run`; `metrics` and
`payloads` belong only on `@mlflow` outside `@computation`.

## Add application-selected comparison values

After the adapter is declared, retrieve its active session from the observed
run and add values whose MLflow presentation is an application decision:

```python
mlflow = observed.adapter(MlflowAdapter)
mlflow.log_metrics({"validation_rmse": 42.1})
mlflow.log_parameters({"candidate_family": "CatBoostRegressor"})
```

The adapter already mirrors OCLP Execution parameters and Evidence records, so
use this only for additional application-specific comparison context.

## Failure policy and model registration

By default, an MLflow failure does not undo OCLP publication. The runtime emits
an `adapter-failed` Event with an integration Diagnostic on the next real
Execution. Set `strict=True` to make a mirror failure fail the application
workflow.

For explicit registry publication, nominate the exact Artifact by its
application-owned name:

```python
from oclp import MlflowAdapter, MlflowModelRegistration


adapter = MlflowAdapter(
    experiment_name="daily-training",
    model_registration=MlflowModelRegistration(
        artifact_name="Validated model",
        registered_model_name="demand-model",
    ),
)
```

Registration is opt-in; the adapter does not infer promotion policy.

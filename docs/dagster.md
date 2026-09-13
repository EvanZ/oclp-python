# Dagster integration

Install the optional integration with:

```bash
pip install "oclp[dagster]"
```

OCLP and Dagster declare different things. OCLP decorators declare durable
application semantics: Artifact representations, Computation contracts, and
ArtifactSets. Native Dagster assets declare orchestration: asset keys,
dependencies, partitions, retries, I/O, and schedules.

The adapter connects those two layers at runtime. It does not infer a
Computation from an arbitrary `@dagster.asset`, create synthetic OCLP records,
or hide data dependencies behind a monolithic workflow asset.

## The mixed-decorator contract

A native Dagster asset and a canonical OCLP declaration can describe one
application operation, but they do not describe the same facts. Keep both
declarations explicit and keep their shared boundary in lockstep.

### Decorator order

The native Dagster decorator is always outermost. `@computation` is closest to
the function; optional OCLP extensions such as `@mlflow` and `@artifact_set`
sit outside it:

```python
@dg.multi_asset(...)  # native Dagster orchestration: outermost
@artifact_set(...)    # optional OCLP output-role declaration
@mlflow(...)          # optional OCLP projection declaration
@computation(
    ...,
    adapters=(dagster_adapter(),),
)
def train_orders(...) -> dict[str, object]:
    ...
```

Python applies decorators from the bottom up. Dagster therefore invokes the
OCLP-decorated callable, and the inner adapter can open an OCLP observation
using the real Dagster execution context. Reversing the order gives an OCLP
declaration a Dagster asset definition rather than an ordinary function.

For an acquisition boundary, use the same outer/inner order with an Artifact
decorator instead of `@computation`:

```python
@dg.asset(...)
@csv_artifact(..., adapters=(dagster_adapter(),))
def load_orders(...) -> pd.DataFrame:
    ...
```

That creates an OCLP Artifact but no OCLP Computation or Execution, so it has
no multi-output contract.

### Which layer owns which facts

| Concern | Owner |
| --- | --- |
| Asset keys, dependencies, partitions, groups, retry policy, scheduler configuration, and I/O-manager selection | Native `@dg.asset` or `@dg.multi_asset` |
| Computation identity, OCLP Artifact representation and metadata, portable input/output ports, Evidence, ArtifactSet roles, and MLflow projections | Canonical OCLP decorators |
| Opening the observation, publishing OCLP records, passing scheduler facts, and returning OCLP handles to Dagster | `dagster_adapter()` |

The adapter must appear in the canonical declaration's `adapters=` tuple, and
the outer asset must declare `required_resource_keys={"oclp"}`. The resource
provides the publisher, implementation source, and any project-level OCLP
adapters or lifecycle configuration.

### Input contract

For a Computation asset, these three names identify the same local input port:

```python
@dg.asset(ins={"orders": dg.AssetIn(key=dg.AssetKey("raw_orders"))})
@computation(inputs={"orders": JsonArtifact}, adapters=(dagster_adapter(),))
def prepare_orders(orders: dict[str, object]) -> dict[str, object]:
    ...
```

- `"orders"` in `ins` is the Dagster input name.
- `"orders"` in `inputs` is the OCLP port name.
- `orders` is the Python function parameter.

The upstream Dagster `AssetKey` may use a different name, as `raw_orders` does
above. When Artifacts cross Dagster workers or runs, use the OCLP Artifact I/O
manager: Dagster passes the exact `ArtifactHandle`, and OCLP verifies and loads
the normal domain value for the function body. The function does not receive a
shadow copy invented for OCLP.

`context: dg.AssetExecutionContext` is an optional implementation parameter.
Use it only when the body needs scheduler facts such as a partition key. The
adapter injects it; it is neither an OCLP input Artifact nor an OCLP Execution
parameter.

### Multi-output contract

Dagster outputs and OCLP output ports are independent sets with an optional
named overlap. A shared name is bridged automatically; each unshared output
remains owned by its declaring layer:

```python
@dg.multi_asset(
    outs={
        "validated": dg.AssetOut(key="validated_orders", io_manager_key="oclp_io"),
        "preview": dg.AssetOut(key="validation_preview"),
    },
    ins={"orders": dg.AssetIn(key=dg.AssetKey("raw_orders"))},
    required_resource_keys={"oclp"},
)
@computation(
    name="Validate orders",
    inputs={"orders": JsonArtifact},
    outputs={
        "validated": JsonArtifact(name="Validated orders"),
        "metrics": JsonArtifact(name="Order metrics"),
    },
    adapters=(dagster_adapter(),),
)
def validate_orders(orders: dict[str, object]) -> dict[str, object]:
    # One named result covers the union of native and OCLP outputs.
    return {
        "validated": orders,
        "preview": {"kind": "validation-preview"},
        "metrics": {"rows": len(orders)},
    }
```

`validated` is shared: OCLP materializes its Artifact, and the adapter returns
its exact `ArtifactHandle` to Dagster. `preview` is Dagster-only: the adapter
passes its raw value to Dagster, which uses the output's selected native I/O
manager. `metrics` is OCLP-only: OCLP materializes it and records it on the
Execution, but it is not a Dagster asset.

For several outputs, the application function must return a named mapping or
an object with same-named fields, such as a dataclass or `NamedTuple`. Do not
return a positional tuple or list. A single-output function may return its raw
value directly. This preserves each Artifact's identity by name even when
Dagster and OCLP expose different subsets:

```python
from typing import NamedTuple


class ValidationOutputs(NamedTuple):
    validated: dict[str, object]  # shared
    preview: dict[str, object]  # Dagster-only
    metrics: dict[str, int]  # OCLP-only
```

Only after OCLP has materialized its declared outputs does the adapter build
the native return value. It returns an `ArtifactHandle` for each shared Dagster
output and the raw value for each Dagster-only output, in the native Dagster
`outs` order. OCLP-only Artifacts are deliberately omitted. The OCLP Artifact
I/O manager persists UUID pointers, not duplicate payload data, for shared
outputs that cross worker or Dagster-run boundaries.

When the local names differ, add an explicit adapter binding from native output
name to OCLP port; same-named outputs need no configuration:

```python
adapters=(
    dagster_adapter(
        output_bindings={"native_validated": "validated"},
    ),
)
```

The adapter validates explicit bindings against the native Dagster outputs and
declared OCLP ports when the asset executes.

### Independent use and combined use

`@dg.multi_asset` does not import-time depend on `@computation`: without the
OCLP declaration it behaves as a normal Dagster multi-asset returning ordinary
Python values. Likewise, `@computation` can run outside Dagster inside a normal
OCLP `@run` / `observe_run(...)` observation. Adding `dagster_adapter()` makes
the combined invocation intentionally depend on the contract above: one body
produces real OCLP Artifacts and the exact handles flow through Dagster.

## Why the adapter is required

`@computation` and Artifact decorators are reusable declarations. Outside an
active `observe_run(...)`, they call their ordinary Python body; stacking only
`@dg.asset` and `@computation` therefore does not publish OCLP records.

Pass `dagster_adapter()` through the canonical decorator's `adapters=` tuple.
When Dagster invokes the asset, it opens the OCLP observation, publishes the
real records, returns exact output handles to Dagster, and writes navigation
metadata to the materialization. The adapter is runtime activation, not an
Artifact, Computation, or Dagster asset decorator itself.

## Before declaring assets: configure the integration once

Register project-level OCLP facts once in Dagster `Definitions`:

```python
from oclp import MlflowAdapter
from oclp.dagster import oclp_dagster_resource

resources = {
    "oclp": oclp_dagster_resource(
        publisher=publisher_for_context,
        source=source_for_context,
        adapters=(
            MlflowAdapter(
                experiment_name="daily-training",
                parent_profile="my_app.mlflow-parent",  # optional
            ),
        ),
        lifecycle=lifecycle_for_context,  # optional
        profiles=profiles_for_context,  # optional
    )
}
```

`publisher` and `source` are required because they select where records are
published and how their implementation is identified. `adapters` holds normal
OCLP runtime integrations, such as an optional `MlflowAdapter`; `lifecycle`
and `profiles` are optional application facts. Configure defaults on the
resource when they apply to the project, or override either application fact on
a specific declaration with `dagster_adapter(lifecycle=..., profiles=...)`.

## Optional runtime mirrors: MLflow

`@mlflow` on a Computation only declares which metrics or payloads to project.
It is inert until the Dagster resource receives an `MlflowAdapter` through its
`adapters=` tuple. The resource creates a fresh adapter session for every OCLP
observation opened by a native asset, so the adapter remains safe across
separate Dagster workers and retries.

For a release cycle that spans jobs, create the persistent MLflow parent before
the cycle begins, place its ID in an application profile, and set
`parent_profile` on the resource-level `MlflowAdapter`. Every native asset
observation then becomes an MLflow child run with the standard
`mlflow.parentRunId` tag. The bike-demand example uses this arrangement: its
release-cycle Artifact persists the idempotently created parent ID, every later
asset reads that Artifact into the profile, and the terminal release sensor
marks the parent finished.

The adapter derives an OCLP run label from the canonical Artifact, Computation,
or ArtifactSet declaration. It derives a stable OCLP run UUID from the Dagster
run plus the specific asset attempt. The raw scheduler run ID, asset key, step key,
partition key, and retry number remain in the generic `dagster` profile.

For a partition whose display label needs application-specific detail, pass a
context-aware `run_name` to the inner adapter. For example, the bike-demand
fold declaration names its Runs **Bike demand temporal validation fold 1**,
**2**, and so on, while each child Execution remains **Train bike demand fold**:

```python
@computation(
    name="Train bike demand fold",
    adapters=(
        dagster_adapter(
            run_name=lambda context: (
                f"Bike demand temporal validation fold {fold_number(context)}"
            )
        ),
    ),
)
def train_fold(...):
    ...
```

## The bike-demand Dagster implementation

The [bike-demand tutorial](bike-demand-example.md) deliberately describes the
ordinary OCLP runner. Its Dagster implementation is separate:

```text
data.py / modeling.py / runner.py     ordinary OCLP runner; no Dagster import
dagster/                              native assets, partitions, sensors, jobs, and resources
dagster_defs.py                       thin Dagster code-location entry point
```

The `dagster` package does not call an OCLP-decorated function from the ordinary
runner. It declares its own OCLP contracts with the SDK's canonical decorators,
then applies `dagster_adapter()` at that native asset boundary. This keeps
Dagster out of the ordinary runner and avoids hidden nested OCLP observations.
The implementations may share ordinary domain helpers, but never application-
defined OCLP decorator factories.

For example, the Dagster source asset uses the same CSV Artifact contract as
the ordinary acquisition while directly calling the undecorated fetch operation:

```python
from oclp import (
    CsvArtifact,
    JsonArtifact,
    MlflowMetrics,
    artifact_set,
    computation,
    csv_artifact,
    mlflow,
)

from bike_demand_service.data import UCI_BIKE_SHARING_DATASET_ID, _fetch_source_frame


@dg.asset(
    key="bike_demand_raw_source",
    io_manager_key="oclp_artifact_io_manager",
    required_resource_keys={"oclp"},
)
@csv_artifact(
    name="UCI Bike Sharing source (CSV)",
    description_from_docstring=True,
    index=False,
    lineterminator="\n",
    adapters=(dagster_adapter(),),
)
def bike_demand_raw_source() -> pd.DataFrame:
    """Acquire the source snapshot for one training cycle."""

    return _fetch_source_frame(UCI_BIKE_SHARING_DATASET_ID)
```

The multi-output feature asset likewise declares its Computation, MLflow, and
ArtifactSet contracts directly. Native Dagster fields stay on
`@dg.multi_asset`; OCLP facts stay on SDK decorators:

```python
@dg.multi_asset(
    outs={
        "features": dg.AssetOut(key="bike_demand_features", io_manager_key="oclp_io"),
        "fold_definition": dg.AssetOut(
            key="bike_demand_fold_definition", io_manager_key="oclp_io"
        ),
        "feature_contract": dg.AssetOut(
            key="bike_demand_feature_contract", io_manager_key="oclp_io"
        ),
        "data_metrics": dg.AssetOut(
            key="bike_demand_data_metrics", io_manager_key="oclp_io"
        ),
    },
    ins={
        "source_snapshot": dg.AssetIn(key=dg.AssetKey("bike_demand_raw_source")),
        "training_plan": dg.AssetIn(key=dg.AssetKey("bike_demand_training_plan")),
    },
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Bike demand CatBoost release",
    members={
        "features": ("features", "training-data"),
        "feature-contract": ("feature_contract", "serving-contract"),
    },
)
@mlflow(metrics=(MlflowMetrics(output_port="data_metrics", prefix="data"),))
@computation(
    name="Prepare bike demand features",
    inputs={"source_snapshot": CsvArtifact, "training_plan": JsonArtifact},
    outputs={
        "features": CsvArtifact(name="Bike demand features"),
        "fold_definition": JsonArtifact(name="Temporal fold definition"),
        "feature_contract": JsonArtifact(name="Feature contract"),
        "data_metrics": JsonArtifact(name="Training data metrics"),
    },
    adapters=(dagster_adapter(),),
)
def bike_demand_prepare_features(
    source_snapshot: pd.DataFrame,
    training_plan: dict[str, object],
) -> dict[str, object]:
    return _prepare_features_value(source_snapshot, training_plan)
```

The first example’s `@dg.asset`, and the second example’s `@dg.multi_asset`,
are intentionally outermost. The OCLP declaration closest to the function
defines the real Artifact or Computation; the adapter makes that declaration
active only when Dagster calls this separate implementation. The full working
graph, including the exact concrete Artifact declarations, dynamic partitions,
sensors, cross-run ArtifactSet assembly, and release lifecycle, is in
[`examples/bike-demand-service/src/bike_demand_service/dagster/definitions.py`](https://github.com/EvanZ/oclp-python/blob/main/examples/bike-demand-service/src/bike_demand_service/dagster/definitions.py).

## How to configure a Computation asset

The canonical OCLP decorator is the inner layer; the native Dagster decorator
is outermost. Python applies decorators from the bottom up, so this order is
required:

```python
import dagster as dg

from oclp import JsonArtifact, computation
from oclp.dagster import dagster_adapter


@dg.asset(
    key="prepared_orders",
    ins={"orders": dg.AssetIn(key=dg.AssetKey("raw_orders"))},
    required_resource_keys={"oclp"},
)
@computation(
    name="Prepare orders",
    inputs={"orders": JsonArtifact},
    outputs={"prepared": JsonArtifact(name="Prepared orders")},
    adapters=(dagster_adapter(),),
)
def prepare_orders(orders: dict[str, object]) -> dict[str, object]:
    return {"prepared_rows": len(orders)}
```

The native asset owns `key`, `ins`, dependencies, partitions, retry policy,
and Dagster configuration. The canonical Computation owns OCLP input/output
ports, Artifact formats, Evidence, MLflow declarations, and other semantic
OCLP metadata. Do not repeat `workflow`, `run_name`, `publisher`, `source`, or
an OCLP context parameter on every asset.

When an OCLP decorator such as `@mlflow` or `@artifact_set` is also present,
keep `@computation` closest to the function and retain `@dg.asset` outermost.
Reversing the stack gives an OCLP declaration a Dagster asset definition rather
than an ordinary function.

## How to configure an Artifact asset

Use an OCLP Artifact decorator when the asset produces a real OCLP Artifact but
not an OCLP Computation or Execution:

```python
import dagster as dg

from oclp import json_artifact
from oclp.dagster import dagster_adapter


@dg.asset(
    key="raw_orders",
    io_manager_key="oclp_io",
    required_resource_keys={"oclp"},
)
@json_artifact(name="Raw orders", adapters=(dagster_adapter(),))
def load_orders() -> dict[str, object]:
    return {"rows": 42}
```

Choose the concrete representation that matches the value, such as
`@csv_artifact`, `@parquet_artifact`, or `@json_artifact`. The outer asset
materializes the resulting `ArtifactHandle`; it does not change the OCLP
semantic record type.

## How to configure Computation parameters from Dagster

Native asset config remains an outer Dagster concern. If a configured field has
the same name as an optional declared Computation parameter, the adapter passes
that value through and the resulting Execution records it as that parameter:

```python


@dg.asset(
    key="evaluate_orders",
    config_schema={"quality_threshold": dg.Field(float, default_value=0.9)},
    required_resource_keys={"oclp"},
)
@computation(
    name="Evaluate orders",
    outputs={"report": JsonArtifact(name="Order quality report")},
    adapters=(dagster_adapter(),),
)
def evaluate_orders(*, quality_threshold: float = 0.9) -> dict[str, float]:
    return {"quality_threshold": quality_threshold}
```

This preserves the selected value as an OCLP Execution parameter without
making the Dagster context part of the application function's contract.

## How to configure a multi-output Computation asset

This example makes every output shared. Native Dagster owns the `AssetOut`
mapping and OCLP owns the Artifact declarations; see the
[multi-output contract](#multi-output-contract) when either layer needs an
additional private output:

```python
@dg.multi_asset(
    outs={
        "validated": dg.AssetOut(key="validated_orders", io_manager_key="oclp_io"),
        "metrics": dg.AssetOut(key="order_metrics", io_manager_key="oclp_io"),
    },
    ins={"orders": dg.AssetIn(key=dg.AssetKey("raw_orders"))},
    required_resource_keys={"oclp"},
)
@computation(
    name="Validate orders",
    inputs={"orders": JsonArtifact},
    outputs={
        "validated": JsonArtifact(name="Validated orders"),
        "metrics": JsonArtifact(name="Order metrics"),
    },
    adapters=(dagster_adapter(),),
)
def validate_orders(orders: dict[str, object]) -> dict[str, object]:
    return {"validated": orders, "metrics": {"rows": len(orders)}}
```

The adapter returns the exact handles in native `outs` order. A native
`multi_asset` remains atomic and non-subsettable, matching the one OCLP
Execution that produces all shared output Artifacts.

## How to pass Artifacts between Dagster assets

Use the catalog-backed I/O manager whenever an Artifact can cross a worker or
Dagster-run boundary:

```python
from pathlib import Path

from oclp.dagster import oclp_artifact_io_manager


resources = {
    "oclp_io": oclp_artifact_io_manager(
        catalog_path=Path("data/oclp/catalog.duckdb"),
        storage_root=Path("data/dagster-artifact-handles"),
    )
}
```

Set `io_manager_key="oclp_io"` on a single-output `@dg.asset`, and on every
`dg.AssetOut` of a `@dg.multi_asset`. The manager persists a small Artifact UUID
pointer per asset partition, then a later worker reloads the exact handle from
the OCLP catalog and verifies payload reads normally. Those pointers are
reconstructible scheduler state; immutable OCLP records and payloads remain
authoritative.

## How to use partitions and dynamic fan-out

Partitions belong on the native Dagster asset, just like any other Dagster
asset option:

```python
cycles = dg.DynamicPartitionsDefinition(name="training_cycle")


@dg.asset(
    key="prepared_orders",
    partitions_def=cycles,
    io_manager_key="oclp_io",
    required_resource_keys={"oclp"},
)
@computation(
    name="Prepare orders",
    outputs={"prepared": JsonArtifact(name="Prepared orders")},
    adapters=(dagster_adapter(),),
)
def prepare_orders() -> dict[str, object]:
    return {"rows": 42}
```

Dynamic partitions often execute in separate processes and scheduler runs, so
do not pass in-memory values between them. Pass Artifact handles through the
catalog I/O manager. Application code may accept an optional `context` argument
when it genuinely needs a partition value; the adapter injects the native
Dagster execution context without making it an OCLP parameter.

## How to declare and assemble ArtifactSets

Put `@artifact_set` outside each canonical producer Computation to declare the
release role of that output. A final native asset accepts the exact producer
handles and uses `@assemble_artifact_set` to publish the complete cross-run
set:

```python
from oclp import ArtifactHandle, artifact_set, assemble_artifact_set


@dg.asset(
    key="prepared_orders",
    io_manager_key="oclp_io",
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Customer-orders release",
    output_port="features",
    role="training-data",
)
@computation(
    name="Prepare orders",
    outputs={"features": JsonArtifact(name="Prepared orders")},
    adapters=(dagster_adapter(),),
)
def prepare_orders() -> dict[str, object]:
    return {"rows": 42}


@dg.asset(
    key="candidate_model",
    ins={"features": dg.AssetIn(key=dg.AssetKey("prepared_orders"))},
    io_manager_key="oclp_io",
    required_resource_keys={"oclp"},
)
@artifact_set(
    name="Customer-orders release",
    output_port="model",
    role="model",
)
@computation(
    name="Train order model",
    inputs={"features": JsonArtifact},
    outputs={"model": JsonArtifact(name="Order model")},
    adapters=(dagster_adapter(),),
)
def train_order_model(features: dict[str, object]) -> dict[str, object]:
    return {"algorithm": "example"}


@dg.asset(
    key="model_release",
    ins={
        "features": dg.AssetIn(key=dg.AssetKey("prepared_orders")),
        "model": dg.AssetIn(key=dg.AssetKey("candidate_model")),
    },
    required_resource_keys={"oclp"},
)
@assemble_artifact_set(
    name="Customer-orders release",
    members={
        "features": ("features", "training-data"),
        "model": ("model", "model"),
    },
    adapters=(dagster_adapter(),),
)
def model_release(features: ArtifactHandle, model: ArtifactHandle) -> None:
    pass
```

Dagster assets do not share an active `OclpRun`: preparation, dynamic folds,
and aggregation can execute in distinct processes and scheduler runs. The
adapter therefore defers the producer `@artifact_set` declarations; publishing
either partial contribution as a completed release would be incorrect. The
terminal asset reloads the durable handles and `@assemble_artifact_set`
publishes the one complete ArtifactSet.

The decorators share release intent but have different jobs:

- `@artifact_set` is reusable producer metadata for outputs within one shared
  OCLP run.
- `@assemble_artifact_set` is the explicit cross-run publication boundary.

The latter creates a real OCLP `ArtifactSet`, not a synthetic Artifact,
Computation, or Execution. Its outer Dagster asset is a terminal orchestration
node, not a data value for an I/O manager.

## How to group separate Dagster runs with a lifecycle

`lifecycle` is an opt-in portable UUID that groups records across separate
scheduler runs. Configure it on the shared resource when it applies broadly,
or override one declaration:

```python
from oclp import JsonArtifact, computation, lifecycle_from_id


@dg.asset(
    key="candidate_model",
    required_resource_keys={"oclp"},
)
@computation(
    name="Train candidate",
    outputs={"model": JsonArtifact(name="Candidate model")},
    adapters=(
        dagster_adapter(
            lifecycle=lambda context: lifecycle_from_id(context.partition_key)
        ),
    ),
)
def train_candidate() -> dict[str, object]:
    return {"algorithm": "example"}
```

Matching lifecycle IDs are a grouping assertion only. They do not introduce
Dagster dependencies or OCLP dataflow. Use the same lifecycle later for
release-backed inference when that work belongs to the same application cycle.

## How to attach application profiles

Use `profiles=` on `oclp_dagster_resource(...)` or `dagster_adapter(...)` for
application-owned facts such as an MLflow parent run ID. The SDK reserves the
`run` and `dagster` profile names:

```python
@dg.asset(key="candidate_model", required_resource_keys={"oclp"})
@computation(
    name="Train candidate",
    outputs={"model": JsonArtifact(name="Candidate model")},
    adapters=(
        dagster_adapter(
            profiles=lambda context: {
                "my_application.mlflow_parent": {
                    "version": "1",
                    "mlflow_parent_run_id": parent_run_id_for(context.partition_key),
                }
            }
        ),
    ),
)
def train_candidate() -> dict[str, object]:
    return {"algorithm": "example"}
```

# Artifact formats and library integrations

## Dagster workflow projection

Install the optional Dagster integration with:

```bash
pip install "oclp[dagster]"
```

`@dg_workflow(...)` projects an existing `@run` workflow as one Dagster asset.
The body still supplies application-specific workflow arguments; its existing
`@computation` and Artifact declarations continue to produce the only OCLP
records. On completion, the projection adds the OCLP run identity, record-store
path, outcome, and an explicit selection of Dagster context fields to the
Dagster asset materialization metadata.

```python
import dagster as dg

from oclp.dagster import dg_workflow


@dg_workflow(
    workflow=train,
    publisher=publisher_for_dagster_context,
    source=source_for_dagster_context,
    asset_key="trained_model",
)
def trained_model(context: dg.AssetExecutionContext) -> None:
    train(...)
```

This is an integration boundary, not an orchestration DSL. It never derives an
OCLP Computation from an asset, creates a parent Execution for a Dagster run,
or reflects arbitrary Dagster context. The initial allow-list is `run_id`,
`job_name`, `asset_key`, `partition_key`, and `retry_number`; select only
values that are useful to navigate the Dagster materialization. `dagster_asset` remains a
lower-level compatibility adapter for applications that need to compose a
custom Dagster decorator stack.

## Dagster graph projections

`@dg_artifact(...)`, `@dg_computation(...)`, and `@dg_artifact_set(...)` build
a granular Dagster graph from application-selected OCLP boundaries. They are
applied outside existing OCLP decorators: the inner declaration remains
canonical and the function body does not change.

```python
import dagster as dg

from oclp import JsonArtifact, computation, json_artifact, run
from oclp.dagster import dg_artifact, dg_computation


@run(name="Customer orders")
def customer_orders() -> None:
    pass


@dg_artifact(
    workflow=customer_orders,
    publisher=publisher_for_context,
    source=source_for_context,
    asset_key="raw_orders",
)
@json_artifact(name="Raw orders")
def load_orders() -> dict[str, int]:
    return {"rows": 42}


@dg_computation(
    workflow=customer_orders,
    publisher=publisher_for_context,
    source=source_for_context,
    asset_key="validated_orders",
    inputs={"orders": dg.AssetIn(key=dg.AssetKey("raw_orders"))},
)
@computation(
    name="Validate orders",
    inputs={"orders": JsonArtifact},
    outputs={"validated": JsonArtifact(name="Validated orders")},
)
def validate_orders(orders: dict[str, int]) -> dict[str, int]:
    return orders
```

`dg_artifact` records the acquired Artifact and exposes it as a Dagster data
asset; it does not create an OCLP Execution. `dg_computation` requires every
declared Artifact input to have an explicit Dagster `AssetIn`, and returns the
exact `ArtifactHandle`, so the downstream OCLP Execution records the upstream
Artifact reference in its ordinary `inputs` field.

For one output, use `asset_key`. For multiple outputs, map every declared OCLP
output port to `dg.AssetOut` through `outputs`; this creates one atomic,
non-subsettable Dagster `multi_asset`, matching the one OCLP Execution that
produces all outputs. A `many(...)` OCLP input maps either to an explicitly
ordered tuple of `AssetIn` values or to one partition-mapped `AssetIn` whose
I/O manager returns the selected handles as an ordered tuple.
`dg_artifact_set` is a visible downstream collection node for a release that
spans independently materialized steps, rather than letting worker-local
contexts publish partial collections. An `@artifact_set` already present on a
projected Computation is deferred for the same reason; use one explicit
`dg_artifact_set` for the cross-step collection.

Every granular projection uses the Dagster job name as the OCLP run title and
appends the partition key when present. All assets in one Dagster job therefore
retain one shared OCLP run UUID and title, while independently launched jobs
and fold partitions have distinct, legible Explorer nodes (for example,
`bike_demand_train_fold_job [cycle-a|fold-2]`). The workflow's `@run(name=...)`
remains the default title outside a Dagster granular projection.

When the original decorated callable lives in another module, a decorated
proxy can keep the Dagster definition declarative without duplicating the
OCLP declaration. Set `target` to the existing callable and have the proxy
call it with the exact Artifact handles it receives:

```python
@dg_computation(
    workflow=train,
    publisher=publisher_for_context,
    source=source_for_context,
    target=prepare_features,
    outputs={
        "features": dg.AssetOut(key="features"),
        "contract": dg.AssetOut(key="feature_contract"),
    },
    inputs={
        "source": dg.AssetIn(key=dg.AssetKey("raw_orders")),
        "plan": dg.AssetIn(key=dg.AssetKey("training_plan")),
    },
)
def customer_feature_assets(source, plan):
    return prepare_features(source, plan)
```

Every projected step independently opens an OCLP observation with a UUID
derived from the Dagster run ID. Executions from all steps and workers in that
Dagster run therefore share `profiles.run`; their `profiles.dagster` binding
retains generic Dagster run, job, asset, step, partition, and retry facts. A retry
creates a new immutable OCLP Execution with the same run identity and a new
retry number. A selected Dagster subset emits records only for steps Dagster
executes.

### Application-owned grouping profiles

`application_profiles` lets the granular projection decorators attach durable,
application-owned facts without teaching the SDK a domain-specific field. It
accepts either a profile mapping or a `context -> profile mapping` factory.
The mapping is attached to the projected Execution alongside the SDK-owned
`run` and `dagster` profiles, and to Artifacts, explicit ArtifactSets, and
release manifests published by that step.

```python
@dg_computation(
    workflow=train,
    publisher=publisher_for_context,
    source=source_for_context,
    asset_key="candidate_model",
    inputs={"features": dg.AssetIn(key=dg.AssetKey("features"))},
    application_profiles=lambda context: {
        "my_application": {
            "version": "1",
            "release_cycle_id": context.partition_key,
        }
    },
)
@computation(...)
def train_candidate(features):
    ...
```

The SDK reserves `run` and `dagster`; application profiles cannot overwrite
them. Use a project-owned profile for business concepts such as a release
cycle, tenant, or customer batch. `deps=` is also available on the granular
decorators when an ordering-only Dagster dependency is needed without adding a
synthetic OCLP Artifact input.

### Partitioned, multi-run graphs

Pass a Dagster `partitions_def` to the projection decorator exactly as you
would to `@dagster.asset`. A proxy can request the Dagster execution context
with `context_parameter="context"` when application code needs to translate a
partition key into an explicit parameter. This keeps the OCLP declaration
canonical while making the orchestration mapping visible in the proxy.

Dynamic partitions execute in separate Dagster runs, so an in-memory
`ArtifactHandle` cannot cross their boundary. Use the SDK's catalog-backed I/O
resource for Artifact-producing assets and outputs:

```python
from pathlib import Path

from oclp.dagster import oclp_artifact_io_manager

resources = {
    "oclp_artifact_io": oclp_artifact_io_manager(
        catalog_path=Path("data/oclp/catalog.duckdb"),
        storage_root=Path("data/dagster-artifact-handles"),
    ),
}
```

Set `io_manager_key="oclp_artifact_io"` on a single-output `dg_artifact` or
`dg_computation`; for a `multi_asset`, set that key on each `dg.AssetOut`.
The resource persists only an OCLP Artifact UUID per asset partition. A later
worker rehydrates the handle from the OCLP catalog and verifies payload reads
normally. The pointer directory is reconstructible scheduler state; immutable
OCLP records and payload locations remain authoritative.

The built-in `DuckdbCatalog` serializes each local catalog operation with an
inter-process lock, including connection setup. Independent Dagster workers on
the same shared filesystem can therefore publish while their long-running
Computations execute concurrently. This is a local coordination mechanism, not
a distributed catalog: workers on different hosts need shared locking and a
catalog service appropriate to that deployment. In every case, use a payload
root that incorporates the Dagster run, step, partition, and retry identity.

An OCLP `Artifact` is format-neutral. The Python SDK's concrete
`ArtifactType` declarations provide the local persistence and loading policy
for an Artifact payload. This page is the canonical inventory of integrations
shipped by the SDK—not a claim that OCLP itself requires any of these
libraries or formats.

The same concrete type is used at both sides of a Computation boundary:

```python
from xgboost import XGBRegressor

from oclp import XGBoostModelArtifact, computation


@computation(
    inputs={"model": XGBoostModelArtifact},
    outputs={"candidate": XGBoostModelArtifact(name="Candidate model")},
)
def promote(model: XGBRegressor) -> XGBRegressor:
    return model
```

The output declaration tells the SDK how to materialize the normal Python
return value. The input declaration validates the Artifact's durable media type;
the parameter annotation selects the adapter that loads verified bytes.

## Supported representations

| Artifact type | Durable format / media type | Optional SDK extra | Function return | Downstream parameter | Status |
| --- | --- | --- | --- | --- | --- |
| `BytesArtifact` | caller-declared media type | — | `bytes` | an application adapter | Supported |
| `FileArtifact` | caller-declared media type | — | `pathlib.Path` | an application adapter | Supported |
| `JsonArtifact` | JSON / `application/json` | — | JSON-compatible value | `dict[...]` or an application adapter | Supported |
| `JsonArtifact(serialization="pandas-table")` | pandas table JSON / `application/json` | pandas in application | `pandas.DataFrame` | `pandas.DataFrame` | Supported |
| `JsonLinesArtifact` | JSON Lines / `application/x-ndjson` | — | `pandas.DataFrame` or iterable of mappings | `pandas.DataFrame`, `list[dict[...]]`, or `tuple[dict[...], ...]` | Supported |
| `CsvArtifact` | CSV / `text/csv` | pandas in application | `pandas.DataFrame` | `pandas.DataFrame` | Supported |
| `ParquetArtifact` | Parquet / `application/vnd.apache.parquet` | `oclp[parquet]` | `pandas.DataFrame` | `pandas.DataFrame` | Supported |
| `ArrowIpcArtifact` | Arrow IPC file / `application/vnd.apache.arrow.file` | `oclp[arrow]` | `pyarrow.Table` or `pandas.DataFrame` | `pyarrow.Table` or `pandas.DataFrame` | Supported |
| `NpyArtifact` | NumPy `.npy` / `application/x-npy` | `oclp[numpy]` | `numpy.ndarray` | `numpy.ndarray` | Supported |
| `NpzArtifact` | NumPy `.npz` / `application/x-npz` | `oclp[numpy]` | mapping of named arrays | `dict[str, numpy.ndarray]` | Supported |
| `YamlArtifact` | YAML / `application/yaml` | `oclp[yaml]` | mapping | `dict[...]` | Supported |
| `TomlArtifact` | TOML / `application/toml` | `oclp[toml]` | mapping | `dict[...]` | Supported |
| `XmlArtifact` | UTF-8 XML / `application/xml` | `oclp[xml]` | XML `str` | `str` or `xml.etree.ElementTree.Element` | Supported |
| `CatBoostModelArtifact` | native CatBoost `.cbm` / `application/x-catboost-model` | `oclp[catboost]` | fitted CatBoost model | `CatBoostRegressor`, `CatBoostClassifier`, or `CatBoostRanker` | Supported |
| `XGBoostModelArtifact` | native XGBoost UBJSON `.ubj` / `application/x-xgboost-ubjson` | `oclp[xgboost]` | fitted XGBoost sklearn model or `Booster` | `XGBRegressor`, `XGBClassifier`, `XGBRanker`, or `Booster` | Supported |
| `LightGBMModelArtifact` | native LightGBM model `.txt` / `application/x-lightgbm-model` | `oclp[lightgbm]` | fitted `lightgbm.Booster` | `lightgbm.Booster` | Supported |
| `SklearnModelArtifact` | skops `.skops` / `application/x-skops` | `oclp[sklearn]` | fitted scikit-learn `BaseEstimator` | the exact annotated sklearn estimator type | Supported |

Media types beginning with `application/x-` are SDK conventions where a format
does not have a registered IANA media type. They make the representation
unambiguous within OCLP records without claiming an external standard.

## Representation configuration belongs beside the output port

The concrete Artifact declaration is the single place to configure how a
returned value becomes durable bytes. The SDK validates known options instead
of accepting opaque serializer `**kwargs` that might vary between libraries or
silently create an invalid representation.

```python
from oclp import JsonLinesArtifact, YamlArtifact, computation


@computation(
    name="Publish predictions and configuration",
    outputs={
        "predictions": JsonLinesArtifact(
            name="Prediction batch",
            sort_keys=True,
            newline="\n",
        ),
        "config": YamlArtifact(
            name="Training configuration",
            indent=2,
            sort_keys=True,
        ),
    },
)
def publish() -> dict[str, object]:
    return {
        "predictions": [{"id": "a", "score": 0.9}],
        "config": {"learning_rate": 0.1},
    }
```

Changing a declared representation option changes the stored bytes and hence
the Artifact digest. The source-bound Computation identifies the code that
made that serialization decision.

`JsonLinesArtifact` is intentionally separate from `JsonArtifact`: records
with one JSON object per line are a different durable contract, not merely a
`lines=True` rendering option. `ArrowIpcArtifact` writes the Arrow IPC file
format (also known as Feather V2), with Arrow's registered file media type.
See [Arrow's format documentation](https://arrow.apache.org/docs/format/Columnar.html).

## XML documents

`XmlArtifact` accepts an XML `str`, persists it as UTF-8 `.xml` bytes, and
validates it with `defusedxml`. DTDs and entity processing are rejected: XML
must be a safe, well-formed document. An explicit XML encoding declaration
must say UTF-8; otherwise it must be omitted. A downstream callable can ask
for the original `str` or a safely parsed `xml.etree.ElementTree.Element`.

The SDK deliberately preserves the returned XML text rather than silently
canonicalizing it. Equivalent XML documents can differ physically in attribute
order, whitespace, and other permitted syntax; XML canonicalization is an
application-level choice, while an OCLP Artifact digest identifies the exact
persisted bytes. See [Canonical XML](https://www.w3.org/TR/xml-c14n/) and
[Python's XML security guidance](https://docs.python.org/3/library/xml.html).

## Native model formats

`CatBoostModelArtifact` materializes the fitted returned object through
CatBoost's native `save_model()` API as `.cbm` bytes. The adapter verifies the
Artifact digest before loading the payload into the downstream annotated model
class.

`XGBoostModelArtifact` explicitly writes `.ubj`, XGBoost's UBJSON model
format, rather than accepting a version-dependent default. It accepts an
XGBoost `Booster` or sklearn-wrapper model with `save_model()` and reloads it
as the exact annotated XGBoost class. XGBoost also supports JSON, but the SDK
uses UBJSON as its shipped native representation to keep artifacts compact
without losing model semantics. See [XGBoost model I/O](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

`LightGBMModelArtifact` persists the native model returned as a
`lightgbm.Booster`. The LightGBM sklearn wrappers expose a fitted Booster as
`booster_`, but do not provide a stable native wrapper-loader boundary, so the
SDK does not reconstruct wrapper internals. Use the `Booster` for the native
artifact contract, or `SklearnModelArtifact` when the portable Python wrapper
object itself is required. [LightGBM documents `booster_` as the underlying
fitted Booster](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).

`SklearnModelArtifact` uses the `skops` `.skops` format rather than `pickle`
or `joblib`. It works with fitted `scikit-learn` `BaseEstimator` values and
returns the same annotated estimator type to the downstream function. The
default adapter refuses unknown types in a payload. A project that intentionally
uses reviewed custom types must configure an additional
`SklearnModelAdapter(trusted_types=(...))` in its `OclpRun` adapter registry;
the producer cannot grant that trust through the Artifact itself. This is a
Python-environment representation, not a cross-language serving format. See
[skops secure persistence](https://skops.readthedocs.io/en/stable/persistence.html).

## Compatibility and safety

The Artifact digest proves that a consumer reads the exact persisted bytes. It
does **not** make a model automatically compatible with every runtime. A
model's source framework and version remain part of its execution provenance;
applications should pin the environment that loads it.

The optional model libraries retain their own system-runtime requirements. For
example, XGBoost and LightGBM installations on some platforms also require an
OpenMP runtime; that is a framework requirement, not an OCLP payload concern.

The SDK intentionally does not currently provide a generic pickle/joblib
Artifact type. Those formats only belong in trusted, version-compatible Python
environments; `skops` and ONNX have different safety and portability tradeoffs.
See [scikit-learn's model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

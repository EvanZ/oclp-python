# Artifacts

OCLP Core is format-neutral. The Python SDK's concrete `ArtifactType`
declarations specify how a normal Python return value becomes durable bytes and
how a downstream typed parameter reloads those verified bytes. They work in an
ordinary `observe_run(...)` context and do not require a scheduler integration.

## How to declare an Artifact

Use a concrete acquisition decorator when a function obtains or creates one
durable value without being an OCLP Computation:

```python
from oclp import json_artifact


@json_artifact(name="Customer orders")
def load_orders() -> dict[str, object]:
    return {"rows": 42}
```

Use the same concrete type on a Computation output and input to declare its
durable representation at both sides of a data boundary:

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

The output declaration selects persistence. The input declaration validates the
durable media type, while the Python annotation selects the adapter that loads
verified bytes.

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

## How to configure a representation

The concrete declaration is the single place to configure how a returned value
becomes durable bytes. The SDK validates known options instead of accepting
opaque serializer `**kwargs` that might vary between libraries or silently
create an invalid representation.

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
the Artifact digest. `JsonLinesArtifact` is deliberately separate from
`JsonArtifact`: records with one JSON object per line are a different durable
contract, not merely a rendering option. `ArrowIpcArtifact` writes Arrow IPC
(Feather V2) with Arrow's registered file media type. See [Arrow's format
documentation](https://arrow.apache.org/docs/format/Columnar.html).

## XML documents

`XmlArtifact` accepts XML text, persists UTF-8 `.xml` bytes, and validates it
with `defusedxml`. DTDs and entity processing are rejected. An explicit XML
encoding declaration must say UTF-8; otherwise it must be omitted. A consumer
can request the original `str` or a safely parsed
`xml.etree.ElementTree.Element`.

The SDK preserves returned XML text rather than silently canonicalizing it.
Equivalent documents can differ physically in attribute order or whitespace;
canonicalization is an application choice, while the Artifact digest identifies
the exact persisted bytes. See [Canonical XML](https://www.w3.org/TR/xml-c14n/)
and [Python's XML security guidance](https://docs.python.org/3/library/xml.html).

## Native model formats

`CatBoostModelArtifact` persists a fitted model through CatBoost's native
`save_model()` API as `.cbm` bytes. `XGBoostModelArtifact` explicitly writes
UBJSON (`.ubj`) and reloads the exact annotated XGBoost class. `LightGBMModelArtifact`
persists a `lightgbm.Booster`; use `SklearnModelArtifact` when the sklearn
wrapper itself is the intended Python contract.

`SklearnModelArtifact` uses `skops` rather than `pickle` or `joblib`. Its
default adapter refuses unknown payload types. A project that intentionally
uses reviewed custom types must configure
`SklearnModelAdapter(trusted_types=(...))` in its `OclpRun` adapter registry;
the producer cannot grant that trust through the Artifact itself. This is a
Python-environment representation, not a cross-language serving format. See
[skops secure persistence](https://skops.readthedocs.io/en/stable/persistence.html).

## Compatibility and safety

The Artifact digest proves that a consumer reads the exact persisted bytes; it
does not make a model compatible with every runtime. Pin the environment that
loads a framework-native model.

The SDK intentionally does not provide a generic pickle/joblib Artifact type.
Those formats belong only in trusted, version-compatible Python environments;
`skops` and ONNX have different safety and portability tradeoffs. See
[scikit-learn's model persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

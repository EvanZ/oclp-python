# Artifact formats and library integrations

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
    id="urn:example:computation:publish",
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

"""Callable-bound Artifact acquisition declarations.

Each concrete decorator owns the contract for one persisted representation.
``@csv_artifact`` is deliberately self-contained: it validates the returned
table, serializes it with the configured CSV options, and publishes the
Artifact through the active SDK runtime.
"""

from __future__ import annotations

import json
import re
import tomllib
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from hashlib import sha256
from inspect import Parameter, signature
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    ParamSpec,
    cast,
    get_args,
    get_origin,
)

from pydantic import Field, field_validator

from oclp.models import JsonValue, OclpModel, ProfileBindings

if TYPE_CHECKING:
    from oclp.publishing import LocalArtifactPublisher, PublishedArtifact

Parameters = ParamSpec("Parameters")
ArtifactIdResolver = Callable[..., str]
_ARTIFACT_TYPE_ATTRIBUTE = "__oclp_artifact_type__"


class ArtifactAdapterError(ValueError):
    """Raised when the SDK cannot adapt a durable Artifact to a Python value."""


class ArtifactIntegrityError(ArtifactAdapterError):
    """Raised when locally read Artifact bytes do not match their digest."""


@dataclass(frozen=True)
class ArtifactHandle:
    """A local handle for one published immutable Artifact.

    This is SDK convenience, not an OCLP core record. It carries an exact
    Artifact reference and the local store location required by the first
    adapter implementation.
    """

    published: PublishedArtifact
    media_types: ClassVar[tuple[str, ...]] = ()

    @property
    def artifact(self):
        """Return the underlying language-agnostic OCLP Artifact record."""

        return self.published.artifact

    @property
    def path(self):
        """Return the locally materialized payload path for this handle."""

        return self.published.path

    @property
    def reference(self):
        """Return the digest-bound OCLP reference for this Artifact."""

        return self.published.reference

    def read_verified_bytes(self) -> bytes:
        """Read local payload bytes only after verifying their OCLP digest."""

        content = self.path.read_bytes()
        actual = sha256(content).hexdigest()
        expected = self.artifact.digest.value
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Artifact {self.artifact.id!r} expected sha256:{expected}, "
                f"found sha256:{actual} at {self.path}"
            )
        return content


class ArtifactAdapter(ABC):
    """Translate one verified Artifact payload to a Python runtime type."""

    @abstractmethod
    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        """Return whether this adapter can materialize ``artifact`` as the type."""

    @abstractmethod
    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        """Verify, deserialize, and return the target runtime value."""


@dataclass(frozen=True)
class ArtifactAdapterRegistry:
    """Ordered adapter registry used by an :class:`oclp.runtime.OclpRun`."""

    adapters: tuple[ArtifactAdapter, ...]

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        """Load one Artifact into the callable's annotated parameter type."""

        matches = tuple(
            adapter
            for adapter in self.adapters
            if adapter.supports(artifact, target_type)
        )
        if not matches:
            raise ArtifactAdapterError(
                f"no Artifact adapter can materialize {type(artifact).__name__} "
                f"as {_type_name(target_type)}"
            )
        if len(matches) > 1:
            raise ArtifactAdapterError(
                f"multiple Artifact adapters can materialize "
                f"{type(artifact).__name__} as {_type_name(target_type)}"
            )
        return matches[0].load(artifact, target_type)


class ArtifactHandleAdapter(ArtifactAdapter):
    """Deliver a verified SDK handle when a callable needs record identity.

    Most Computations request a domain value such as a pandas DataFrame or a
    CatBoost model. Declarative collection builders instead need no payload;
    they keep the handle so the runtime can use its exact digest-bound
    reference in an ArtifactSet.
    """

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return target_type is ArtifactHandle

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        return artifact


class PandasCsvAdapter(ArtifactAdapter):
    """Materialize a CSV Artifact as a pandas DataFrame."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "text/csv"
            and _is_pandas_dataframe_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from pandas import read_csv
        except ImportError as error:  # pragma: no cover - exercised by package users.
            raise RuntimeError(
                "PandasCsvAdapter requires pandas; install pandas in the application"
            ) from error
        return read_csv(BytesIO(artifact.read_verified_bytes()))


class PandasParquetAdapter(ArtifactAdapter):
    """Materialize a Parquet Artifact as a pandas DataFrame."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/vnd.apache.parquet"
            and _is_pandas_dataframe_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from pandas import read_parquet
        except ImportError as error:  # pragma: no cover - exercised by package users.
            raise RuntimeError(
                "PandasParquetAdapter requires pandas; install pandas in "
                "the application"
            ) from error
        try:
            return read_parquet(
                BytesIO(artifact.read_verified_bytes()), engine="pyarrow"
            )
        except ImportError as error:  # pragma: no cover - exercised by package users.
            raise RuntimeError(
                "PandasParquetAdapter requires pyarrow; install oclp[parquet]"
            ) from error


class PandasJsonTableAdapter(ArtifactAdapter):
    """Materialize a pandas-table JSON Artifact as a pandas DataFrame."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/json"
            and _is_pandas_dataframe_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from pandas import read_json
        except ImportError as error:  # pragma: no cover - exercised by package users.
            raise RuntimeError(
                "PandasJsonTableAdapter requires pandas; install pandas in "
                "the application"
            ) from error
        content = artifact.read_verified_bytes()
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactAdapterError(
                "PandasJsonTableAdapter requires valid JSON"
            ) from error
        if not _is_pandas_table_json(value):
            raise ArtifactAdapterError(
                "PandasJsonTableAdapter requires pandas table JSON "
                "(an object with 'schema' and 'data' fields)"
            )
        try:
            return read_json(BytesIO(content), orient="table")
        except (TypeError, ValueError) as error:
            raise ArtifactAdapterError(
                "PandasJsonTableAdapter could not load the declared pandas "
                "table JSON payload"
            ) from error


class JsonMappingAdapter(ArtifactAdapter):
    """Materialize a JSON Artifact as a Python mapping.

    This is intentionally separate from :class:`PandasJsonTableAdapter`:
    JSON is the durable representation, while a mapping and a DataFrame are
    distinct runtime choices made by the consuming callable's annotation.
    """

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return artifact.artifact.media_type == "application/json" and _is_mapping_type(
            target_type
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            value = json.loads(artifact.read_verified_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactAdapterError(
                "JsonMappingAdapter requires valid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactAdapterError(
                "JsonMappingAdapter requires a JSON object for a mapping input"
            )
        return value


class PandasJsonLinesAdapter(ArtifactAdapter):
    """Materialize a JSON Lines Artifact as a pandas DataFrame."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-ndjson"
            and _is_pandas_dataframe_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from pandas import read_json
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "PandasJsonLinesAdapter requires pandas; install pandas in the "
                "application"
            ) from error
        try:
            return read_json(BytesIO(artifact.read_verified_bytes()), lines=True)
        except ValueError as error:
            raise ArtifactAdapterError(
                "PandasJsonLinesAdapter requires valid JSON Lines"
            ) from error


class JsonLinesRecordsAdapter(ArtifactAdapter):
    """Materialize JSON Lines as a declared list or tuple of mappings."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-ndjson"
            and _is_mapping_collection_type(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            records = [
                json.loads(line)
                for line in artifact.read_verified_bytes().decode("utf-8").splitlines()
                if line
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactAdapterError(
                "JsonLinesRecordsAdapter requires valid JSON Lines"
            ) from error
        if not all(isinstance(record, dict) for record in records):
            raise ArtifactAdapterError(
                "JsonLinesRecordsAdapter requires every JSON Lines record to be an "
                "object"
            )
        return tuple(records) if _is_tuple_type(target_type) else records


class ArrowIpcTableAdapter(ArtifactAdapter):
    """Materialize an Arrow IPC file as a ``pyarrow.Table``."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/vnd.apache.arrow.file"
            and _is_pyarrow_table_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            import pyarrow as pa
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "ArrowIpcTableAdapter requires pyarrow; install oclp[arrow]"
            ) from error
        try:
            return pa.ipc.open_file(
                pa.BufferReader(artifact.read_verified_bytes())
            ).read_all()
        except (OSError, ValueError) as error:
            raise ArtifactAdapterError(
                "ArrowIpcTableAdapter requires a valid Arrow IPC file"
            ) from error


class PandasArrowIpcAdapter(ArtifactAdapter):
    """Materialize an Arrow IPC file as a pandas DataFrame."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/vnd.apache.arrow.file"
            and _is_pandas_dataframe_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            import pyarrow as pa
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "PandasArrowIpcAdapter requires pyarrow; install oclp[arrow]"
            ) from error
        try:
            table = pa.ipc.open_file(
                pa.BufferReader(artifact.read_verified_bytes())
            ).read_all()
        except (OSError, ValueError) as error:
            raise ArtifactAdapterError(
                "PandasArrowIpcAdapter requires a valid Arrow IPC file"
            ) from error
        return table.to_pandas()


class NpyArrayAdapter(ArtifactAdapter):
    """Materialize a NumPy ``.npy`` Artifact as an ``ndarray``."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-npy"
            and _is_numpy_array_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "NpyArrayAdapter requires numpy; install oclp[numpy]"
            ) from error
        try:
            return np.load(BytesIO(artifact.read_verified_bytes()), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ArtifactAdapterError(
                "NpyArrayAdapter requires a valid .npy file"
            ) from error


class NpzArrayArchiveAdapter(ArtifactAdapter):
    """Materialize an ``.npz`` Artifact as a mapping of named arrays."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return artifact.artifact.media_type == "application/x-npz" and _is_mapping_type(
            target_type
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "NpzArrayArchiveAdapter requires numpy; install oclp[numpy]"
            ) from error
        try:
            with np.load(
                BytesIO(artifact.read_verified_bytes()), allow_pickle=False
            ) as archive:
                return {name: archive[name] for name in archive.files}
        except (OSError, ValueError) as error:
            raise ArtifactAdapterError(
                "NpzArrayArchiveAdapter requires a valid .npz archive"
            ) from error


class YamlMappingAdapter(ArtifactAdapter):
    """Materialize a safe YAML mapping Artifact as a Python mapping."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return artifact.artifact.media_type == "application/yaml" and _is_mapping_type(
            target_type
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from yaml import YAMLError, safe_load
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "YamlMappingAdapter requires PyYAML; install oclp[yaml]"
            ) from error
        try:
            value = safe_load(artifact.read_verified_bytes())
        except YAMLError as error:
            raise ArtifactAdapterError(
                "YamlMappingAdapter requires valid YAML"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactAdapterError(
                "YamlMappingAdapter requires a YAML object for a mapping input"
            )
        return value


class TomlMappingAdapter(ArtifactAdapter):
    """Materialize a TOML Artifact as a Python mapping."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return artifact.artifact.media_type == "application/toml" and _is_mapping_type(
            target_type
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            value = tomllib.loads(artifact.read_verified_bytes().decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise ArtifactAdapterError(
                "TomlMappingAdapter requires valid TOML"
            ) from error
        if not isinstance(value, dict):  # pragma: no cover - tomllib contract guard.
            raise ArtifactAdapterError(
                "TomlMappingAdapter requires a TOML object for a mapping input"
            )
        return value


class XmlTextAdapter(ArtifactAdapter):
    """Materialize a safe UTF-8 XML Artifact as its original text."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return artifact.artifact.media_type == "application/xml" and target_type is str

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        content = artifact.read_verified_bytes()
        _safe_xml_root(content)
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:  # pragma: no cover - pre-parse guard.
            raise ArtifactAdapterError(
                "XmlTextAdapter requires a UTF-8 XML payload"
            ) from error


class XmlElementAdapter(ArtifactAdapter):
    """Materialize a safely parsed XML Artifact as an ElementTree Element."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/xml"
            and _is_xml_element_annotation(target_type)
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        return _safe_xml_root(artifact.read_verified_bytes())


def artifact_handle(published: PublishedArtifact) -> ArtifactHandle:
    """Return the internal resolved handle for a published Artifact.

    Concrete types such as :class:`CsvArtifact` and
    :class:`CatBoostModelArtifact` are declarations: they own a durable
    representation and its compatibility rules. A resolved handle stays
    generic until the consumer's annotated Python parameter selects an adapter.
    """

    return ArtifactHandle(published)


class ArtifactType(OclpModel, ABC):
    """A concrete durable representation used for Artifact inputs and outputs.

    An Artifact type is the SDK-level counterpart to the protocol's generic
    :class:`oclp.models.Artifact` record. It declares the representation's
    media type, validates and persists a Python return value, and supplies the
    compatibility rule for an input port. The same object is used by
    ``@csv_artifact`` acquisition and ``@computation(outputs=...)``.
    """

    id: str | ArtifactIdResolver | None = None
    # ``ArtifactType`` also acts as an input compatibility declaration, such
    # as ``inputs={"table": CsvArtifact}``, where no record is produced.
    # Keep the field optional for that use; every record-producing path
    # validates that an application supplied it rather than inventing a label.
    name: str | None = Field(default=None, min_length=1)
    key: str | None = Field(default=None, min_length=1)
    path: str | None = Field(default=None, min_length=1)
    profiles: ProfileBindings | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)
    schema_uri: str | None = Field(default=None, min_length=1)
    media_types: ClassVar[tuple[str, ...]] = ()

    @field_validator("id")
    @classmethod
    def artifact_id_is_nonempty(cls, value: str | ArtifactIdResolver | None):
        if isinstance(value, str) and not value:
            raise ValueError("Artifact IDs must be non-empty")
        return value

    def resolve_id(
        self,
        *,
        function: Callable[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        default: str,
    ) -> str:
        """Resolve an explicitly declared Artifact ID for one function call."""

        declaration = self.id
        if declaration is None:
            return default
        if isinstance(declaration, str):
            return declaration

        bound = signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        resolver_signature = signature(declaration)
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in resolver_signature.parameters.values()
        )
        resolver_arguments = (
            arguments
            if accepts_kwargs
            else {
                name: value
                for name, value in arguments.items()
                if name in resolver_signature.parameters
            }
        )
        try:
            artifact_id = declaration(**resolver_arguments)
        except TypeError as error:
            raise ValueError(
                "Artifact ID resolver must accept named parameters from "
                f"{function.__qualname__}"
            ) from error
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("Artifact ID resolvers must return a non-empty string")
        return artifact_id

    def validate_id_resolver(self, function: Callable[..., object]) -> None:
        """Reject ID resolvers that require parameters absent from the callable."""

        if not callable(self.id):
            return
        function_signature = signature(function)
        function_parameters = function_signature.parameters
        accepts_var_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in function_parameters.values()
        )
        resolver_signature = signature(self.id)
        missing = [
            parameter.name
            for parameter in resolver_signature.parameters.values()
            if parameter.kind
            in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            and parameter.default is Parameter.empty
            and parameter.name not in function_parameters
            and not accepts_var_kwargs
        ]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(
                f"Artifact ID resolver parameters {names} do not exist on "
                f"{function.__qualname__}"
            )

    @abstractmethod
    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        """Validate, materialize, and publish the returned value."""

    def handle(self, published: PublishedArtifact) -> ArtifactHandle:
        """Return the generic resolved handle for a published payload."""

        return artifact_handle(published)


class CsvArtifact(ArtifactType):
    """Pandas DataFrame representation persisted as a ``text/csv`` Artifact."""

    media_types: ClassVar[tuple[str, ...]] = ("text/csv",)
    media_type: Literal["text/csv"] = "text/csv"
    suffix: Literal["csv"] = "csv"

    index: bool = False
    lineterminator: str = Field(default="\n", min_length=1)
    na_rep: str = ""
    float_format: str | None = None
    date_format: str | None = None
    columns: tuple[str, ...] | None = None

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        to_csv = getattr(value, "to_csv", None)
        if not callable(to_csv):
            raise TypeError(
                "CsvArtifact requires a pandas DataFrame or another value with "
                "a compatible to_csv() method"
            )
        options = {
            "index": self.index,
            "lineterminator": self.lineterminator,
            "na_rep": self.na_rep,
            "float_format": self.float_format,
            "date_format": self.date_format,
            "columns": list(self.columns) if self.columns is not None else None,
        }
        try:
            serializer_signature = signature(to_csv)
        except (TypeError, ValueError):  # pragma: no cover - unusual callables.
            supported_options = options
        else:
            accepts_keywords = any(
                parameter.kind is Parameter.VAR_KEYWORD
                for parameter in serializer_signature.parameters.values()
            )
            supported_options = (
                options
                if accepts_keywords
                else {
                    option: option_value
                    for option, option_value in options.items()
                    if option in serializer_signature.parameters
                }
            )
        text = to_csv(**supported_options)
        if not isinstance(text, str):  # pragma: no cover - pandas contract guard.
            raise TypeError("pandas DataFrame.to_csv() did not return text")
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=text.encode("utf-8"),
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


def csv_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    index: bool = False,
    lineterminator: str = "\n",
    na_rep: str = "",
    float_format: str | None = None,
    date_format: str | None = None,
    columns: tuple[str, ...] | None = None,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a DataFrame-producing callable into a CSV Artifact boundary.

    The decorated callable requires an active :class:`oclp.runtime.OclpRun`.
    Its body still returns a pandas DataFrame, but callers receive a
    generic :class:`ArtifactHandle`. The runtime later adapts that durable handle
    to a Computation parameter's annotated in-memory type.
    """

    decorator = _decorate_artifact(
        CsvArtifact(
            id=id,
            name=name,
            index=index,
            lineterminator=lineterminator,
            na_rep=na_rep,
            float_format=float_format,
            date_format=date_format,
            columns=columns,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )
    return cast(
        Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]],
        decorator,
    )


class ParquetArtifact(ArtifactType):
    """Pandas DataFrame representation persisted as a Parquet Artifact."""

    media_types: ClassVar[tuple[str, ...]] = ("application/vnd.apache.parquet",)
    media_type: Literal["application/vnd.apache.parquet"] = (
        "application/vnd.apache.parquet"
    )
    suffix: Literal["parquet"] = "parquet"

    index: bool = False
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "zstd"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        dataframe = _pandas_dataframe(value, decorator_name="ParquetArtifact")
        destination = BytesIO()
        try:
            dataframe.to_parquet(
                destination,
                engine="pyarrow",
                index=self.index,
                compression=self.compression,
            )
        except ImportError as error:  # pragma: no cover - exercised by package users.
            raise RuntimeError(
                "@parquet_artifact requires pyarrow; install oclp[parquet]"
            ) from error
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=destination.getvalue(),
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


def parquet_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    index: bool = False,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "zstd",
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a DataFrame-producing callable into a Parquet Artifact boundary."""

    decorator = _decorate_artifact(
        ParquetArtifact(
            id=id,
            name=name,
            index=index,
            compression=compression,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )
    return cast(
        Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]],
        decorator,
    )


class JsonArtifact(ArtifactType):
    """Persistence policy for one value represented as JSON.

    ``serialization='json'`` accepts ordinary JSON-compatible Python values.
    ``serialization='pandas-table'`` is an explicit opt-in for a DataFrame
    encoded with pandas' ``orient='table'`` convention.  Both produce the same
    durable JSON representation; adapters determine how a
    consumer interprets the bytes.
    """

    serialization: Literal["json", "pandas-table"] = "json"
    media_types: ClassVar[tuple[str, ...]] = ("application/json",)
    media_type: Literal["application/json"] = "application/json"
    suffix: Literal["json"] = "json"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if self.serialization == "pandas-table":
            dataframe = _pandas_dataframe(value, decorator_name="JsonArtifact")
            text = dataframe.to_json(orient="table", index=False, date_format="iso")
            if not isinstance(text, str):  # pragma: no cover - pandas contract guard.
                raise TypeError("pandas DataFrame.to_json() did not return text")
            return publisher.artifact_for_bytes(
                artifact_id=artifact_id,
                name=name,
                relative_path=relative_path,
                content=text.encode("utf-8"),
                media_type=self.media_type,
                created_at=created_at,
                profiles=self.profiles,
                annotations=self.annotations,
                schema_uri=self.schema_uri,
            )
        return publisher.json_artifact(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            value=value,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class JsonLinesArtifact(ArtifactType):
    """JSON Lines representation for a table or a sequence of JSON objects.

    This is deliberately a separate type from :class:`JsonArtifact`: one line
    per record is a distinct durable representation, not merely a cosmetic JSON
    serializer option. It accepts a pandas DataFrame or an iterable of mapping
    records and writes canonical compact JSON records separated by ``newline``.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/x-ndjson",)
    media_type: Literal["application/x-ndjson"] = "application/x-ndjson"
    suffix: Literal["jsonl"] = "jsonl"

    newline: Literal["\n", "\r\n"] = "\n"
    sort_keys: bool = True
    ensure_ascii: bool = False

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        records = _json_line_records(value)
        try:
            content = (
                self.newline.join(
                    json.dumps(
                        record,
                        sort_keys=self.sort_keys,
                        ensure_ascii=self.ensure_ascii,
                        separators=(",", ":"),
                    )
                    for record in records
                )
                + self.newline
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise TypeError(
                "JsonLinesArtifact requires JSON-compatible record objects"
            ) from error
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class ArrowIpcArtifact(ArtifactType):
    """Apache Arrow IPC file representation for a table-like value."""

    media_types: ClassVar[tuple[str, ...]] = ("application/vnd.apache.arrow.file",)
    media_type: Literal["application/vnd.apache.arrow.file"] = (
        "application/vnd.apache.arrow.file"
    )
    suffix: Literal["arrow"] = "arrow"

    preserve_index: bool = False
    compression: Literal["lz4", "zstd"] | None = "zstd"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        try:
            import pyarrow as pa
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "ArrowIpcArtifact requires pyarrow; install oclp[arrow]"
            ) from error
        if isinstance(value, pa.Table):
            table = value
        elif _is_pandas_dataframe(value):
            table = pa.Table.from_pandas(value, preserve_index=self.preserve_index)
        else:
            raise TypeError(
                "ArrowIpcArtifact requires a pyarrow.Table or pandas.DataFrame"
            )
        destination = pa.BufferOutputStream()
        options = pa.ipc.IpcWriteOptions(compression=self.compression)
        with pa.ipc.new_file(destination, table.schema, options=options) as writer:
            writer.write_table(table)
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=destination.getvalue().to_pybytes(),
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class NpyArtifact(ArtifactType):
    """One NumPy array persisted as a safe ``.npy`` Artifact."""

    media_types: ClassVar[tuple[str, ...]] = ("application/x-npy",)
    media_type: Literal["application/x-npy"] = "application/x-npy"
    suffix: Literal["npy"] = "npy"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        array = _numpy_array(value, artifact_name="NpyArtifact")
        destination = BytesIO()
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "NpyArtifact requires numpy; install oclp[numpy]"
            ) from error
        np.save(destination, array, allow_pickle=False)
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=destination.getvalue(),
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class NpzArtifact(ArtifactType):
    """Named NumPy arrays persisted as a deterministic ``.npz`` archive."""

    media_types: ClassVar[tuple[str, ...]] = ("application/x-npz",)
    media_type: Literal["application/x-npz"] = "application/x-npz"
    suffix: Literal["npz"] = "npz"

    compression: Literal["stored", "deflated"] = "deflated"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, Mapping) or not value:
            raise TypeError(
                "NpzArtifact requires a non-empty mapping of names to numpy arrays"
            )
        destination = BytesIO()
        compression = (
            zipfile.ZIP_STORED if self.compression == "stored" else zipfile.ZIP_DEFLATED
        )
        with zipfile.ZipFile(destination, mode="w", compression=compression) as archive:
            for array_name in sorted(value):
                if not isinstance(array_name, str) or not array_name:
                    raise TypeError("NpzArtifact array names must be non-empty strings")
                array = _numpy_array(value[array_name], artifact_name="NpzArtifact")
                array_bytes = BytesIO()
                try:
                    import numpy as np
                except ImportError as error:  # pragma: no cover - package user path.
                    raise RuntimeError(
                        "NpzArtifact requires numpy; install oclp[numpy]"
                    ) from error
                np.save(array_bytes, array, allow_pickle=False)
                member = zipfile.ZipInfo(
                    filename=f"{array_name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                member.compress_type = compression
                archive.writestr(member, array_bytes.getvalue())
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=destination.getvalue(),
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class YamlArtifact(ArtifactType):
    """Safe YAML mapping representation for durable configuration data."""

    media_types: ClassVar[tuple[str, ...]] = ("application/yaml",)
    media_type: Literal["application/yaml"] = "application/yaml"
    suffix: Literal["yaml"] = "yaml"

    indent: int = Field(default=2, ge=1)
    width: int = Field(default=88, ge=1)
    sort_keys: bool = True

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, Mapping):
            raise TypeError("YamlArtifact requires a mapping value")
        try:
            from yaml import YAMLError, safe_dump
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "YamlArtifact requires PyYAML; install oclp[yaml]"
            ) from error
        try:
            content = safe_dump(
                dict(value),
                allow_unicode=True,
                default_flow_style=False,
                indent=self.indent,
                sort_keys=self.sort_keys,
                width=self.width,
            ).encode("utf-8")
        except YAMLError as error:
            raise TypeError(
                "YamlArtifact requires safe YAML-compatible values"
            ) from error
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class TomlArtifact(ArtifactType):
    """TOML mapping representation for durable configuration data."""

    media_types: ClassVar[tuple[str, ...]] = ("application/toml",)
    media_type: Literal["application/toml"] = "application/toml"
    suffix: Literal["toml"] = "toml"

    multiline_strings: bool = False

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, Mapping):
            raise TypeError("TomlArtifact requires a mapping value")
        try:
            from tomli_w import dumps
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "TomlArtifact requires tomli-w; install oclp[toml]"
            ) from error
        try:
            content = dumps(
                dict(value),
                multiline_strings=self.multiline_strings,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise TypeError("TomlArtifact requires TOML-compatible values") from error
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class XmlArtifact(ArtifactType):
    """Well-formed, UTF-8 XML without DTD or entity processing.

    The returned XML text remains the durable representation: the SDK validates
    it but deliberately does not canonicalize or otherwise rewrite it. XML
    canonicalization is an application-level semantic choice, while OCLP's
    digest always identifies the exact persisted bytes.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/xml",)
    media_type: Literal["application/xml"] = "application/xml"
    suffix: Literal["xml"] = "xml"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, str):
            raise TypeError(
                "XmlArtifact requires the decorated function to return XML text"
            )
        content = value.encode("utf-8")
        _require_utf8_xml_declaration(value)
        _safe_xml_root(content)
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class BytesArtifact(ArtifactType):
    """Raw bytes persisted with an explicitly declared media type and suffix."""

    media_type: str = Field(default="application/octet-stream", min_length=1)
    suffix: str = Field(default="bin", min_length=1)

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, bytes):
            raise TypeError("BytesArtifact requires the function to return bytes")
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=value,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class FileArtifact(ArtifactType):
    """An existing local file copied into the Artifact payload store.

    Use a concrete representation such as :class:`CatBoostModelArtifact` when
    the SDK knows how to serialize the returned Python object directly.
    """

    media_type: str = Field(min_length=1)
    suffix: str = Field(min_length=1)

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        if not isinstance(value, Path):
            raise TypeError("FileArtifact requires the function to return pathlib.Path")
        return publisher.artifact_for_file(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            source_path=value,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class CatBoostModelArtifact(ArtifactType):
    """Native CatBoost ``.cbm`` model representation.

    A decorated function returns its fitted CatBoost model. The SDK writes a
    temporary native model file, copies its bytes into the configured Artifact
    store, and later loads a verified payload for an annotated CatBoost input.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/x-catboost-model",)
    media_type: Literal["application/x-catboost-model"] = "application/x-catboost-model"
    suffix: Literal["cbm"] = "cbm"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        save_model = getattr(value, "save_model", None)
        if not callable(save_model):
            raise TypeError(
                "CatBoostModelArtifact requires a fitted CatBoost model with "
                "save_model()"
            )
        with TemporaryDirectory(prefix="oclp-catboost-") as directory:
            source_path = Path(directory) / "model.cbm"
            save_model(str(source_path))
            content = source_path.read_bytes()
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class CatBoostModelAdapter(ArtifactAdapter):
    """Load a verified CatBoost Artifact as the annotated estimator class."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-catboost-model"
            and getattr(target_type, "__module__", "").startswith("catboost")
            and getattr(target_type, "__name__", "")
            in {"CatBoostRegressor", "CatBoostClassifier", "CatBoostRanker"}
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        if not isinstance(target_type, type):  # pragma: no cover - guarded above.
            raise TypeError("CatBoostModelAdapter requires a CatBoost class annotation")
        artifact.read_verified_bytes()
        try:
            model = target_type()
            model.load_model(str(artifact.path))
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "CatBoostModelArtifact requires catboost; install oclp[catboost]"
            ) from error
        return model


class XGBoostModelArtifact(ArtifactType):
    """Native XGBoost UBJSON (``.ubj``) model representation.

    A decorated function returns a fitted XGBoost sklearn estimator or a
    ``Booster``. The SDK writes the native UBJSON model into its Artifact
    payload store. UBJSON is explicit here rather than relying on an XGBoost
    version's default serialization format.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/x-xgboost-ubjson",)
    media_type: Literal["application/x-xgboost-ubjson"] = "application/x-xgboost-ubjson"
    suffix: Literal["ubj"] = "ubj"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        save_model = getattr(value, "save_model", None)
        if not callable(save_model):
            raise TypeError(
                "XGBoostModelArtifact requires a fitted XGBoost model with save_model()"
            )
        with TemporaryDirectory(prefix="oclp-xgboost-") as directory:
            source_path = Path(directory) / "model.ubj"
            save_model(str(source_path))
            content = source_path.read_bytes()
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class XGBoostModelAdapter(ArtifactAdapter):
    """Load a verified XGBoost UBJSON Artifact as its annotated model type."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-xgboost-ubjson"
            and getattr(target_type, "__module__", "").startswith("xgboost")
            and getattr(target_type, "__name__", "")
            in {"Booster", "XGBRegressor", "XGBClassifier", "XGBRanker"}
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        if not isinstance(target_type, type):  # pragma: no cover - guarded above.
            raise TypeError("XGBoostModelAdapter requires an XGBoost class annotation")
        artifact.read_verified_bytes()
        try:
            model = target_type()
            model.load_model(str(artifact.path))
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "XGBoostModelArtifact requires xgboost; install oclp[xgboost]"
            ) from error
        return model


class LightGBMModelArtifact(ArtifactType):
    """Native LightGBM Booster model representation.

    LightGBM's portable native model is its ``Booster`` payload. Its sklearn
    wrappers expose that fitted Booster through ``booster_`` but do not offer a
    stable native wrapper loader, so this Artifact intentionally accepts and
    returns ``lightgbm.Booster`` rather than rebuilding wrapper internals.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/x-lightgbm-model",)
    media_type: Literal["application/x-lightgbm-model"] = "application/x-lightgbm-model"
    suffix: Literal["txt"] = "txt"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        try:
            from lightgbm import Booster
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "LightGBMModelArtifact requires lightgbm; install oclp[lightgbm]"
            ) from error
        if not isinstance(value, Booster):
            raise TypeError(
                "LightGBMModelArtifact requires a fitted lightgbm.Booster; use "
                "SklearnModelArtifact for a sklearn wrapper object"
            )
        with TemporaryDirectory(prefix="oclp-lightgbm-") as directory:
            source_path = Path(directory) / "model.txt"
            value.save_model(str(source_path))
            content = source_path.read_bytes()
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class LightGBMModelAdapter(ArtifactAdapter):
    """Load a verified native LightGBM Artifact as a ``Booster``."""

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-lightgbm-model"
            and getattr(target_type, "__module__", "").startswith("lightgbm")
            and getattr(target_type, "__name__", "") == "Booster"
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        try:
            from lightgbm import Booster
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "LightGBMModelArtifact requires lightgbm; install oclp[lightgbm]"
            ) from error
        artifact.read_verified_bytes()
        try:
            return Booster(model_file=str(artifact.path))
        except (OSError, ValueError) as error:
            raise ArtifactAdapterError(
                "LightGBMModelAdapter requires a valid LightGBM model file"
            ) from error


class SklearnModelArtifact(ArtifactType):
    """Fitted scikit-learn estimator persisted in the ``skops`` format.

    The SDK deliberately uses the inspectable ``skops`` format instead of a
    pickle-based format. This preserves a normal fitted estimator boundary
    while requiring the consumer to explicitly configure any non-default
    trusted types before they can be constructed.
    """

    media_types: ClassVar[tuple[str, ...]] = ("application/x-skops",)
    media_type: Literal["application/x-skops"] = "application/x-skops"
    suffix: Literal["skops"] = "skops"

    def persist(
        self,
        *,
        publisher: LocalArtifactPublisher,
        artifact_id: str,
        name: str,
        relative_path: str,
        value: object,
        created_at: datetime,
    ) -> PublishedArtifact:
        try:
            from sklearn.base import BaseEstimator
            from skops.io import dumps
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "SklearnModelArtifact requires scikit-learn and skops; "
                "install oclp[sklearn]"
            ) from error
        if not isinstance(value, BaseEstimator):
            raise TypeError(
                "SklearnModelArtifact requires a fitted scikit-learn BaseEstimator"
            )
        content = dumps(value)
        return publisher.artifact_for_bytes(
            artifact_id=artifact_id,
            name=name,
            relative_path=relative_path,
            content=content,
            media_type=self.media_type,
            created_at=created_at,
            profiles=self.profiles,
            annotations=self.annotations,
            schema_uri=self.schema_uri,
        )


class SklearnModelAdapter(ArtifactAdapter):
    """Load a verified ``skops`` Artifact as an annotated sklearn estimator.

    ``trusted_types`` is consumer policy. The Artifact producer never gets to
    declare which unknown types a consumer should construct. The default allows
    only types that ``skops`` considers trusted by default.
    """

    def __init__(self, *, trusted_types: tuple[str, ...] = ()) -> None:
        self.trusted_types = trusted_types

    def supports(self, artifact: ArtifactHandle, target_type: object) -> bool:
        return (
            artifact.artifact.media_type == "application/x-skops"
            and isinstance(target_type, type)
            and getattr(target_type, "__module__", "").startswith("sklearn")
        )

    def load(self, artifact: ArtifactHandle, target_type: object) -> object:
        if not isinstance(target_type, type):  # pragma: no cover - guarded above.
            raise TypeError("SklearnModelAdapter requires a sklearn class annotation")
        try:
            from skops.io import get_untrusted_types, loads
        except ImportError as error:  # pragma: no cover - package user path.
            raise RuntimeError(
                "SklearnModelArtifact requires scikit-learn and skops; "
                "install oclp[sklearn]"
            ) from error
        content = artifact.read_verified_bytes()
        untrusted = set(get_untrusted_types(data=content))
        permitted = set(self.trusted_types)
        rejected = sorted(untrusted.difference(permitted))
        if rejected:
            names = ", ".join(rejected)
            raise ArtifactAdapterError(
                "SklearnModelArtifact contains untrusted types: "
                f"{names}. Register SklearnModelAdapter(trusted_types=(...)) "
                "only after reviewing them."
            )
        model = loads(content, trusted=list(self.trusted_types))
        if not isinstance(model, target_type):
            raise ArtifactAdapterError(
                "SklearnModelArtifact loaded "
                f"{type(model).__module__}.{type(model).__qualname__}, not "
                f"the declared {_type_name(target_type)}"
            )
        return model


DEFAULT_ARTIFACT_ADAPTERS = ArtifactAdapterRegistry(
    (
        ArtifactHandleAdapter(),
        PandasCsvAdapter(),
        PandasParquetAdapter(),
        PandasJsonTableAdapter(),
        JsonMappingAdapter(),
        PandasJsonLinesAdapter(),
        JsonLinesRecordsAdapter(),
        ArrowIpcTableAdapter(),
        PandasArrowIpcAdapter(),
        NpyArrayAdapter(),
        NpzArrayArchiveAdapter(),
        YamlMappingAdapter(),
        TomlMappingAdapter(),
        XmlTextAdapter(),
        XmlElementAdapter(),
        CatBoostModelAdapter(),
        XGBoostModelAdapter(),
        LightGBMModelAdapter(),
        SklearnModelAdapter(),
    )
)


def json_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
    serialization: Literal["json", "pandas-table"] = "json",
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a JSON-compatible callable result into a durable JSON Artifact.

    Pass ``serialization='pandas-table'`` only when a DataFrame result should
    use pandas' ``orient='table'`` convention and be loadable by the pandas
    table adapter.  The default is generic JSON suitable for mapping-oriented
    consumers.
    """

    decorator = _decorate_artifact(
        JsonArtifact(
            id=id,
            name=name,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
            serialization=serialization,
        )
    )
    return cast(
        Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]],
        decorator,
    )


def json_lines_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    newline: Literal["\n", "\r\n"] = "\n",
    sort_keys: bool = True,
    ensure_ascii: bool = False,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn table or record-producing code into a JSON Lines Artifact boundary."""

    return _artifact_decorator_for(
        JsonLinesArtifact(
            id=id,
            name=name,
            newline=newline,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def arrow_ipc_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    preserve_index: bool = False,
    compression: Literal["lz4", "zstd"] | None = "zstd",
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a PyArrow table or pandas DataFrame into an Arrow IPC Artifact."""

    return _artifact_decorator_for(
        ArrowIpcArtifact(
            id=id,
            name=name,
            preserve_index=preserve_index,
            compression=compression,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def npy_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn an ndarray-producing callable into a safe ``.npy`` Artifact."""

    return _artifact_decorator_for(
        NpyArtifact(
            id=id,
            name=name,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def npz_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    compression: Literal["stored", "deflated"] = "deflated",
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn named-array code into a deterministic ``.npz`` Artifact boundary."""

    return _artifact_decorator_for(
        NpzArtifact(
            id=id,
            name=name,
            compression=compression,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def yaml_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    indent: int = 2,
    width: int = 88,
    sort_keys: bool = True,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a mapping-producing callable into a safe YAML Artifact boundary."""

    return _artifact_decorator_for(
        YamlArtifact(
            id=id,
            name=name,
            indent=indent,
            width=width,
            sort_keys=sort_keys,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def toml_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    multiline_strings: bool = False,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn a mapping-producing callable into a TOML Artifact boundary."""

    return _artifact_decorator_for(
        TomlArtifact(
            id=id,
            name=name,
            multiline_strings=multiline_strings,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def xml_artifact(
    *,
    id: str | ArtifactIdResolver | None = None,
    name: str,
    profiles: ProfileBindings | None = None,
    annotations: dict[str, JsonValue] | None = None,
    schema_uri: str | None = None,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Turn an XML-text-producing callable into a safe XML Artifact boundary.

    The SDK rejects DTDs and entity declarations, preserving the returned UTF-8
    XML text only after safe well-formedness validation.
    """

    return _artifact_decorator_for(
        XmlArtifact(
            id=id,
            name=name,
            profiles=profiles,
            annotations=annotations or {},
            schema_uri=schema_uri,
        )
    )


def _artifact_decorator_for(
    artifact: ArtifactType,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    """Return a precisely typed wrapper around the common decorator machinery."""

    return cast(
        Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]],
        _decorate_artifact(artifact),
    )


def artifact_type(function: Callable[..., object]) -> ArtifactType:
    """Return the Artifact type bound to an acquisition callable."""

    declared_type = getattr(function, _ARTIFACT_TYPE_ATTRIBUTE, None)
    if not isinstance(declared_type, ArtifactType):
        name = getattr(function, "__qualname__", repr(function))
        raise ValueError(f"callable {name!r} has no OCLP Artifact type")
    return declared_type


def _decorate_artifact(
    artifact: ArtifactType,
) -> Callable[[Callable[Parameters, object]], Callable[Parameters, ArtifactHandle]]:
    def decorate(
        function: Callable[Parameters, object],
    ) -> Callable[Parameters, ArtifactHandle]:
        if not callable(function):
            raise TypeError("Artifact decorators can only decorate a callable")
        if getattr(function, _ARTIFACT_TYPE_ATTRIBUTE, None) is not None:
            raise ValueError("a callable can have only one OCLP Artifact type")
        artifact.validate_id_resolver(function)
        try:
            setattr(function, _ARTIFACT_TYPE_ATTRIBUTE, artifact)
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "Artifact decorators require a callable that accepts attached metadata"
            ) from error

        @wraps(function)
        def observed(
            *args: Parameters.args,
            **kwargs: Parameters.kwargs,
        ) -> ArtifactHandle:
            # Local import keeps declaration independent of runtime state and
            # avoids an import cycle.
            from oclp.runtime import active_run

            run = active_run()
            if run is None:
                raise RuntimeError(
                    "Artifact-producing decorators require an active OclpRun"
                )
            return cast(
                ArtifactHandle,
                run.acquire(function, artifact, tuple(args), kwargs),
            )

        setattr(observed, _ARTIFACT_TYPE_ATTRIBUTE, artifact)
        return observed

    return decorate


def _pandas_dataframe(value: object, *, decorator_name: str):
    dataframe_type = _pandas_dataframe_type()
    if not isinstance(value, dataframe_type):
        raise TypeError(
            f"{decorator_name} requires the decorated function to return "
            "pandas.DataFrame"
        )
    return value


def _is_pandas_dataframe(value: object) -> bool:
    """Check a value's type without importing pandas for unrelated Artifacts."""

    return _is_pandas_dataframe_annotation(type(value))


def _json_line_records(value: object) -> list[dict[str, object]]:
    """Normalize a DataFrame or iterable mapping collection for JSON Lines."""

    if _is_pandas_dataframe(value):
        records = _pandas_dataframe(value, decorator_name="JsonLinesArtifact").to_dict(
            orient="records"
        )
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        records = list(value)
    else:
        raise TypeError(
            "JsonLinesArtifact requires a pandas.DataFrame or iterable of mapping "
            "records"
        )
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("JsonLinesArtifact requires every record to be a mapping")
    return [dict(record) for record in records]


def _numpy_array(value: object, *, artifact_name: str):
    """Require an ndarray while importing numpy only for its own integration."""

    try:
        from numpy import ndarray
    except ImportError as error:  # pragma: no cover - package user path.
        raise RuntimeError(
            f"{artifact_name} requires numpy; install oclp[numpy]"
        ) from error
    if not isinstance(value, ndarray):
        raise TypeError(f"{artifact_name} requires a numpy.ndarray")
    return value


def _pandas_dataframe_type():
    try:
        from pandas import DataFrame
    except ImportError as error:  # pragma: no cover - exercised by package users.
        raise RuntimeError(
            "Artifact DataFrame decorators require pandas; install pandas in "
            "the application"
        ) from error
    return DataFrame


def _is_pandas_dataframe_annotation(target_type: object) -> bool:
    """Check a pandas DataFrame annotation without importing pandas eagerly."""

    return getattr(target_type, "__name__", None) == "DataFrame" and getattr(
        target_type, "__module__", ""
    ).startswith("pandas.")


def _is_pyarrow_table_annotation(target_type: object) -> bool:
    """Check a ``pyarrow.Table`` annotation without importing PyArrow eagerly."""

    return getattr(target_type, "__name__", None) == "Table" and getattr(
        target_type, "__module__", ""
    ).startswith("pyarrow")


def _is_numpy_array_annotation(target_type: object) -> bool:
    """Check an ndarray annotation without importing NumPy eagerly."""

    return getattr(target_type, "__name__", None) == "ndarray" and getattr(
        target_type, "__module__", ""
    ).startswith("numpy")


def _safe_xml_root(content: bytes):
    """Safely parse XML while forbidding DTD and entity expansion attacks."""

    try:
        from defusedxml import ElementTree as safe_element_tree
        from defusedxml.common import DefusedXmlException
    except ImportError as error:  # pragma: no cover - package user path.
        raise RuntimeError(
            "XmlArtifact requires defusedxml; install oclp[xml]"
        ) from error
    try:
        return safe_element_tree.fromstring(content)
    except (DefusedXmlException, safe_element_tree.ParseError) as error:
        raise ArtifactAdapterError(
            "XmlArtifact requires well-formed XML without DTD or entity processing"
        ) from error


_XML_DECLARATION_ENCODING = re.compile(
    r"^\s*<\?xml\s+[^>]*\bencoding\s*=\s*(['\"])(?P<encoding>[^'\"]+)\1",
    re.IGNORECASE,
)


def _require_utf8_xml_declaration(value: str) -> None:
    """Reject declarations that disagree with the SDK's UTF-8 byte contract."""

    declaration = _XML_DECLARATION_ENCODING.match(value)
    if declaration is None:
        return
    encoding = declaration.group("encoding").replace("_", "-").lower()
    if encoding not in {"utf-8", "utf8"}:
        raise TypeError(
            "XmlArtifact persists UTF-8 bytes; XML declarations must specify UTF-8 "
            "or omit the encoding declaration"
        )


def _is_xml_element_annotation(target_type: object) -> bool:
    """Check an ``xml.etree.ElementTree.Element`` annotation lazily."""

    return (
        getattr(target_type, "__name__", None) == "Element"
        and getattr(target_type, "__module__", "") == "xml.etree.ElementTree"
    )


def _type_name(value: object) -> str:
    """Render an annotated runtime type in a useful adapter error message."""

    return getattr(value, "__qualname__", repr(value))


def _is_mapping_type(target_type: object) -> bool:
    """Return whether a callable annotation requests a mapping-like object."""

    origin = get_origin(target_type)
    return target_type in {dict, Mapping} or origin in {dict, Mapping}


def _is_mapping_collection_type(target_type: object) -> bool:
    """Return whether an annotation asks for a list or tuple of mappings."""

    origin = get_origin(target_type)
    collection_type = origin or target_type
    if collection_type not in {list, tuple}:
        return False
    arguments = get_args(target_type)
    if not arguments:
        return True
    return _is_mapping_type(arguments[0])


def _is_tuple_type(target_type: object) -> bool:
    """Return whether an annotation asks for a tuple rather than a list."""

    return target_type is tuple or get_origin(target_type) is tuple


def _is_pandas_table_json(value: Any) -> bool:
    """Check the structural minimum required by pandas ``orient='table'``."""

    return (
        isinstance(value, dict)
        and isinstance(value.get("schema"), dict)
        and isinstance(value.get("data"), list)
    )

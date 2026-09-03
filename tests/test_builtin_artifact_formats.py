"""Round-trip tests for the built-in structured-data Artifact integrations."""

from __future__ import annotations

import pytest

from oclp import (
    ArrowIpcArtifact,
    GitSource,
    NpyArtifact,
    NpzArtifact,
    OclpRun,
    TomlArtifact,
    YamlArtifact,
    computation,
)
from oclp.publishing import LocalArtifactPublisher


def _publisher(tmp_path):
    return LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    )


def _source() -> GitSource:
    return GitSource(
        repository="https://github.com/example/artifacts.git",
        commit="a" * 40,
    )


numpy = pytest.importorskip("numpy")
pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
pytest.importorskip("tomli_w")


@computation(
    id="urn:example:computation:publish-arrow",
    name="Publish Arrow table",
    outputs={"table": ArrowIpcArtifact(name="Example Arrow table")},
)
def publish_arrow() -> object:
    return pyarrow.table({"team": ["a", "b"], "score": [1, 2]})


@computation(
    id="urn:example:computation:sum-arrow",
    name="Sum Arrow table",
    inputs={"table": ArrowIpcArtifact},
)
def sum_arrow(table: pyarrow.Table) -> int:
    return int(sum(table.column("score").to_pylist()))


@computation(
    id="urn:example:computation:publish-npy",
    name="Publish NumPy array",
    outputs={"array": NpyArtifact(name="Example array")},
)
def publish_npy() -> numpy.ndarray:
    return numpy.array([[1.0, 2.0], [3.0, 4.0]])


@computation(
    id="urn:example:computation:sum-npy",
    name="Sum NumPy array",
    inputs={"array": NpyArtifact},
)
def sum_npy(array: numpy.ndarray) -> float:
    return float(array.sum())


@computation(
    id="urn:example:computation:publish-npz",
    name="Publish NumPy archive",
    outputs={"arrays": NpzArtifact(name="Example array archive")},
)
def publish_npz() -> dict[str, numpy.ndarray]:
    return {
        "features": numpy.array([[1.0], [2.0]]),
        "target": numpy.array([1.0, 2.0]),
    }


@computation(
    id="urn:example:computation:sum-npz",
    name="Sum NumPy archive",
    inputs={"arrays": NpzArtifact},
)
def sum_npz(arrays: dict[str, numpy.ndarray]) -> float:
    return float(arrays["features"].sum() + arrays["target"].sum())


@computation(
    id="urn:example:computation:publish-configs",
    name="Publish configuration formats",
    outputs={
        "yaml": YamlArtifact(name="Example YAML configuration"),
        "toml": TomlArtifact(name="Example TOML configuration"),
    },
)
def publish_configs() -> dict[str, dict[str, object]]:
    config = {"model": {"learning_rate": 0.1}, "name": "example"}
    return {"yaml": config, "toml": config}


@computation(
    id="urn:example:computation:read-yaml",
    name="Read YAML configuration",
    inputs={"config": YamlArtifact},
)
def read_yaml(config: dict[str, object]) -> str:
    return str(config["name"])


@computation(
    id="urn:example:computation:read-toml",
    name="Read TOML configuration",
    inputs={"config": TomlArtifact},
)
def read_toml(config: dict[str, object]) -> float:
    return float(config["model"]["learning_rate"])


def test_arrow_numpy_yaml_and_toml_artifacts_round_trip(tmp_path) -> None:
    with _publisher(tmp_path) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="builtin-format-round-trip",
            source=_source(),
        ) as observed:
            arrow_table = publish_arrow()
            arrow_handle = observed.outputs_for(arrow_table)["table"]
            array = publish_npy()
            npy_handle = observed.outputs_for(array)["array"]
            archive = publish_npz()
            npz_handle = observed.outputs_for(archive)["arrays"]
            configs = publish_configs()
            config_handles = observed.outputs_for(configs)

            arrow_total = sum_arrow(arrow_handle)
            npy_total = sum_npy(npy_handle)
            npz_total = sum_npz(npz_handle)
            yaml_name = read_yaml(config_handles["yaml"])
            toml_rate = read_toml(config_handles["toml"])

    assert arrow_total == 3
    assert npy_total == 10.0
    assert npz_total == 6.0
    assert yaml_name == "example"
    assert toml_rate == 0.1
    assert arrow_handle.artifact.media_type == "application/vnd.apache.arrow.file"
    assert npy_handle.path.suffix == ".npy"
    assert npz_handle.path.suffix == ".npz"
    assert config_handles["yaml"].path.suffix == ".yaml"
    assert config_handles["toml"].path.suffix == ".toml"

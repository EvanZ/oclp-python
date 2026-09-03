"""Round-trip and safety tests for the XML Artifact integration."""

from __future__ import annotations

from datetime import UTC, datetime
from xml.etree.ElementTree import Element

import pytest

from oclp import (
    ArtifactAdapterError,
    GitSource,
    OclpRun,
    XmlArtifact,
    computation,
    xml_artifact,
)
from oclp.publishing import LocalArtifactPublisher

pytest.importorskip("defusedxml")


@computation(
    id="urn:example:computation:publish-xml",
    name="Publish XML document",
    outputs={"document": XmlArtifact(name="Example XML document")},
)
def publish_xml() -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><report><score>7</score></report>'


@xml_artifact(name="Acquired XML document")
def acquire_xml() -> str:
    return "<source><name>bike-demand</name></source>"


@computation(
    id="urn:example:computation:read-xml-text",
    name="Read XML document text",
    inputs={"document": XmlArtifact},
)
def read_xml_text(document: str) -> int:
    return document.count("score")


@computation(
    id="urn:example:computation:read-xml-element",
    name="Read XML document element",
    inputs={"document": XmlArtifact},
)
def read_xml_element(document: Element) -> str:
    return document.findtext("score") or ""


def _publisher(tmp_path) -> LocalArtifactPublisher:
    return LocalArtifactPublisher(
        catalog_path=tmp_path / "records" / "catalog.duckdb",
        record_root=tmp_path / "records",
        payload_root=tmp_path / "payloads",
    )


def _source() -> GitSource:
    return GitSource(
        repository="https://github.com/example/xml.git",
        commit="a" * 40,
    )


def test_xml_artifact_round_trips_text_and_a_safe_element(tmp_path) -> None:
    with _publisher(tmp_path) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="xml-round-trip",
            source=_source(),
        ) as observed:
            result = publish_xml()
            handle = observed.outputs_for(result)["document"]
            assert read_xml_text(handle) == 2
            assert read_xml_element(handle) == "7"

    assert handle.artifact.media_type == "application/xml"
    assert handle.path.suffix == ".xml"
    assert handle.read_verified_bytes() == result.encode("utf-8")


def test_xml_artifact_rejects_dtds_and_non_utf8_declarations(tmp_path) -> None:
    artifact = XmlArtifact(name="Unsafe XML")
    with _publisher(tmp_path) as publisher:
        with pytest.raises(ArtifactAdapterError, match="DTD or entity"):
            artifact.persist(
                publisher=publisher,
                artifact_id="urn:example:artifact:unsafe-xml",
                name="Unsafe XML",
                relative_path="unsafe.xml",
                value='<!DOCTYPE report [<!ENTITY secret SYSTEM "file:///etc/passwd">]><report>&secret;</report>',
                created_at=datetime.now(UTC),
            )
        with pytest.raises(TypeError, match="UTF-8"):
            artifact.persist(
                publisher=publisher,
                artifact_id="urn:example:artifact:latin-xml",
                name="Latin XML",
                relative_path="latin.xml",
                value='<?xml version="1.0" encoding="ISO-8859-1"?><report/>',
                created_at=datetime.now(UTC),
            )


def test_xml_artifact_decorator_acquires_a_durable_document(tmp_path) -> None:
    with _publisher(tmp_path) as publisher:
        with OclpRun(
            publisher=publisher,
            namespace="urn:example",
            run_id="xml-acquisition",
            source=_source(),
        ) as observed:
            handle = acquire_xml()
            artifact = observed.artifact_for(handle)

    assert artifact.artifact.id == "urn:example:artifact:acquire-xml:xml-acquisition"
    assert handle.read_verified_bytes() == b"<source><name>bike-demand</name></source>"

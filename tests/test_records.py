from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp import (
    Artifact,
    ArtifactSet,
    ArtifactSetMember,
    Invocation,
    canonical_json_bytes,
    parse_record,
    record_digest,
)
from oclp.models import Digest, RecordReference

FIXTURES = Path(__file__).parent / "fixtures"


def test_canonical_serialization_is_stable() -> None:
    artifact = Artifact(
        id="artifact.example",
        media_type="application/octet-stream",
        digest=Digest(value="a" * 64),
        size=12,
    )

    assert canonical_json_bytes(artifact) == canonical_json_bytes(artifact)
    assert str(record_digest(artifact)).startswith("sha256:")


def test_record_name_is_optional_human_metadata_not_identity() -> None:
    artifact = Artifact(
        id="artifact.example",
        name="Input document",
        media_type="application/octet-stream",
        digest=Digest(value="a" * 64),
        size=12,
    )

    assert artifact.name == "Input document"
    assert b'"name":"Input document"' in canonical_json_bytes(artifact)
    with pytest.raises(ValidationError):
        Artifact(
            id="artifact.example",
            name="",
            media_type="application/octet-stream",
            digest=Digest(value="a" * 64),
            size=12,
        )


def test_artifact_and_artifact_set_created_at_are_immutable_record_metadata() -> None:
    created_at = datetime(2026, 8, 27, 5, tzinfo=UTC)
    artifact = Artifact(
        id="urn:example:artifact:created",
        media_type="text/plain",
        digest=Digest(value="a" * 64),
        size=12,
        created_at=created_at,
    )
    revised_artifact = artifact.model_copy(
        update={"created_at": datetime(2026, 8, 27, 5, 1, tzinfo=UTC)}
    )
    artifact_set = ArtifactSet(
        id="urn:example:artifact-set:created",
        members=(
            ArtifactSetMember(
                name="document",
                artifact=RecordReference(
                    id=artifact.id,
                    digest=record_digest(artifact),
                ),
            ),
        ),
        created_at=created_at,
    )

    assert b'"created_at":"2026-08-27T05:00:00Z"' in canonical_json_bytes(artifact)
    assert b'"created_at":"2026-08-27T05:00:00Z"' in canonical_json_bytes(artifact_set)
    assert record_digest(artifact) != record_digest(revised_artifact)


@pytest.mark.parametrize("record_type", (Artifact, ArtifactSet))
def test_created_at_requires_an_explicit_offset(
    record_type: type[Artifact | ArtifactSet],
) -> None:
    kwargs: dict[str, object] = {
        "id": "urn:example:artifact:created",
        "created_at": datetime(2026, 8, 27, 5),
    }
    if record_type is Artifact:
        kwargs.update(
            media_type="text/plain",
            digest=Digest(value="a" * 64),
            size=12,
        )
    else:
        kwargs.update(
            members=(
                ArtifactSetMember(
                    name="document",
                    artifact=RecordReference(
                        id="urn:example:artifact:document",
                        digest=Digest(value="b" * 64),
                    ),
                ),
            ),
        )

    with pytest.raises(ValidationError, match="created_at must include a UTC offset"):
        record_type(**kwargs)


def test_legacy_invocation_omits_absent_outputs_from_canonical_json() -> None:
    invocation = Invocation(
        id="invocation.example",
        definition=RecordReference(id="definition.example"),
    )

    assert b'"outputs"' not in canonical_json_bytes(invocation)
    assert b'"name"' not in canonical_json_bytes(invocation)


def test_profiles_are_a_typed_contract_surface_distinct_from_annotations() -> None:
    invocation = Invocation(
        id="invocation.example",
        definition=RecordReference(id="definition.example"),
        profiles={"lifecycle": {"version": "0.1.0-draft"}},
        annotations={"example.org/owner": "data-platform"},
    )

    assert invocation.profiles["lifecycle"]["version"] == "0.1.0-draft"
    assert invocation.annotations == {"example.org/owner": "data-platform"}
    assert b'"profiles"' in canonical_json_bytes(invocation)
    with pytest.raises(ValidationError):
        Invocation(
            id="invalid-profile.example",
            definition=RecordReference(id="definition.example"),
            profiles={"": {"version": "0.1.0-draft"}},
        )


def test_unprofiled_records_canonically_declare_null_profiles() -> None:
    invocation = Invocation(
        id="invocation.example",
        definition=RecordReference(id="definition.example"),
    )

    assert invocation.profiles is None
    assert b'"profiles":null' in canonical_json_bytes(invocation)

    with pytest.raises(ValidationError):
        Invocation(
            id="empty-profiles.example",
            definition=RecordReference(id="definition.example"),
            profiles={},
        )


def test_valid_conformance_fixtures_are_accepted() -> None:
    for path in (FIXTURES / "valid").glob("*.json"):
        parse_record(json.loads(path.read_text()))


def test_invalid_conformance_fixtures_are_rejected() -> None:
    for path in (FIXTURES / "invalid").glob("*.json"):
        with pytest.raises(ValidationError):
            parse_record(json.loads(path.read_text()))

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
    Computation,
    Execution,
    GitSource,
    canonical_json_bytes,
    parse_record,
    record_digest,
)
from oclp.models import Digest, Implementation, RecordReference

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


def test_execution_omits_absent_outputs_from_canonical_json() -> None:
    execution = Execution(
        id="execution.example",
        computation=RecordReference(id="computation.example"),
    )

    assert b'"outputs"' not in canonical_json_bytes(execution)
    assert b'"name"' not in canonical_json_bytes(execution)


def test_computation_omits_absent_evidence_requirements_from_canonical_json() -> None:
    computation = Computation(
        id="urn:example:computation:unqualified",
        implementation=Implementation(
            kind="other",
            locator="example:unqualified",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )

    assert computation.required_evidence is None
    assert b'"required_evidence"' not in canonical_json_bytes(computation)


def test_git_source_overlay_requires_an_explicit_dirty_marker() -> None:
    overlay = RecordReference(
        id="urn:example:artifact-set:source-overlay",
        digest=Digest(value="a" * 64),
    )

    with pytest.raises(ValidationError, match="dirty=true"):
        GitSource(
            repository="https://github.com/example/reports.git",
            commit="b" * 40,
            overlay=overlay,
        )

    source = GitSource(
        repository="https://github.com/example/reports.git",
        commit="b" * 40,
        dirty=True,
        overlay=overlay,
    )
    assert source.dirty is True


def test_profiles_are_a_typed_contract_surface_distinct_from_annotations() -> None:
    execution = Execution(
        id="execution.example",
        computation=RecordReference(id="computation.example"),
        profiles={"lifecycle": {"version": "0.2.0-draft"}},
        annotations={"example.org/owner": "data-platform"},
    )

    assert execution.profiles["lifecycle"]["version"] == "0.2.0-draft"
    assert execution.annotations == {"example.org/owner": "data-platform"}
    assert b'"profiles"' in canonical_json_bytes(execution)
    with pytest.raises(ValidationError):
        Execution(
            id="invalid-profile.example",
            computation=RecordReference(id="computation.example"),
            profiles={"": {"version": "0.2.0-draft"}},
        )


def test_unprofiled_records_canonically_declare_null_profiles() -> None:
    execution = Execution(
        id="execution.example",
        computation=RecordReference(id="computation.example"),
    )

    assert execution.profiles is None
    assert b'"profiles":null' in canonical_json_bytes(execution)

    with pytest.raises(ValidationError):
        Execution(
            id="empty-profiles.example",
            computation=RecordReference(id="computation.example"),
            profiles={},
        )


def test_valid_conformance_fixtures_are_accepted() -> None:
    for path in (FIXTURES / "valid").glob("*.json"):
        parse_record(json.loads(path.read_text()))


def test_invalid_conformance_fixtures_are_rejected() -> None:
    for path in (FIXTURES / "invalid").glob("*.json"):
        with pytest.raises(ValidationError):
            parse_record(json.loads(path.read_text()))

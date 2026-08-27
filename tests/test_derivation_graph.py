"""Tests for the resolved OCLP derivation-DAG contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oclp import (
    Artifact,
    ArtifactSet,
    ArtifactSetMember,
    ComputationDefinition,
    DerivationValidationError,
    Invocation,
    OrchestrationValidationError,
    record_digest,
    validate_derivation_graph,
    validate_invocation_hierarchy,
)
from oclp.models import Digest, Implementation, RecordReference


def test_resolved_input_and_output_bindings_form_a_valid_dag() -> None:
    source = _artifact("source", "a")
    result = _artifact("result", "b")
    definition = ComputationDefinition(
        id="urn:example:definition:transform",
        implementation=Implementation(
            kind="other",
            locator="example:transform",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    invocation = Invocation(
        id="urn:example:invocation:transform",
        definition=_reference(definition),
        inputs={"source": (_reference(source),)},
        outputs={"result": (_reference(result),)},
    )

    validate_derivation_graph((source, definition, invocation, result))


def test_invocation_may_publish_an_artifact_set_output() -> None:
    source = _artifact("source", "a")
    result = _artifact("result", "b")
    release = ArtifactSet(
        id="urn:example:artifact-set:release",
        members=(ArtifactSetMember(name="result", artifact=_reference(result)),),
    )
    definition = ComputationDefinition(
        id="urn:example:definition:package",
        implementation=Implementation(
            kind="other",
            locator="example:package",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    invocation = Invocation(
        id="urn:example:invocation:package",
        definition=_reference(definition),
        inputs={"source": (_reference(source),)},
        outputs={
            "result": (_reference(result),),
            "release": (_reference(release),),
        },
    )

    validate_derivation_graph((source, definition, invocation, result, release))


def test_invocation_may_consume_an_artifact_set_input() -> None:
    member = _artifact("model-state", "a")
    package = ArtifactSet(
        id="urn:example:artifact-set:model-package",
        members=(ArtifactSetMember(name="model.joblib", artifact=_reference(member)),),
    )
    result = _artifact("result", "b")
    definition = ComputationDefinition(
        id="urn:example:definition:score",
        implementation=Implementation(
            kind="other",
            locator="example:score",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    invocation = Invocation(
        id="urn:example:invocation:score",
        definition=_reference(definition),
        inputs={"model_package": (_reference(package),)},
        outputs={"result": (_reference(result),)},
    )

    validate_derivation_graph((member, package, definition, invocation, result))


def test_invocation_outputs_must_be_content_bound() -> None:
    with pytest.raises(ValidationError, match="outputs must include record digests"):
        Invocation(
            id="urn:example:invocation:transform",
            definition=RecordReference(id="urn:example:definition:transform"),
            outputs={"result": (RecordReference(id="urn:example:artifact:result"),)},
        )


def test_derivation_validator_rejects_redundant_implementation_digest() -> None:
    source = _artifact("source", "a")
    definition = ComputationDefinition(
        id="urn:example:definition:transform",
        implementation=Implementation(
            kind="other",
            locator="example:transform",
            digest=source.digest,
            artifact=_reference(source),
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )

    with pytest.raises(
        DerivationValidationError, match="must omit implementation.digest"
    ):
        validate_derivation_graph((source, definition))


def test_derivation_validator_rejects_unresolved_inputs() -> None:
    definition = ComputationDefinition(
        id="urn:example:definition:transform",
        implementation=Implementation(
            kind="other",
            locator="example:transform",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    invocation = Invocation(
        id="urn:example:invocation:transform",
        definition=_reference(definition),
        inputs={
            "source": (
                RecordReference(
                    id="urn:example:artifact:missing",
                    digest=Digest(value="c" * 64),
                ),
            )
        },
    )

    with pytest.raises(DerivationValidationError, match="does not resolve"):
        validate_derivation_graph((definition, invocation))


def test_derivation_validator_rejects_a_cycle() -> None:
    source = _artifact("source", "a")
    definition = ComputationDefinition(
        id="urn:example:definition:identity",
        implementation=Implementation(
            kind="other",
            locator="example:identity",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    invocation = Invocation(
        id="urn:example:invocation:identity",
        definition=_reference(definition),
        inputs={"value": (_reference(source),)},
        outputs={"value": (_reference(source),)},
    )

    with pytest.raises(DerivationValidationError, match="contains a cycle"):
        validate_derivation_graph((source, definition, invocation))


def test_invocation_hierarchy_resolves_an_id_only_parent() -> None:
    definition = ComputationDefinition(
        id="urn:example:definition:flow",
        implementation=Implementation(
            kind="other",
            locator="example:flow",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    parent = Invocation(
        id="urn:example:invocation:parent-run",
        definition=_reference(definition),
    )
    child = Invocation(
        id="urn:example:invocation:child-task",
        definition=_reference(definition),
        parent_invocation=RecordReference(id=parent.id),
    )

    validate_invocation_hierarchy((definition, parent, child))


def test_invocation_hierarchy_rejects_a_cycle() -> None:
    definition = ComputationDefinition(
        id="urn:example:definition:flow",
        implementation=Implementation(
            kind="other",
            locator="example:flow",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )
    first = Invocation(
        id="urn:example:invocation:first",
        definition=_reference(definition),
        parent_invocation=RecordReference(id="urn:example:invocation:second"),
    )
    second = Invocation(
        id="urn:example:invocation:second",
        definition=_reference(definition),
        parent_invocation=RecordReference(id=first.id),
    )

    with pytest.raises(OrchestrationValidationError, match="contains a cycle"):
        validate_invocation_hierarchy((definition, first, second))


def _artifact(name: str, digest_character: str) -> Artifact:
    return Artifact(
        id=f"urn:example:artifact:{name}",
        media_type="application/octet-stream",
        digest=Digest(value=digest_character * 64),
        size=1,
    )


def _reference(record: object) -> RecordReference:
    return RecordReference(id=record.id, digest=record_digest(record))

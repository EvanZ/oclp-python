"""Tests for the resolved OCLP derivation-DAG and Execution hierarchy."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest

from oclp import (
    Artifact,
    ArtifactSet,
    ArtifactSetMember,
    Computation,
    DerivationValidationError,
    Execution,
    OrchestrationValidationError,
    ParameterValidationError,
    validate_derivation_graph,
    validate_execution_hierarchy,
)
from oclp.models import (
    Digest,
    Implementation,
    ParameterDefinition,
    RecordReference,
)


def _computation(name: str) -> Computation:
    return Computation(
        id=_id(f"computation:{name}"),
        implementation=Implementation(
            kind="other",
            locator=f"example:{name}",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )


def test_resolved_input_and_output_bindings_form_a_valid_dag() -> None:
    source = _artifact("source", "a")
    result = _artifact("result", "b")
    computation = _computation("transform")
    execution = Execution(
        id=_id("execution:transform"),
        computation=_reference(computation),
        inputs={"source": (_reference(source),)},
        outputs={"result": (_reference(result),)},
    )

    validate_derivation_graph((source, computation, execution, result))


def test_execution_may_publish_an_artifact_set_output() -> None:
    source = _artifact("source", "a")
    result = _artifact("result", "b")
    release = ArtifactSet(
        id=_id("artifact-set:release"),
        members=(ArtifactSetMember(name="result", artifact=_reference(result)),),
    )
    computation = _computation("package")
    execution = Execution(
        id=_id("execution:package"),
        computation=_reference(computation),
        inputs={"source": (_reference(source),)},
        outputs={"result": (_reference(result),), "release": (_reference(release),)},
    )

    validate_derivation_graph((source, computation, execution, result, release))


def test_execution_may_consume_an_artifact_set_input() -> None:
    member = _artifact("model-state", "a")
    package = ArtifactSet(
        id=_id("artifact-set:model-package"),
        members=(ArtifactSetMember(name="model.joblib", artifact=_reference(member)),),
    )
    result = _artifact("result", "b")
    computation = _computation("score")
    execution = Execution(
        id=_id("execution:score"),
        computation=_reference(computation),
        inputs={"model_package": (_reference(package),)},
        outputs={"result": (_reference(result),)},
    )

    validate_derivation_graph((member, package, computation, execution, result))


def test_execution_outputs_reference_artifacts_by_uuid_only() -> None:
    execution = Execution(
        id=_id("execution:transform"),
        computation=RecordReference(id=_id("computation:transform")),
        outputs={"result": (RecordReference(id=_id("artifact:result")),)},
    )

    assert execution.outputs == {
        "result": (RecordReference(id=_id("artifact:result")),)
    }


def test_derivation_validator_rejects_redundant_implementation_digest() -> None:
    source = _artifact("source", "a")
    computation = Computation(
        id=_id("computation:transform"),
        implementation=Implementation(
            kind="other",
            locator="example:transform",
            digest=source.digest,
            artifact=_reference(source),
            source={"kind": "opaque", "reason": "test fixture"},
        ),
    )

    with pytest.raises(
        DerivationValidationError,
        match="must omit implementation.digest",
    ):
        validate_derivation_graph((source, computation))


def test_derivation_validator_rejects_unresolved_inputs() -> None:
    computation = _computation("transform")
    execution = Execution(
        id=_id("execution:transform"),
        computation=_reference(computation),
        inputs={
            "source": (
                RecordReference(
                    id=_id("artifact:missing")
                ),
            )
        },
    )

    with pytest.raises(DerivationValidationError, match="does not resolve"):
        validate_derivation_graph((computation, execution))


def test_derivation_validator_rejects_a_cycle() -> None:
    source = _artifact("source", "a")
    computation = _computation("identity")
    execution = Execution(
        id=_id("execution:identity"),
        computation=_reference(computation),
        inputs={"value": (_reference(source),)},
        outputs={"value": (_reference(source),)},
    )

    with pytest.raises(DerivationValidationError, match="contains a cycle"):
        validate_derivation_graph((source, computation, execution))


def test_derivation_validator_enforces_declared_execution_parameters() -> None:
    computation = Computation(
        id=_id("computation:parameterized"),
        implementation=Implementation(
            kind="other",
            locator="example:parameterized",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
        parameter_definitions=(
            ParameterDefinition(name="fold_number", schema={"type": "integer"}),
            ParameterDefinition(
                name="mode",
                schema={"enum": ["fast", "full"]},
                required=False,
            ),
        ),
    )
    valid = Execution(
        id=_id("execution:parameterized-valid"),
        computation=_reference(computation),
        parameters={"fold_number": 2, "mode": "fast"},
    )
    invalid = Execution(
        id=_id("execution:parameterized-invalid"),
        computation=_reference(computation),
        parameters={"fold_number": "two"},
    )

    validate_derivation_graph((computation, valid))
    with pytest.raises(ParameterValidationError, match="does not satisfy"):
        validate_derivation_graph((computation, invalid))


def test_execution_hierarchy_resolves_an_id_only_parent() -> None:
    computation = _computation("flow")
    parent = Execution(
        id=_id("execution:parent-run"), computation=_reference(computation)
    )
    child = Execution(
        id=_id("execution:child-task"),
        computation=_reference(computation),
        parent_execution=RecordReference(id=parent.id),
    )

    validate_execution_hierarchy((computation, parent, child))


def test_execution_hierarchy_rejects_a_cycle() -> None:
    computation = _computation("flow")
    first = Execution(
        id=_id("execution:first"),
        computation=_reference(computation),
        parent_execution=RecordReference(id=_id("execution:second")),
    )
    second = Execution(
        id=_id("execution:second"),
        computation=_reference(computation),
        parent_execution=RecordReference(id=first.id),
    )

    with pytest.raises(OrchestrationValidationError, match="contains a cycle"):
        validate_execution_hierarchy((computation, first, second))


def _artifact(name: str, digest_character: str) -> Artifact:
    return Artifact(
        id=_id(f"artifact:{name}"),
        media_type="application/octet-stream",
        digest=Digest(value=digest_character * 64),
        size=1,
    )


def _reference(record: object) -> RecordReference:
    return RecordReference(id=record.id)


def _id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"test:derivation:{name}"))

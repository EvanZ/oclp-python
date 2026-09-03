"""Parsing and validation helpers for OCLP records."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from oclp.canonical import canonical_json_bytes, record_digest
from oclp.models import (
    OCLP_RECORD_ADAPTER,
    Computation,
    Event,
    Evidence,
    Execution,
    GitSource,
    OclpRecord,
    RecordReference,
)


class DerivationValidationError(ValueError):
    """A resolved OCLP record set violates a derivation or implementation rule."""


class OrchestrationValidationError(ValueError):
    """A resolved OCLP record set violates the Execution hierarchy contract."""


class AcceptanceValidationError(ValueError):
    """A successful Execution lacks its Computation's required Evidence."""


class ParameterValidationError(ValueError):
    """An Execution does not satisfy its Computation parameter contract."""


def parse_record(value: Any) -> OclpRecord:
    return OCLP_RECORD_ADAPTER.validate_python(value)


def load_record(path: str | Path) -> OclpRecord:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_record(json.load(handle))


def validate_derivation_graph(records: Iterable[OclpRecord]) -> None:
    """Validate the resolved Artifact/ArtifactSet -> Execution -> output DAG.

    Record-level parsing permits unbound input references because an individual
    record cannot resolve them. A collection offered as a derivation graph must
    bind and resolve every Computation, input Artifact or ArtifactSet, and
    output Artifact or ArtifactSet reference used by an Execution.
    """

    by_digest = {record_digest(record).value: record for record in records}
    adjacency: dict[str, set[str]] = defaultdict(set)

    for record in by_digest.values():
        if record.kind != "computation":
            continue
        if record.implementation.artifact is not None:
            artifact_digest = _require_reference(
                record.implementation.artifact,
                by_digest,
                expected_kind="artifact",
                label=f"Computation {record.id} implementation artifact",
            )
            artifact = by_digest[artifact_digest]
            if (
                record.implementation.digest is not None
                and artifact.digest == record.implementation.digest
            ):
                raise DerivationValidationError(
                    f"Computation {record.id} must omit implementation.digest when it "
                    "duplicates its implementation Artifact content digest"
                )
        source = record.implementation.source
        if isinstance(source, GitSource) and source.overlay is not None:
            _require_reference(
                source.overlay,
                by_digest,
                expected_kind="artifact_set",
                label=f"Computation {record.id} Git source overlay",
            )

    for execution_digest, record in by_digest.items():
        if record.kind != "execution":
            continue
        computation_digest = _require_reference(
            record.computation,
            by_digest,
            expected_kind="computation",
            label=f"Execution {record.id} computation",
        )
        computation = by_digest[computation_digest]
        assert isinstance(computation, Computation)
        _validate_execution_parameters(record, computation)
        for port, references in record.inputs.items():
            for reference in references:
                input_digest = _require_reference(
                    reference,
                    by_digest,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Execution {record.id} input {port!r}",
                )
                adjacency[input_digest].add(execution_digest)
        for port, references in (record.outputs or {}).items():
            for reference in references:
                output_digest = _require_reference(
                    reference,
                    by_digest,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Execution {record.id} output {port!r}",
                )
                adjacency[execution_digest].add(output_digest)

    _raise_on_cycle(adjacency)


def _validate_execution_parameters(
    execution: Execution, computation: Computation
) -> None:
    """Validate concrete JSON bindings against the selected Computation."""

    definitions = {
        definition.name: definition for definition in computation.parameter_definitions
    }
    unexpected = sorted(set(execution.parameters).difference(definitions))
    if unexpected:
        raise ParameterValidationError(
            f"Execution {execution.id} has undeclared parameters: "
            f"{', '.join(unexpected)}"
        )
    missing = sorted(
        definition.name
        for definition in computation.parameter_definitions
        if definition.required and definition.name not in execution.parameters
    )
    if missing:
        raise ParameterValidationError(
            f"Execution {execution.id} is missing required parameters: "
            f"{', '.join(missing)}"
        )
    for name, value in execution.parameters.items():
        definition = definitions[name]
        try:
            Draft202012Validator.check_schema(definition.schema)
        except SchemaError as error:
            raise ParameterValidationError(
                f"Computation {computation.id} has invalid JSON Schema for parameter "
                f"{name!r}: {error.message}"
            ) from error
        errors = tuple(Draft202012Validator(definition.schema).iter_errors(value))
        if errors:
            raise ParameterValidationError(
                f"Execution {execution.id} parameter {name!r} does not satisfy "
                f"its declared JSON Schema: {errors[0].message}"
            )


def validate_execution_hierarchy(records: Iterable[OclpRecord]) -> None:
    """Validate resolved parent-child Execution relationships independently.

    A parent reference is an execution relationship, not a data-derivation
    binding. ID-only references are deliberately supported so a parent may
    publish an output manifest that content-binds child Executions without an
    immutable record-digest cycle.
    """

    by_digest = {record_digest(record).value: record for record in records}
    executions_by_id: dict[str, list[tuple[str, OclpRecord]]] = defaultdict(list)
    for digest, record in by_digest.items():
        if record.kind == "execution":
            executions_by_id[record.id].append((digest, record))

    adjacency: dict[str, set[str]] = defaultdict(set)
    for child_digest, record in by_digest.items():
        if record.kind != "execution" or record.parent_execution is None:
            continue
        parent_digest = _resolve_execution_parent(
            record.parent_execution,
            by_digest,
            executions_by_id,
            label=f"Execution {record.id} parent_execution",
        )
        adjacency[parent_digest].add(child_digest)

    _raise_on_orchestration_cycle(adjacency)


def validate_execution_acceptance(records: Iterable[OclpRecord]) -> None:
    """Validate Evidence requirements for Executions claimed as successful.

    A Computation may declare ``required_evidence`` evaluator bindings. The
    requirement takes effect only once an Event claims its Execution
    ``status="succeeded"``.
    Each exact evaluator must then have at least one Evidence record that
    content-binds that exact Execution and reports ``outcome="pass"``.

    This validator verifies durable record relationships and stated outcomes;
    it does not evaluate the producer-defined evaluator itself.
    """

    by_digest = {record_digest(record).value: record for record in records}
    executions_by_id: dict[str, list[tuple[str, Execution]]] = defaultdict(list)
    for digest, record in by_digest.items():
        if isinstance(record, Execution):
            executions_by_id[record.id].append((digest, record))

    evidence_records = tuple(
        record for record in by_digest.values() if isinstance(record, Evidence)
    )
    successful_execution_digests = _successful_execution_digests(
        by_digest,
        executions_by_id,
    )

    for execution_digest in successful_execution_digests:
        execution = by_digest[execution_digest]
        assert isinstance(execution, Execution)
        computation_digest = _require_reference(
            execution.computation,
            by_digest,
            expected_kind="computation",
            label=f"Execution {execution.id} computation",
        )
        computation = by_digest[computation_digest]
        assert isinstance(computation, Computation)

        for evaluator in computation.required_evidence or ():
            has_passing_evidence = any(
                _evidence_satisfies_requirement(
                    evidence,
                    execution=execution,
                    execution_digest=execution_digest,
                    evaluator=evaluator,
                )
                for evidence in evidence_records
            )
            if not has_passing_evidence:
                raise AcceptanceValidationError(
                    f"Execution {execution.id} is claimed succeeded but lacks "
                    f"passing Evidence for evaluator {evaluator.locator}"
                )


def _require_reference(
    reference: RecordReference,
    records: dict[str, OclpRecord],
    *,
    expected_kind: str | tuple[str, ...],
    label: str,
) -> str:
    if reference.digest is None:
        raise DerivationValidationError(f"{label} must include a record digest")
    digest = reference.digest.value
    target = records.get(digest)
    if target is None:
        raise DerivationValidationError(f"{label} does not resolve in this record set")
    expected_kinds = (
        (expected_kind,) if isinstance(expected_kind, str) else expected_kind
    )
    if target.kind not in expected_kinds:
        expected_label = " or ".join(expected_kinds)
        raise DerivationValidationError(
            f"{label} must resolve to a {expected_label}, got {target.kind}"
        )
    if target.id != reference.id:
        raise DerivationValidationError(
            f"{label} ID does not match its resolved record"
        )
    return digest


def _successful_execution_digests(
    records: dict[str, OclpRecord],
    executions_by_id: dict[str, list[tuple[str, Execution]]],
) -> set[str]:
    successful: set[str] = set()
    for event in records.values():
        if not isinstance(event, Event) or event.status != "succeeded":
            continue
        reference = event.execution
        if reference.digest is not None:
            target = records.get(reference.digest.value)
            if target is None or not isinstance(target, Execution):
                raise AcceptanceValidationError(
                    f"successful Event {event.id} execution does not resolve to an "
                    "Execution"
                )
            if target.id != reference.id:
                raise AcceptanceValidationError(
                    f"successful Event {event.id} execution ID does not match its "
                    "resolved record"
                )
            successful.add(reference.digest.value)
            continue

        matches = executions_by_id.get(reference.id, [])
        if len(matches) != 1:
            raise AcceptanceValidationError(
                f"successful Event {event.id} execution is ambiguous without a "
                "record digest"
            )
        successful.add(matches[0][0])
    return successful


def _evidence_satisfies_requirement(
    evidence: Evidence,
    *,
    execution: Execution,
    execution_digest: str,
    evaluator: object,
) -> bool:
    return (
        evidence.subject.id == execution.id
        and evidence.subject.digest is not None
        and evidence.subject.digest.value == execution_digest
        and canonical_json_bytes(evidence.evaluator) == canonical_json_bytes(evaluator)
        and evidence.outcome == "pass"
    )


def _raise_on_cycle(adjacency: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise DerivationValidationError("derivation graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for neighbor in adjacency.get(node, ()):
            visit(neighbor)
        visiting.remove(node)
        visited.add(node)

    for node in tuple(adjacency):
        visit(node)


def _resolve_execution_parent(
    reference: RecordReference,
    by_digest: dict[str, OclpRecord],
    executions_by_id: dict[str, list[tuple[str, OclpRecord]]],
    *,
    label: str,
) -> str:
    if reference.digest is not None:
        target = by_digest.get(reference.digest.value)
        if target is None:
            raise OrchestrationValidationError(
                f"{label} does not resolve in this record set"
            )
        if target.kind != "execution":
            raise OrchestrationValidationError(
                f"{label} must resolve to an execution, got {target.kind}"
            )
        if target.id != reference.id:
            raise OrchestrationValidationError(
                f"{label} ID does not match its resolved record"
            )
        return reference.digest.value

    matches = executions_by_id.get(reference.id, [])
    if not matches:
        raise OrchestrationValidationError(
            f"{label} does not resolve in this record set"
        )
    if len(matches) != 1:
        raise OrchestrationValidationError(
            f"{label} is ambiguous without a record digest"
        )
    return matches[0][0]


def _raise_on_orchestration_cycle(adjacency: dict[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise OrchestrationValidationError("execution hierarchy contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in tuple(adjacency):
        visit(node)

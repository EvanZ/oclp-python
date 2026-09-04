"""Parsing and validation helpers for OCLP records."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from oclp.canonical import canonical_json_bytes
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

    by_id = _records_by_id(records)
    adjacency: dict[str, set[str]] = defaultdict(set)

    for record in by_id.values():
        if record.kind != "computation":
            continue
        if record.implementation.artifact is not None:
            artifact_id = _require_reference(
                record.implementation.artifact,
                by_id,
                expected_kind="artifact",
                label=f"Computation {record.id} implementation artifact",
            )
            artifact = by_id[artifact_id]
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
                by_id,
                expected_kind="artifact_set",
                label=f"Computation {record.id} Git source overlay",
            )

    for execution_id, record in by_id.items():
        if record.kind != "execution":
            continue
        computation_id = _require_reference(
            record.computation,
            by_id,
            expected_kind="computation",
            label=f"Execution {record.id} computation",
        )
        computation = by_id[computation_id]
        assert isinstance(computation, Computation)
        _validate_execution_parameters(record, computation)
        for port, references in record.inputs.items():
            for reference in references:
                input_id = _require_reference(
                    reference,
                    by_id,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Execution {record.id} input {port!r}",
                )
                adjacency[input_id].add(execution_id)
        for port, references in (record.outputs or {}).items():
            for reference in references:
                output_id = _require_reference(
                    reference,
                    by_id,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Execution {record.id} output {port!r}",
                )
                adjacency[execution_id].add(output_id)

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

    by_id = _records_by_id(records)

    adjacency: dict[str, set[str]] = defaultdict(set)
    for child_id, record in by_id.items():
        if record.kind != "execution" or record.parent_execution is None:
            continue
        parent_id = _resolve_execution_parent(
            record.parent_execution,
            by_id,
            label=f"Execution {record.id} parent_execution",
        )
        adjacency[parent_id].add(child_id)

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

    by_id = _records_by_id(records)

    evidence_records = tuple(
        record for record in by_id.values() if isinstance(record, Evidence)
    )
    successful_execution_ids = _successful_execution_ids(by_id)

    for execution_id in successful_execution_ids:
        execution = by_id[execution_id]
        assert isinstance(execution, Execution)
        computation_id = _require_reference(
            execution.computation,
            by_id,
            expected_kind="computation",
            label=f"Execution {execution.id} computation",
        )
        computation = by_id[computation_id]
        assert isinstance(computation, Computation)

        for evaluator in computation.required_evidence or ():
            has_passing_evidence = any(
                _evidence_satisfies_requirement(
                    evidence,
                    execution=execution,
                    execution_id=execution_id,
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
    target = records.get(reference.id)
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
    return reference.id


def _successful_execution_ids(
    records: dict[str, OclpRecord],
) -> set[str]:
    successful: set[str] = set()
    for event in records.values():
        if not isinstance(event, Event) or event.status != "succeeded":
            continue
        reference = event.execution
        target = records.get(reference.id)
        if target is None or not isinstance(target, Execution):
            raise AcceptanceValidationError(
                f"successful Event {event.id} execution does not resolve to an "
                "Execution"
            )
        successful.add(reference.id)
    return successful


def _evidence_satisfies_requirement(
    evidence: Evidence,
    *,
    execution: Execution,
    execution_id: str,
    evaluator: object,
) -> bool:
    return (
        evidence.subject.id == execution_id == execution.id
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
    by_id: dict[str, OclpRecord],
    *,
    label: str,
) -> str:
    target = by_id.get(reference.id)
    if target is None:
        raise OrchestrationValidationError(
            f"{label} does not resolve in this record set"
        )
    if target.kind != "execution":
        raise OrchestrationValidationError(
            f"{label} must resolve to an execution, got {target.kind}"
        )
    return reference.id


def _records_by_id(records: Iterable[OclpRecord]) -> dict[str, OclpRecord]:
    """Resolve a record collection by protocol identity and reject revisions."""

    by_id: dict[str, OclpRecord] = {}
    for record in records:
        existing = by_id.get(record.id)
        if (
            existing is not None
            and canonical_json_bytes(existing) != canonical_json_bytes(record)
        ):
            raise DerivationValidationError(
                f"record ID {record.id} identifies more than one immutable record"
            )
        by_id[record.id] = record
    return by_id


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

"""Parsing and validation helpers for OCLP records."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oclp.canonical import record_digest
from oclp.models import OCLP_RECORD_ADAPTER, GitSource, OclpRecord, RecordReference


class DerivationValidationError(ValueError):
    """A resolved OCLP record set violates a derivation or implementation rule."""


class OrchestrationValidationError(ValueError):
    """A resolved OCLP record set violates the Invocation hierarchy contract."""


def parse_record(value: Any) -> OclpRecord:
    return OCLP_RECORD_ADAPTER.validate_python(value)


def load_record(path: str | Path) -> OclpRecord:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_record(json.load(handle))


def validate_derivation_graph(records: Iterable[OclpRecord]) -> None:
    """Validate the resolved Artifact/ArtifactSet -> Invocation -> output DAG.

    Record-level parsing permits unbound input references because an individual
    record cannot resolve them. A collection offered as a derivation graph must
    bind and resolve every Definition, input Artifact or ArtifactSet, and
    output Artifact or ArtifactSet reference used by an Invocation.
    """

    by_digest = {record_digest(record).value: record for record in records}
    adjacency: dict[str, set[str]] = defaultdict(set)

    for record in by_digest.values():
        if record.kind != "definition":
            continue
        if record.implementation.artifact is not None:
            artifact_digest = _require_reference(
                record.implementation.artifact,
                by_digest,
                expected_kind="artifact",
                label=f"Definition {record.id} implementation artifact",
            )
            artifact = by_digest[artifact_digest]
            if (
                record.implementation.digest is not None
                and artifact.digest == record.implementation.digest
            ):
                raise DerivationValidationError(
                    f"Definition {record.id} must omit implementation.digest when it "
                    "duplicates its implementation Artifact content digest"
                )
        source = record.implementation.source
        if isinstance(source, GitSource) and source.overlay is not None:
            _require_reference(
                source.overlay,
                by_digest,
                expected_kind="artifact_set",
                label=f"Definition {record.id} Git source overlay",
            )

    for invocation_digest, record in by_digest.items():
        if record.kind != "invocation":
            continue
        _require_reference(
            record.definition,
            by_digest,
            expected_kind="definition",
            label=f"Invocation {record.id} definition",
        )
        for port, references in record.inputs.items():
            for reference in references:
                input_digest = _require_reference(
                    reference,
                    by_digest,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Invocation {record.id} input {port!r}",
                )
                adjacency[input_digest].add(invocation_digest)
        for port, references in (record.outputs or {}).items():
            for reference in references:
                output_digest = _require_reference(
                    reference,
                    by_digest,
                    expected_kind=("artifact", "artifact_set"),
                    label=f"Invocation {record.id} output {port!r}",
                )
                adjacency[invocation_digest].add(output_digest)

    _raise_on_cycle(adjacency)


def validate_invocation_hierarchy(records: Iterable[OclpRecord]) -> None:
    """Validate resolved parent-child Invocation relationships independently.

    A parent reference is an execution relationship, not a data-derivation
    binding. ID-only references are deliberately supported so a parent may
    publish an output manifest that content-binds child Invocations without an
    immutable record-digest cycle.
    """

    by_digest = {record_digest(record).value: record for record in records}
    invocations_by_id: dict[str, list[tuple[str, OclpRecord]]] = defaultdict(list)
    for digest, record in by_digest.items():
        if record.kind == "invocation":
            invocations_by_id[record.id].append((digest, record))

    adjacency: dict[str, set[str]] = defaultdict(set)
    for child_digest, record in by_digest.items():
        if record.kind != "invocation" or record.parent_invocation is None:
            continue
        parent_digest = _resolve_invocation_parent(
            record.parent_invocation,
            by_digest,
            invocations_by_id,
            label=f"Invocation {record.id} parent_invocation",
        )
        adjacency[parent_digest].add(child_digest)

    _raise_on_orchestration_cycle(adjacency)


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


def _resolve_invocation_parent(
    reference: RecordReference,
    by_digest: dict[str, OclpRecord],
    invocations_by_id: dict[str, list[tuple[str, OclpRecord]]],
    *,
    label: str,
) -> str:
    if reference.digest is not None:
        target = by_digest.get(reference.digest.value)
        if target is None:
            raise OrchestrationValidationError(
                f"{label} does not resolve in this record set"
            )
        if target.kind != "invocation":
            raise OrchestrationValidationError(
                f"{label} must resolve to an invocation, got {target.kind}"
            )
        if target.id != reference.id:
            raise OrchestrationValidationError(
                f"{label} ID does not match its resolved record"
            )
        return reference.digest.value

    matches = invocations_by_id.get(reference.id, [])
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
            raise OrchestrationValidationError("invocation hierarchy contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in tuple(adjacency):
        visit(node)

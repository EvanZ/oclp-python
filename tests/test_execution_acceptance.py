"""Tests for Computation-declared Evidence required for execution success."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest

from oclp import (
    AcceptanceValidationError,
    Computation,
    Event,
    Evidence,
    Execution,
    record_digest,
    validate_execution_acceptance,
)
from oclp.models import Implementation, RecordReference


def test_successful_execution_requires_its_declared_passing_evidence() -> None:
    evaluator = _evaluator()
    computation = _computation(evaluator=evaluator)
    execution = _execution(computation)
    terminal = _succeeded_terminal(execution)
    evidence = Evidence(
        id="urn:example:evidence:quality",
        subject=_reference(execution),
        evaluator=evaluator,
        outcome="pass",
        observed_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
    )

    validate_execution_acceptance((computation, execution, evidence, terminal))


@pytest.mark.parametrize("outcome", (None, "fail", "error"))
def test_successful_execution_rejects_missing_or_nonpassing_evidence(
    outcome: Literal["fail", "error"] | None,
) -> None:
    evaluator = _evaluator()
    computation = _computation(evaluator=evaluator)
    execution = _execution(computation)
    terminal = _succeeded_terminal(execution)
    records: tuple[object, ...] = (computation, execution, terminal)
    if outcome is not None:
        records = (
            computation,
            execution,
            Evidence(
                id=f"urn:example:evidence:{outcome}",
                subject=_reference(execution),
                evaluator=evaluator,
                outcome=outcome,
                observed_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
            ),
            terminal,
        )

    with pytest.raises(AcceptanceValidationError, match="lacks passing Evidence"):
        validate_execution_acceptance(records)  # type: ignore[arg-type]


def test_failed_execution_does_not_need_passing_evidence() -> None:
    computation = _computation(evaluator=_evaluator())
    execution = _execution(computation)
    terminal = Event(
        id="urn:example:event:execution:terminal",
        execution=_reference(execution),
        event_type="execution-terminal",
        occurred_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
        sequence=1,
        status="failed",
    )

    validate_execution_acceptance((computation, execution, terminal))


def test_successful_execution_without_declared_evidence_is_valid() -> None:
    computation = _computation(evaluator=None)
    execution = _execution(computation)

    validate_execution_acceptance(
        (computation, execution, _succeeded_terminal(execution))
    )


def test_required_evidence_must_bind_the_exact_execution_revision() -> None:
    evaluator = _evaluator()
    computation = _computation(evaluator=evaluator)
    execution = _execution(computation)
    terminal = _succeeded_terminal(execution)
    id_only_evidence = Evidence(
        id="urn:example:evidence:id-only",
        subject=RecordReference(id=execution.id),
        evaluator=evaluator,
        outcome="pass",
        observed_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
    )

    with pytest.raises(AcceptanceValidationError, match="lacks passing Evidence"):
        validate_execution_acceptance(
            (computation, execution, id_only_evidence, terminal)
        )


def test_required_evidence_must_bind_the_exact_evaluator() -> None:
    computation = _computation(evaluator=_evaluator())
    execution = _execution(computation)
    terminal = _succeeded_terminal(execution)
    evidence = Evidence(
        id="urn:example:evidence:wrong-evaluator",
        subject=_reference(execution),
        evaluator=Implementation(
            kind="other",
            locator="example:another-quality-check",
            source={"kind": "opaque", "reason": "test fixture"},
        ),
        outcome="pass",
        observed_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
    )

    with pytest.raises(AcceptanceValidationError, match="lacks passing Evidence"):
        validate_execution_acceptance((computation, execution, evidence, terminal))


def _evaluator() -> Implementation:
    return Implementation(
        kind="other",
        locator="example:quality",
        source={"kind": "opaque", "reason": "test fixture"},
    )


def _computation(*, evaluator: Implementation | None) -> Computation:
    return Computation(
        id="urn:example:computation:quality-checked",
        implementation=_implementation(),
        required_evidence=(evaluator,) if evaluator is not None else None,
    )


def _implementation() -> Implementation:
    return Implementation(
        kind="other",
        locator="example:quality-checked",
        source={"kind": "opaque", "reason": "test fixture"},
    )


def _execution(computation: Computation) -> Execution:
    return Execution(
        id="urn:example:execution:quality-checked:run-1",
        computation=_reference(computation),
    )


def _succeeded_terminal(execution: Execution) -> Event:
    return Event(
        id="urn:example:event:execution:terminal",
        execution=_reference(execution),
        event_type="execution-terminal",
        occurred_at=datetime(2026, 8, 30, 18, 0, tzinfo=UTC),
        sequence=1,
        status="succeeded",
    )


def _reference(record: object) -> RecordReference:
    return RecordReference(id=record.id, digest=record_digest(record))  # type: ignore[attr-defined]

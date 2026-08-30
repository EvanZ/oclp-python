"""Explicit lifecycle publication helpers for completed demo Invocations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from oclp import Evidence, ExecutionContext, Invocation, LifecycleEvent
from oclp.models import RecordReference

from bike_demand_service.publication import LocalPublisher

LIFECYCLE_PROFILE = {"lifecycle": {"version": "0.1.0-draft"}}


def publish_started(
    *,
    publisher: LocalPublisher,
    invocation: Invocation,
    attempt_id: str,
    started_at: datetime,
    execution: ExecutionContext | None = None,
) -> RecordReference:
    """Publish an Invocation and the lifecycle facts known when it starts."""

    invocation_ref = publisher.publish(invocation)
    event_prefix = invocation.id.replace("urn:oclp-bike-demand:invocation:", "")
    publisher.publish(
        LifecycleEvent(
            id=f"urn:oclp-bike-demand:event:{event_prefix}:requested",
            invocation=invocation_ref,
            event_type="invocation-requested",
            occurred_at=started_at,
            sequence=0,
        )
    )
    publisher.publish(
        LifecycleEvent(
            id=f"urn:oclp-bike-demand:event:{event_prefix}:started",
            invocation=invocation_ref,
            event_type="attempt-started",
            occurred_at=started_at,
            sequence=1,
            attempt_id=attempt_id,
            execution=execution,
        )
    )
    if invocation.outputs:
        publisher.publish(
            LifecycleEvent(
                id=f"urn:oclp-bike-demand:event:{event_prefix}:artifacts-published",
                invocation=invocation_ref,
                event_type="artifacts-published",
                occurred_at=started_at,
                sequence=2,
                attempt_id=attempt_id,
                data={"outputs": _references_json(invocation.outputs)},
            )
        )
    return invocation_ref


def publish_terminal(
    *,
    publisher: LocalPublisher,
    invocation: Invocation,
    invocation_ref: RecordReference,
    attempt_id: str,
    completed_at: datetime,
    sequence: int,
) -> None:
    """Publish the terminal lifecycle observation for an already-started run."""

    event_prefix = invocation.id.replace("urn:oclp-bike-demand:invocation:", "")
    publisher.publish(
        LifecycleEvent(
            id=f"urn:oclp-bike-demand:event:{event_prefix}:terminal",
            invocation=invocation_ref,
            event_type="invocation-terminal",
            occurred_at=completed_at,
            sequence=sequence,
            attempt_id=attempt_id,
            status="succeeded",
        )
    )


def publish_completed(
    *,
    publisher: LocalPublisher,
    invocation: Invocation,
    attempt_id: str,
    started_at: datetime,
    completed_at: datetime,
    evidence: Evidence | None = None,
    execution: ExecutionContext | None = None,
) -> RecordReference:
    """Publish one observed successful Invocation with standard lifecycle facts."""

    invocation_ref = publish_started(
        publisher=publisher,
        invocation=invocation,
        attempt_id=attempt_id,
        started_at=started_at,
        execution=execution,
    )
    sequence = 3 if invocation.outputs else 2
    event_prefix = invocation.id.replace("urn:oclp-bike-demand:invocation:", "")
    if evidence is not None:
        evidence_ref = publisher.publish(evidence)
        publisher.publish(
            LifecycleEvent(
                id=f"urn:oclp-bike-demand:event:{event_prefix}:evidence-published",
                invocation=invocation_ref,
                event_type="evidence-published",
                occurred_at=completed_at,
                sequence=sequence,
                attempt_id=attempt_id,
                data={"evidence": evidence_ref.model_dump(mode="json")},
            )
        )
        sequence += 1
    publish_terminal(
        publisher=publisher,
        invocation=invocation,
        invocation_ref=invocation_ref,
        attempt_id=attempt_id,
        completed_at=completed_at,
        sequence=sequence,
    )
    return invocation_ref


def _references_json(
    references: dict[str, tuple[RecordReference, ...]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert immutable references into JSON-safe Event detail."""

    return {
        port: [reference.model_dump(mode="json") for reference in values]
        for port, values in references.items()
    }

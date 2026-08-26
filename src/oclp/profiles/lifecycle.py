"""Portable lifecycle chronology for OCLP Invocations."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from oclp.models import LifecycleEvent, OclpModel

LIFECYCLE_PROFILE = "lifecycle"
LIFECYCLE_PROFILE_VERSION = "0.1.0-draft"

INVOCATION_REQUESTED = "invocation-requested"
ATTEMPT_STARTED = "attempt-started"
INVOCATION_TERMINAL = "invocation-terminal"


class LifecycleBinding(OclpModel):
    """The value carried under an Invocation's ``profiles.lifecycle`` key."""

    version: Literal["0.1.0-draft"]


class LifecycleEventObservation(OclpModel):
    """The profile-relevant subset of an Invocation's Event."""

    event_type: str = Field(min_length=1)
    occurred_at: datetime
    sequence: int = Field(ge=0)
    attempt_id: str | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None


class LifecycleTimeline(OclpModel):
    """Portable timeline facts resolved from one lifecycle-profile claim."""

    requested_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None


class LifecycleTimelineVector(OclpModel):
    """A conformance-vector representation of Invocation-associated Events."""

    binding: LifecycleBinding
    events: tuple[LifecycleEventObservation, ...]

    @model_validator(mode="after")
    def has_a_valid_timeline(self) -> LifecycleTimelineVector:
        _timeline_from_observations(self.events)
        return self


def lifecycle_timeline(
    binding: Any,
    events: Iterable[LifecycleEvent],
) -> LifecycleTimeline:
    """Validate a profile claim and resolve portable chronology from its Events."""

    LifecycleBinding.model_validate(binding)
    observations = tuple(
        LifecycleEventObservation(
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            sequence=event.sequence,
            attempt_id=event.attempt_id,
            status=event.status,
        )
        for event in events
    )
    return _timeline_from_observations(observations)


def _timeline_from_observations(
    events: tuple[LifecycleEventObservation, ...],
) -> LifecycleTimeline:
    requested = [event for event in events if event.event_type == INVOCATION_REQUESTED]
    if len(requested) != 1:
        raise ValueError(
            "lifecycle profile requires exactly one invocation-requested Event"
        )
    request = requested[0]
    if request.sequence != 0:
        raise ValueError("lifecycle invocation-requested Event must have sequence 0")
    if request.attempt_id is not None or request.status is not None:
        raise ValueError(
            "lifecycle invocation-requested Event must omit attempt_id and status"
        )

    started = [event for event in events if event.event_type == ATTEMPT_STARTED]
    attempt_ids = [event.attempt_id for event in started]
    if any(attempt_id is None for attempt_id in attempt_ids):
        raise ValueError("lifecycle attempt-started Events must include attempt_id")
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError(
            "lifecycle attempt-started Events must use unique attempt_id values"
        )
    if any(event.status is not None for event in started):
        raise ValueError("lifecycle attempt-started Events must omit status")
    if any(event.sequence <= request.sequence for event in started):
        raise ValueError(
            "lifecycle attempt-started Events must follow invocation-requested"
        )

    terminal = [event for event in events if event.event_type == INVOCATION_TERMINAL]
    if len(terminal) > 1:
        raise ValueError(
            "lifecycle profile permits at most one invocation-terminal Event"
        )
    if terminal:
        completed = terminal[0]
        if completed.status is None:
            raise ValueError("lifecycle invocation-terminal Event must include status")
        if completed.sequence <= request.sequence:
            raise ValueError("lifecycle invocation-terminal Event must follow request")
        if started and completed.sequence <= max(event.sequence for event in started):
            raise ValueError(
                "lifecycle invocation-terminal Event must follow every "
                "attempt-started Event"
            )
        return LifecycleTimeline(
            requested_at=request.occurred_at,
            started_at=min((event.occurred_at for event in started), default=None),
            completed_at=completed.occurred_at,
            status=completed.status,
        )

    return LifecycleTimeline(
        requested_at=request.occurred_at,
        started_at=min((event.occurred_at for event in started), default=None),
    )

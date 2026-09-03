"""Portable lifecycle chronology for OCLP Executions."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from oclp.models import Event, OclpModel

LIFECYCLE_PROFILE = "lifecycle"
LIFECYCLE_PROFILE_VERSION = "0.2.0-draft"

EXECUTION_STARTED = "execution-started"
EXECUTION_TERMINAL = "execution-terminal"


class LifecycleBinding(OclpModel):
    """The value carried under an Execution's ``profiles.lifecycle`` key."""

    version: Literal["0.2.0-draft"]
    run_id: str | None = Field(default=None, min_length=1)
    run_name: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def names_only_an_identified_lifecycle_run(self) -> LifecycleBinding:
        if self.run_name is not None and self.run_id is None:
            raise ValueError("lifecycle run_name requires run_id")
        return self


class LifecycleObservation(OclpModel):
    """The profile-relevant subset of an Execution's Event."""

    event_type: str = Field(min_length=1)
    occurred_at: datetime
    sequence: int = Field(ge=0)
    status: Literal["succeeded", "failed", "skipped"] | None = None


class LifecycleTimeline(OclpModel):
    """Portable timeline facts resolved from one lifecycle-profile claim."""

    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None


class LifecycleTimelineVector(OclpModel):
    """A conformance-vector representation of Execution-associated Events."""

    binding: LifecycleBinding
    events: tuple[LifecycleObservation, ...]

    @model_validator(mode="after")
    def has_a_valid_timeline(self) -> LifecycleTimelineVector:
        _timeline_from_observations(self.events)
        return self


def lifecycle_timeline(
    binding: Any,
    events: Iterable[Event],
) -> LifecycleTimeline:
    """Validate a profile claim and resolve portable chronology from its Events."""

    LifecycleBinding.model_validate(binding)
    observations = tuple(
        LifecycleObservation(
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            sequence=event.sequence,
            status=event.status,
        )
        for event in events
    )
    return _timeline_from_observations(observations)


def _timeline_from_observations(
    events: tuple[LifecycleObservation, ...],
) -> LifecycleTimeline:
    started = [event for event in events if event.event_type == EXECUTION_STARTED]
    if len(started) != 1:
        raise ValueError(
            "lifecycle profile requires exactly one execution-started Event"
        )
    start = started[0]
    if start.sequence != 0:
        raise ValueError("lifecycle execution-started Event must have sequence 0")
    if start.status is not None:
        raise ValueError("lifecycle execution-started Event must omit status")

    terminal = [event for event in events if event.event_type == EXECUTION_TERMINAL]
    if len(terminal) > 1:
        raise ValueError(
            "lifecycle profile permits at most one execution-terminal Event"
        )
    if terminal:
        completed = terminal[0]
        if completed.status is None:
            raise ValueError("lifecycle execution-terminal Event must include status")
        if completed.sequence <= start.sequence:
            raise ValueError("lifecycle execution-terminal Event must follow start")
        return LifecycleTimeline(
            started_at=start.occurred_at,
            completed_at=completed.occurred_at,
            status=completed.status,
        )

    return LifecycleTimeline(started_at=start.occurred_at)

"""Portable run chronology and grouping for OCLP Executions."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from oclp.models import Event, OclpModel

RUN_PROFILE = "run"
RUN_PROFILE_VERSION = "0.3.0-draft"

EXECUTION_STARTED = "execution-started"
EXECUTION_TERMINAL = "execution-terminal"


class RunBinding(OclpModel):
    """The value carried under an Execution's ``profiles.run`` key."""

    version: Literal["0.3.0-draft"]
    run_id: UUID
    run_name: str = Field(min_length=1)


class RunObservation(OclpModel):
    """The profile-relevant subset of an Execution's Event."""

    event_type: str = Field(min_length=1)
    occurred_at: datetime
    sequence: int = Field(ge=0)
    status: Literal["succeeded", "failed", "skipped"] | None = None


class RunTimeline(OclpModel):
    """Portable timeline facts resolved from one run-profile claim."""

    started_at: datetime
    completed_at: datetime | None = None
    status: Literal["succeeded", "failed", "skipped"] | None = None


class RunTimelineVector(OclpModel):
    """A conformance-vector representation of Execution-associated Events."""

    binding: RunBinding
    events: tuple[RunObservation, ...]

    @model_validator(mode="after")
    def has_a_valid_timeline(self) -> RunTimelineVector:
        _timeline_from_observations(self.events)
        return self


def run_timeline(
    binding: Any,
    events: Iterable[Event],
) -> RunTimeline:
    """Validate a profile claim and resolve portable chronology from its Events."""

    RunBinding.model_validate(binding)
    observations = tuple(
        RunObservation(
            event_type=event.event_type,
            occurred_at=event.occurred_at,
            sequence=event.sequence,
            status=event.status,
        )
        for event in events
    )
    return _timeline_from_observations(observations)


def _timeline_from_observations(
    events: tuple[RunObservation, ...],
) -> RunTimeline:
    started = [event for event in events if event.event_type == EXECUTION_STARTED]
    if len(started) != 1:
        raise ValueError(
            "run profile requires exactly one execution-started Event"
        )
    start = started[0]
    if start.sequence != 0:
        raise ValueError("run execution-started Event must have sequence 0")
    if start.status is not None:
        raise ValueError("run execution-started Event must omit status")

    terminal = [event for event in events if event.event_type == EXECUTION_TERMINAL]
    if len(terminal) > 1:
        raise ValueError(
            "run profile permits at most one execution-terminal Event"
        )
    if terminal:
        completed = terminal[0]
        if completed.status is None:
            raise ValueError("run execution-terminal Event must include status")
        if completed.sequence <= start.sequence:
            raise ValueError("run execution-terminal Event must follow start")
        return RunTimeline(
            started_at=start.occurred_at,
            completed_at=completed.occurred_at,
            status=completed.status,
        )

    return RunTimeline(started_at=start.occurred_at)

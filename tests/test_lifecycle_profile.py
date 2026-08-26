"""Conformance coverage for the lifecycle profile."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp.models import LifecycleEvent, RecordReference
from oclp.profiles import LifecycleTimelineVector, lifecycle_timeline

PROFILE_ROOT = Path(__file__).parent / "profiles" / "lifecycle"


def test_valid_lifecycle_timeline_vector_is_accepted() -> None:
    value = json.loads((PROFILE_ROOT / "valid" / "timeline.json").read_text())

    timeline = LifecycleTimelineVector.model_validate(value)

    assert timeline.events[-1].status == "succeeded"


@pytest.mark.parametrize(
    "name",
    ["terminal-without-status.json", "wrong-profile-version.json"],
)
def test_invalid_lifecycle_vectors_are_rejected(name: str) -> None:
    value = json.loads((PROFILE_ROOT / "invalid" / name).read_text())

    with pytest.raises((ValidationError, ValueError)):
        LifecycleTimelineVector.model_validate(value)


def test_lifecycle_timeline_resolves_portable_event_times() -> None:
    invocation = RecordReference(id="urn:example:invocation:transform")
    events = (
        LifecycleEvent(
            id="urn:example:event:requested",
            invocation=invocation,
            event_type="invocation-requested",
            occurred_at=datetime(2026, 8, 24, 18, 0, tzinfo=UTC),
            sequence=0,
        ),
        LifecycleEvent(
            id="urn:example:event:started",
            invocation=invocation,
            event_type="attempt-started",
            occurred_at=datetime(2026, 8, 24, 18, 0, 1, tzinfo=UTC),
            sequence=1,
            attempt_id="attempt-1",
        ),
        LifecycleEvent(
            id="urn:example:event:terminal",
            invocation=invocation,
            event_type="invocation-terminal",
            occurred_at=datetime(2026, 8, 24, 18, 0, 4, tzinfo=UTC),
            sequence=2,
            status="succeeded",
            attempt_id="attempt-1",
        ),
    )

    timeline = lifecycle_timeline(
        {"version": "0.1.0-draft"},
        events,
    )

    assert timeline.requested_at == events[0].occurred_at
    assert timeline.started_at == events[1].occurred_at
    assert timeline.completed_at == events[2].occurred_at
    assert timeline.status == "succeeded"

"""Conformance coverage for the lifecycle profile."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp import canonical_json_bytes, record_digest
from oclp.models import Event, RecordReference
from oclp.profiles import LifecycleTimelineVector, lifecycle_timeline


def _profile_root() -> Path:
    configured = os.environ.get("OCLP_PROFILES_ROOT")
    if configured is None:
        pytest.skip("set OCLP_PROFILES_ROOT to run the profile conformance suite")
    return Path(configured).resolve() / "tests" / "profiles" / "lifecycle"


def _manifest() -> dict[str, object]:
    return json.loads((_profile_root() / "manifest.json").read_text())


def test_valid_lifecycle_timeline_vector_is_accepted() -> None:
    for entry in _manifest()["valid"]:
        assert isinstance(entry, dict)
        timeline = LifecycleTimelineVector.model_validate(
            json.loads((_profile_root() / entry["path"]).read_text())
        )

        assert canonical_json_bytes(timeline).decode() == entry["canonical_json"]
        assert str(record_digest(timeline)) == entry["digest"]


def test_invalid_lifecycle_vectors_are_rejected() -> None:
    for name in _manifest()["invalid"]:
        assert isinstance(name, str)
        value = json.loads((_profile_root() / name).read_text())

        with pytest.raises((ValidationError, ValueError)):
            LifecycleTimelineVector.model_validate(value)


def test_lifecycle_timeline_resolves_portable_event_times() -> None:
    execution = RecordReference(id="urn:example:execution:transform")
    events = (
        Event(
            id="urn:example:event:started",
            execution=execution,
            event_type="execution-started",
            occurred_at=datetime(2026, 8, 24, 18, 0, tzinfo=UTC),
            sequence=0,
        ),
        Event(
            id="urn:example:event:terminal",
            execution=execution,
            event_type="execution-terminal",
            occurred_at=datetime(2026, 8, 24, 18, 0, 4, tzinfo=UTC),
            sequence=1,
            status="succeeded",
        ),
    )

    timeline = lifecycle_timeline(
        {"version": "0.2.0-draft"},
        events,
    )

    assert timeline.started_at == events[0].occurred_at
    assert timeline.completed_at == events[1].occurred_at
    assert timeline.status == "succeeded"


def test_lifecycle_binding_can_identify_a_shared_run() -> None:
    vector = LifecycleTimelineVector.model_validate(
        {
            "binding": {
                "version": "0.2.0-draft",
                "run_id": "urn:example:lifecycle:nightly:2026-09-01",
                "run_name": "Nightly feature build",
            },
            "events": [
                {
                    "event_type": "execution-started",
                    "occurred_at": "2026-09-01T01:00:00Z",
                    "sequence": 0,
                }
            ],
        }
    )

    assert vector.binding.run_id == "urn:example:lifecycle:nightly:2026-09-01"
    assert vector.binding.run_name == "Nightly feature build"


def test_lifecycle_binding_rejects_a_run_name_without_an_identity() -> None:
    with pytest.raises((ValidationError, ValueError), match="run_name requires run_id"):
        LifecycleTimelineVector.model_validate(
            {
                "binding": {"version": "0.2.0-draft", "run_name": "Nightly"},
                "events": [
                    {
                        "event_type": "execution-started",
                        "occurred_at": "2026-09-01T01:00:00Z",
                        "sequence": 0,
                    }
                ],
            }
        )

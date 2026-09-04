"""Conformance coverage for the run profile."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from oclp import canonical_json_bytes, record_digest
from oclp.models import Event, RecordReference
from oclp.profiles import RunTimelineVector, run_timeline


def _profile_root() -> Path:
    configured = os.environ.get("OCLP_PROFILES_ROOT")
    if configured is None:
        pytest.skip("set OCLP_PROFILES_ROOT to run the profile conformance suite")
    return Path(configured).resolve() / "tests" / "profiles" / "run"


def _manifest() -> dict[str, object]:
    return json.loads((_profile_root() / "manifest.json").read_text())


def test_valid_run_timeline_vector_is_accepted() -> None:
    for entry in _manifest()["valid"]:
        assert isinstance(entry, dict)
        timeline = RunTimelineVector.model_validate(
            json.loads((_profile_root() / entry["path"]).read_text())
        )

        assert canonical_json_bytes(timeline).decode() == entry["canonical_json"]
        assert str(record_digest(timeline)) == entry["digest"]


def test_invalid_run_vectors_are_rejected() -> None:
    for name in _manifest()["invalid"]:
        assert isinstance(name, str)
        value = json.loads((_profile_root() / name).read_text())

        with pytest.raises((ValidationError, ValueError)):
            RunTimelineVector.model_validate(value)


def test_run_timeline_resolves_portable_event_times() -> None:
    execution = RecordReference(id=_id("execution:transform"))
    events = (
        Event(
            id=_id("event:started"),
            execution=execution,
            event_type="execution-started",
            occurred_at=datetime(2026, 8, 24, 18, 0, tzinfo=UTC),
            sequence=0,
        ),
        Event(
            id=_id("event:terminal"),
            execution=execution,
            event_type="execution-terminal",
            occurred_at=datetime(2026, 8, 24, 18, 0, 4, tzinfo=UTC),
            sequence=1,
            status="succeeded",
        ),
    )

    timeline = run_timeline(
        {
            "version": "0.3.0-draft",
            "run_id": "f61f7e4b-5f9f-4e9d-9f0d-3e91a9fa7d4b",
            "run_name": "Transform",
        },
        events,
    )

    assert timeline.started_at == events[0].occurred_at
    assert timeline.completed_at == events[1].occurred_at
    assert timeline.status == "succeeded"


def test_run_binding_can_identify_a_shared_run() -> None:
    vector = RunTimelineVector.model_validate(
        {
            "binding": {
                "version": "0.3.0-draft",
                "run_id": "cfcec749-8509-4ea1-8aa9-03e9969d3aa4",
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

    assert str(vector.binding.run_id) == "cfcec749-8509-4ea1-8aa9-03e9969d3aa4"
    assert vector.binding.run_name == "Nightly feature build"


def test_run_binding_requires_a_uuid_identity_and_name() -> None:
    with pytest.raises((ValidationError, ValueError)):
        RunTimelineVector.model_validate(
            {
                "binding": {"version": "0.3.0-draft", "run_name": "Nightly"},
                "events": [
                    {
                        "event_type": "execution-started",
                        "occurred_at": "2026-09-01T01:00:00Z",
                        "sequence": 0,
                    }
                ],
            }
        )


def _id(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"test:run:{name}"))

"""Verify this SDK against an explicitly selected OCLP standards checkout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from oclp import (
    canonical_json_bytes,
    parse_record,
    record_digest,
    validate_derivation_graph,
)


def _standard_root() -> Path:
    configured = os.environ.get("OCLP_STANDARD_ROOT")
    if configured is None:
        pytest.skip("set OCLP_STANDARD_ROOT to run the standards conformance suite")
    return Path(configured).resolve()


def test_core_conformance_vectors() -> None:
    standard_root = _standard_root()
    fixtures = standard_root / "tests" / "conformance"
    manifest = json.loads((fixtures / "manifest.json").read_text())

    parsed_records = []
    for entry in manifest["valid"]:
        assert isinstance(entry, dict)
        record = parse_record(json.loads((fixtures / entry["path"]).read_text()))
        assert canonical_json_bytes(record).decode() == entry["canonical_json"]
        assert str(record_digest(record)) == entry["digest"]
        parsed_records.append(record)

    for fixture_path in manifest["invalid"]:
        with pytest.raises(ValidationError):
            parse_record(json.loads((fixtures / fixture_path).read_text()))

    validate_derivation_graph(parsed_records)

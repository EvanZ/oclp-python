"""Tests for callable-bound OCLP Definition declarations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oclp import GitSource, definition, definition_record, definition_template
from oclp.models import PortDefinition


@definition(
    id="urn:example:definition:normalize-report",
    name="Normalize report",
    input_ports=(PortDefinition(name="source", media_types=("application/json",)),),
    output_ports=(PortDefinition(name="report", media_types=("application/json",)),),
)
def normalize_report(value: str) -> str:
    """An ordinary callable with colocated OCLP Definition metadata."""

    return value.strip()


def test_definition_decorator_keeps_callable_behavior_and_derives_locator() -> None:
    record = definition_record(
        normalize_report,
        source=GitSource(
            repository="https://github.com/example/reports.git",
            commit="a" * 40,
            path="src/reports.py",
        ),
    )

    assert normalize_report(" report ") == "report"
    assert definition_template(normalize_report).id == record.id
    assert record.implementation.locator.endswith(".normalize_report")
    assert record.input_ports[0].name == "source"
    assert record.output_ports[0].name == "report"


def test_definition_template_requires_decorated_callable() -> None:
    def undecorated() -> None:
        pass

    with pytest.raises(ValueError, match="has no OCLP Definition template"):
        definition_template(undecorated)


def test_definition_rejects_duplicate_ports_when_declared() -> None:
    with pytest.raises(ValidationError, match="port names must be unique"):
        definition(
            id="urn:example:definition:invalid",
            name="Invalid definition",
            input_ports=(PortDefinition(name="source"), PortDefinition(name="source")),
        )

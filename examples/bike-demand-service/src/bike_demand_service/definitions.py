"""Materialize callable-bound OCLP Definitions for the bike-demand demo."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from oclp import ComputationDefinition, GitSource, OpaqueSource, definition_record


def _implementation_source(project_root: Path) -> GitSource | OpaqueSource:
    """Bind Definitions to this checkout when Git metadata is available."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        ).stdout.strip()
        repository = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            check=True,
            cwd=project_root,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return OpaqueSource(
            reason="Git source metadata was unavailable at observation time."
        )
    return GitSource(
        repository=repository or "https://github.com/EvanZ/oclp-python.git",
        commit=commit,
        path="examples/bike-demand-service/src/bike_demand_service",
    )


def definitions(
    project_root: Path,
    *,
    functions: Mapping[str, Callable[..., object]],
) -> dict[str, ComputationDefinition]:
    """Bind decorated demo callables to the Git source observed for this run."""

    source = _implementation_source(project_root)
    return {
        key: definition_record(function, source=source)
        for key, function in functions.items()
    }

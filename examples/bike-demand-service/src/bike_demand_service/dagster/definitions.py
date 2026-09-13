"""Dagster code-location definition for the bike-demand example."""

from __future__ import annotations

import dagster as dg

from bike_demand_service.dagster.assets import ALL_ASSETS
from bike_demand_service.dagster.jobs import ALL_JOBS
from bike_demand_service.dagster.resources import resource_definitions
from bike_demand_service.dagster.sensors import ALL_SENSORS

defs = dg.Definitions(
    assets=ALL_ASSETS,
    jobs=ALL_JOBS,
    sensors=ALL_SENSORS,
    resources=resource_definitions(),
)

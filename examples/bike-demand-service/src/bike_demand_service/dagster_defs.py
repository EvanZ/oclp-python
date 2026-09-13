"""Thin Dagster code-location entry point for the bike-demand example.

The focused native Dagster declarations live in ``bike_demand_service.dagster``.
This module exists only because ``dagster dev -m`` needs one stable entry point.
"""

from bike_demand_service.dagster.definitions import defs

__all__ = ["defs"]

"""Small entry point for the scaffold before pipeline commands exist."""

from __future__ import annotations

import argparse

from bike_demand_service.pipeline import PLANNED_STAGES


def main() -> None:
    """Print the planned demo boundaries until executable commands are added."""

    parser = argparse.ArgumentParser(description="OCLP bike-demand service demo")
    parser.add_argument(
        "command",
        choices=("status",),
        nargs="?",
        default="status",
        help="Show the scaffold's planned stages.",
    )
    parser.parse_args()

    print("OCLP bike-demand service demo (scaffold)")
    print("No data is downloaded and no model is trained yet.\n")
    for number, stage in enumerate(PLANNED_STAGES, start=1):
        print(f"{number}. {stage.name} — {stage.description}")

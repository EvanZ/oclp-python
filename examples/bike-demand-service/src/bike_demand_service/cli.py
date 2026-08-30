"""Command-line entry point for the locally observable batch milestone."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from bike_demand_service.pipeline import PLANNED_STAGES
from bike_demand_service.runner import run_demo


def main() -> None:
    """Describe the demo or run its complete batch lifecycle."""

    parser = argparse.ArgumentParser(description="OCLP bike-demand service demo")
    parser.add_argument(
        "command",
        choices=("status", "run"),
        nargs="?",
        default="status",
        help="Show the planned stages or run the batch lifecycle.",
    )
    parser.add_argument(
        "--run-id",
        help="Stable run label; defaults to a UTC timestamped bike-demand run.",
    )
    arguments = parser.parse_args()

    if arguments.command == "status":
        print("OCLP bike-demand service demo")
        print("The batch milestone is ready; FastAPI inference is deferred.\n")
        for number, stage in enumerate(PLANNED_STAGES, start=1):
            print(f"{number}. {stage.name} — {stage.description}")
        return

    run_id = arguments.run_id or datetime.now(UTC).strftime(
        "bike-demand-%Y%m%dT%H%M%SZ"
    )
    result = run_demo(run_id=run_id)
    print(f"Completed OCLP bike-demand run: {result.run_id}")
    print(f"OCLP records: {result.oclp_root}")
    print(f"Root Invocation: {result.root_invocation.id}")
    print(f"Model release: {result.model_release.id}")
    print(f"MLflow tracking URI: {result.mlflow_tracking_uri}")

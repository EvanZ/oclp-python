"""Command-line entry point for local OCLP training and inference."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_service.runner import run_demo
from bike_demand_service.service import create_app


def main() -> None:
    """Run local model training or serve a specific released model."""

    parser = argparse.ArgumentParser(description="OCLP bike-demand service demo")
    parser.add_argument(
        "command",
        choices=("run", "serve"),
        nargs="?",
        default="run",
        help="Run model training or serve one release manifest.",
    )
    parser.add_argument(
        "--materialization-id",
        help="Local output-folder key; defaults to a UTC timestamped value.",
    )
    parser.add_argument(
        "--temporal-validation-rmse-max",
        type=float,
        default=250,
        help="Maximum validation RMSE accepted by the temporal quality gate.",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=3,
        help="Number of temporal cross-validation folds (minimum 2).",
    )
    parser.add_argument(
        "--release-manifest",
        type=Path,
        help="Path to the SDK-created release-manifest.json required by serve.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface used by serve (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port used by serve (default: 8000).",
    )
    arguments = parser.parse_args()

    if arguments.command == "serve":
        if arguments.release_manifest is None:
            parser.error("serve requires --release-manifest PATH")
        import uvicorn

        uvicorn.run(
            create_app(release_manifest_path=arguments.release_manifest),
            host=arguments.host,
            port=arguments.port,
        )
        return

    materialization_id = arguments.materialization_id or datetime.now(UTC).strftime(
        "bike-demand-%Y%m%dT%H%M%SZ"
    )
    result = run_demo(
        materialization_id=materialization_id,
        fold_count=arguments.fold_count,
        temporal_validation_rmse_max=arguments.temporal_validation_rmse_max,
    )
    print(f"Completed OCLP bike-demand training run: {result.training_run_id}")
    print(f"OCLP records: {result.oclp_root}")
    print(f"Model release: {result.model_release.id}")
    print(f"Release manifest: {result.model_release_manifest_path}")
    print(f"Release smoke execution: {result.release_smoke_execution.id}")
    print(f"Release smoke response: {result.release_smoke_response.id}")
    print(f"MLflow tracking URI: {result.mlflow_tracking_uri}")

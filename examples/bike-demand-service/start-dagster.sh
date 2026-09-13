#!/usr/bin/env bash
# Start the bike-demand Dagster code location from any working directory.

set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
port="${DAGSTER_PORT:-3000}"

cd "$example_dir"
exec uv run --extra dagster dagster dev \
  -m bike_demand_service.dagster_defs \
  --port "$port" \
  "$@"

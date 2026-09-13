#!/usr/bin/env bash
# Start the bike-demand local development stack: MLflow, Dagster, and Cyclops.

set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_root="$(cd -- "$example_dir/../.." && pwd)"
explorer_root="${OCLP_EXPLORER_ROOT:-$(cd -- "$sdk_root/../oclp-explorer" 2>/dev/null && pwd || true)}"
example_python="$example_dir/.venv/bin/python"
explorer_python="$explorer_root/.venv/bin/python"
explorer_api_bin="$explorer_root/.venv/bin/oclp-explorer"
oclp_dir="$example_dir/data/oclp-0.3"
mlflow_root="$example_dir/data/mlflow"
runtime_dir="$example_dir/data/local-services"
mlflow_port="${MLFLOW_PORT:-5000}"
dagster_port="${DAGSTER_PORT:-3000}"
cyclops_api_port="${CYCLOPS_API_PORT:-8002}"
cyclops_port="${CYCLOPS_PORT:-5175}"

listener_pids() {
  /usr/sbin/lsof -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

process_tree_has_marker() {
  local pid="$1"
  local command_marker="$2"
  for _ in {1..8}; do
    local command
    command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command" == *"$command_marker"* ]] && return

    local parent_pid
    parent_pid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')"
    [[ -z "$parent_pid" || "$parent_pid" == "1" || "$parent_pid" == "$pid" ]] && break
    pid="$parent_pid"
  done
  return 1
}

wait_for_listener() {
  local port="$1"
  local label="$2"
  local log_file="$3"
  for _ in {1..150}; do
    [[ -n "$(listener_pids "$port")" ]] && return
    sleep 0.2
  done
  echo "$label did not start on port $port. Recent log output:"
  tail -n 40 "$log_file" || true
  exit 1
}

wait_for_health() {
  local url="$1"
  local label="$2"
  local log_file="$3"
  for _ in {1..150}; do
    curl -fsS "$url" >/dev/null 2>&1 && return
    sleep 0.2
  done
  echo "$label started but did not become healthy. Recent log output:"
  tail -n 60 "$log_file" || true
  exit 1
}

start_service() {
  local port="$1"
  local label="$2"
  local command_marker="$3"
  local log_file="$4"
  shift 4

  local pids
  pids="$(listener_pids "$port")"
  if [[ -n "$pids" ]]; then
    for pid in $pids; do
      if ! process_tree_has_marker "$pid" "$command_marker"; then
        local command
        command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
        echo "Refusing to use port $port: PID $pid is not $label."
        echo "Command: $command"
        exit 1
      fi
    done
    echo "$label is already running on port $port."
    return
  fi

  : >"$log_file"
  nohup "$@" >"$log_file" 2>&1 < /dev/null &
  wait_for_listener "$port" "$label" "$log_file"
  echo "$label started on port $port."
}

if [[ ! -x "$example_python" ]]; then
  echo "Bike-demand environment is unavailable: $example_python"
  echo "Create it with: cd $example_dir && uv sync --group dev --extra dagster"
  exit 1
fi
if [[ -z "$explorer_root" || ! -d "$explorer_root/apps/cyclops" ]]; then
  echo "Cyclops checkout is unavailable: ${OCLP_EXPLORER_ROOT:-$sdk_root/../oclp-explorer}"
  echo "Set OCLP_EXPLORER_ROOT to an oclp-explorer checkout."
  exit 1
fi
if [[ ! -x "$explorer_api_bin" || ! -x "$explorer_python" ]]; then
  echo "Cyclops environment is unavailable: $explorer_root/.venv"
  echo "Create it with: cd $explorer_root && uv sync --all-groups"
  exit 1
fi
if ! "$explorer_python" -c 'import fasteners' >/dev/null 2>&1; then
  echo "Cyclops is missing its declared OCLP DuckDB dependencies."
  echo "Sync them with: cd $explorer_root && uv sync --all-groups"
  exit 1
fi
if ! uv_bin="$(command -v uv)"; then
  echo "uv is required to start MLflow and Dagster."
  exit 1
fi
if ! npm_bin="$(command -v npm)"; then
  echo "npm is required to start the Cyclops UI."
  exit 1
fi

mkdir -p "$oclp_dir" "$mlflow_root" "$runtime_dir"
cd "$example_dir"

start_service \
  "$mlflow_port" \
  "MLflow" \
  "mlflow" \
  "$runtime_dir/mlflow.log" \
  "$uv_bin" run mlflow ui \
  --backend-store-uri "sqlite:///$mlflow_root/mlflow.db" \
  --host 127.0.0.1 \
  --port "$mlflow_port"
wait_for_health "http://127.0.0.1:$mlflow_port" "MLflow" "$runtime_dir/mlflow.log"

start_service \
  "$dagster_port" \
  "Dagster" \
  "dagster" \
  "$runtime_dir/dagster.log" \
  env "DAGSTER_PORT=$dagster_port" "$example_dir/start-dagster.sh"
wait_for_health "http://127.0.0.1:$dagster_port" "Dagster" "$runtime_dir/dagster.log"

start_service \
  "$cyclops_api_port" \
  "Cyclops API" \
  "oclp-explorer" \
  "$runtime_dir/cyclops-api.log" \
  "$explorer_api_bin" \
  --oclp-dir "$oclp_dir" \
  --port "$cyclops_api_port"
wait_for_health \
  "http://127.0.0.1:$cyclops_api_port/api/health" \
  "Cyclops API" \
  "$runtime_dir/cyclops-api.log"

start_service \
  "$cyclops_port" \
  "Cyclops UI" \
  "vite" \
  "$runtime_dir/cyclops-ui.log" \
  "$npm_bin" --prefix "$explorer_root/apps/cyclops" run dev
wait_for_health \
  "http://127.0.0.1:$cyclops_port" \
  "Cyclops UI" \
  "$runtime_dir/cyclops-ui.log"

echo
echo "Local bike-demand stack is ready:"
echo "  Dagster:  http://127.0.0.1:$dagster_port"
echo "  MLflow:   http://127.0.0.1:$mlflow_port"
echo "  Cyclops:  http://127.0.0.1:$cyclops_port"
echo "Logs: $runtime_dir"

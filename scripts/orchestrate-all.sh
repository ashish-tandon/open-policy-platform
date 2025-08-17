#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
INV="$ROOT/docs/reference/services.inventory.yaml"
MANAGER="$ROOT/scripts/manage-service.sh"
ACTION="${1:-redeploy}"

if ! command -v yq >/dev/null 2>&1; then
  echo "yq is required (https://mikefarah.gitbook.io/yq/)" >&2
  exit 1
fi

services=($(yq '.services[].name' "$INV"))
paths=($(yq '.services[].path' "$INV"))
ports=($(yq '.services[].port' "$INV"))

pids=()
for i in "${!services[@]}"; do
  name=${services[$i]}
  path=${paths[$i]}
  port=${ports[$i]}
  ( SERVICE_NAME="$name" SERVICE_PATH="$path" SERVICE_PORT="$port" "$MANAGER" "$ACTION" ) &
  pids+=("$!")
  echo "[orchestrator] $ACTION scheduled for $name ($path:$port)"

done

for pid in "${pids[@]}"; do
  wait "$pid" || true

done

echo "[orchestrator] $ACTION complete for ${#services[@]} services"
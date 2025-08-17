#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"

check() {
  local path="$1"
  echo "Checking ${BASE_URL}${path}"
  curl -fsS "${BASE_URL}${path}" >/dev/null
}

check "/api/v1/health"
check "/api/v1/health/detailed"
check "/api/v1/dashboard/system"

echo "Smoke tests passed"
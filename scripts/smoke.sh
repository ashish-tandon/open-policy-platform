#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

# Wait for API to respond
for i in {1..30}; do
  if curl -fsS "$API_URL/api/v1/health" >/dev/null; then
    break
  fi
  sleep 1
done

bash "$(cd "$(dirname "$0")" && pwd)/smoke-test.sh"
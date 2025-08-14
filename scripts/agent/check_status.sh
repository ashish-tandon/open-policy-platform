#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}

echo "API health:" && curl -fsS "$API_URL/api/v1/health" | jq -r . || true
echo "API detailed:" && curl -fsS "$API_URL/api/v1/health/detailed" | jq -r . || true

echo "Workers status:" && curl -fsS "$API_URL/api/v1/admin/workers/status" | jq -r . || true

echo "Enqueue ping:" && curl -fsS -X POST "$API_URL/api/v1/admin/workers/ping" | jq -r . || true


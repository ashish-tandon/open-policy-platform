#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:5173}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

curl -fsS "$API_URL/api/v1/health" >/dev/null && echo "API OK: $API_URL"
curl -fsS "$API_URL/api/v1/health/detailed" >/dev/null && echo "API detailed OK"

curl -fsS "$WEB_URL" >/dev/null && echo "Web OK: $WEB_URL" || echo "Web check skipped (dev server may not be running)"

nc -z "$DB_HOST" "$DB_PORT" && echo "DB OK: $DB_HOST:$DB_PORT" || echo "DB port not open yet"
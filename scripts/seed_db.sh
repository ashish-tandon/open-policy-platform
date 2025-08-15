#!/usr/bin/env bash
set -euo pipefail

DB_USER=${DB_USER:-openpolicy}
DB_NAME=${DB_NAME:-openpolicy_app}

if docker compose ps postgres >/dev/null 2>&1; then
  docker compose exec -T postgres sh -lc "psql -U $DB_USER -d $DB_NAME -f /seed_db.sql"
else
  echo "postgres container not running" >&2
  exit 1
fi


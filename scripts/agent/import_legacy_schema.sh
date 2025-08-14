#!/usr/bin/env bash
set -euo pipefail

# Imports schema-only from a legacy SQL dump into Postgres inside docker compose
# Usage: bash scripts/agent/import_legacy_schema.sh [/absolute/path/to/dump.sql]

DBFILE=${1:-"$(pwd)/openparliament.public.sql"}
TARGET_DB=${TARGET_DB:-openparliament_public}
TARGET_ROLE=${TARGET_ROLE:-op}
PSQL_USER=${PSQL_USER:-openpolicy}

if [ ! -f "$DBFILE" ]; then
  echo "Dump not found: $DBFILE" >&2
  exit 1
fi

echo "Ensuring postgres service..."
DOCKER_BUILDKIT=1 docker compose up -d postgres >/dev/null

PSQL_CONT=$(docker ps --format '{{.Names}}' | grep -m1 '^open-policy-platform-postgres-1$' || true)
if [ -z "$PSQL_CONT" ]; then
  echo "Postgres container not found" >&2
  exit 1
fi

echo "Creating role $TARGET_ROLE if missing and recreating DB $TARGET_DB (owner=$TARGET_ROLE)..."
docker compose exec -T postgres psql -U "$PSQL_USER" -d postgres -v ON_ERROR_STOP=1 <<SQL
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$TARGET_ROLE') THEN
    CREATE ROLE $TARGET_ROLE LOGIN;
  END IF;
END$$;
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$TARGET_DB';
DROP DATABASE IF EXISTS $TARGET_DB;
CREATE DATABASE $TARGET_DB OWNER $TARGET_ROLE;
SQL

echo "Filtering schema-only from: $DBFILE"
TMP_SCHEMA=$(mktemp)
awk '
  BEGIN{skip=0}
  /^[[:space:]]*COPY[[:space:]]/ { skip=1; next }
  skip && /^\\\.$/ { skip=0; next }
  skip { next }
  /^[[:space:]]*INSERT[[:space:]]/ { next }
  { print }
' "$DBFILE" > "$TMP_SCHEMA"
echo "Schema-only file: $TMP_SCHEMA ($(du -h "$TMP_SCHEMA" | awk '{print $1}'))"

echo "Copying and importing schema into $TARGET_DB..."
docker cp "$TMP_SCHEMA" "$PSQL_CONT:/tmp/schema_only.sql"
docker compose exec -T postgres psql -U "$PSQL_USER" -d "$TARGET_DB" -v ON_ERROR_STOP=1 -f /tmp/schema_only.sql >/dev/null
echo "Schema import completed."



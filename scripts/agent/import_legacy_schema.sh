#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 /absolute/path/to/schema.sql" >&2
	exit 1
fi

SCHEMA_SQL="$1"
DB_USER="openpolicy"
DB_PASS="openpolicy123"
DB_DB="openpolicy_app"
DB_HOST="localhost"
DB_PORT="5432"

# Import schema into a dedicated schema for diffing
TARGET_SCHEMA="legacy_schema"

export PGPASSWORD="$DB_PASS"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_DB" -v ON_ERROR_STOP=1 -c "CREATE SCHEMA IF NOT EXISTS ${TARGET_SCHEMA};"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_DB" -v ON_ERROR_STOP=1 -c "SET search_path TO ${TARGET_SCHEMA};" < "$SCHEMA_SQL"

echo "Imported legacy schema into schema '${TARGET_SCHEMA}' in database ${DB_DB}."



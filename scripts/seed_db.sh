#!/usr/bin/env bash
set -euo pipefail

# Seed Open Policy Platform databases inside the postgres service
# Requires: docker compose stack running with 'postgres' service healthy

APP_DB="${APP_DB:-openpolicy_app}"
SCRAPERS_DB="${SCRAPERS_DB:-openpolicy_scrapers}"
USER_NAME="${POSTGRES_USER:-openpolicy}"
SERVICE="${POSTGRES_SERVICE:-postgres}"

# Copy seed files into container
docker compose cp scripts/seed_app.sql "$SERVICE":/seed_app.sql
docker compose cp scripts/seed_scrapers.sql "$SERVICE":/seed_scrapers.sql

# Apply seeds
docker compose exec -T "$SERVICE" bash -lc "psql -U $USER_NAME -d $APP_DB -f /seed_app.sql"
docker compose exec -T "$SERVICE" bash -lc "psql -U $USER_NAME -d $SCRAPERS_DB -f /seed_scrapers.sql"

echo "Seed completed: $APP_DB and $SCRAPERS_DB"

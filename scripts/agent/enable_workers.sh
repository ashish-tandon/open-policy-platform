#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

export COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1

docker compose --profile workers up -d --build redis celery-worker flower

# Give worker a moment to connect
sleep 5

# Status
curl -fsS http://localhost:8000/api/v1/admin/workers/status || true


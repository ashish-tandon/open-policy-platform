#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Prepare env
if [[ ! -f .env ]]; then
	cp env.example .env
fi
# Ensure SECRET_KEY exists to avoid warnings
if ! grep -q '^SECRET_KEY=' .env; then
	SECRET=$(openssl rand -hex 32 2>/dev/null || echo "dev-secret-key")
	echo "SECRET_KEY=$SECRET" >> .env
fi

export COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1
# Core services only to keep dev lean
services=(postgres api web scraper-runner)

docker compose up -d --build "${services[@]}"

# Wait for API health
printf "Waiting for API health"
for i in $(seq 1 60); do
	if curl -fsS http://localhost:8000/api/v1/health >/dev/null; then
		echo -e "\nAPI is healthy"
		break
	fi
	printf "."; sleep 2
	if [[ $i -eq 60 ]]; then echo "\nTimed out waiting for API" && exit 1; fi
done

# Initialize scrapers DB tables (idempotent)
bash scripts/agent/init_scrapers_db.sh || true

# Smoke test
bash scripts/smoke-test.sh || true

echo "Container status:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'


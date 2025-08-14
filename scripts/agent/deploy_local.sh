#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
WEB_URL=${WEB_URL:-http://localhost:5173}

test -f .env || cp env.example .env

# Optional: set a SECRET_KEY if not present
if ! grep -q '^SECRET_KEY=' .env 2>/dev/null; then
  echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env || true
fi

DOCKER_BUILDKIT=1 docker compose up -d --build postgres api web scraper-runner

sleep 5
bash scripts/smoke-test.sh

echo "Deployment complete:"
echo "- API: $API_URL"
echo "- WEB: $WEB_URL"


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Try via Docker Compose API container
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
	echo "Initializing scrapers DB via API container..."
	docker compose run --rm api python backend/scripts/init_scrapers_db.py || true
else
	echo "Docker not available; initializing locally..."
	python backend/scripts/init_scrapers_db.py
fi

echo "Done."
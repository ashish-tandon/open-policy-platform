#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT_DIR"

BRANCH=${WATCH_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
INTERVAL=${WATCH_INTERVAL_SECONDS:-60}
RESULTS_DIR="scripts/agent/results"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

while true; do
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  OUT="$RESULTS_DIR/$TS.log"
  {
    log "Fetching updates (branch=$BRANCH)"
    git fetch origin "$BRANCH"
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse "origin/$BRANCH")
    if [ "$LOCAL" != "$REMOTE" ]; then
      log "Rebasing onto origin/$BRANCH"
      git pull --rebase origin "$BRANCH"
    else
      log "No new commits"
    fi

    log "Rebuilding and starting services"
    DOCKER_BUILDKIT=1 docker compose up -d --build postgres api web scraper-runner

    log "Running smoke test"
    bash scripts/smoke-test.sh || true

    log "Running backend tests in API container"
    # Ensure API is up before running tests
    for i in {1..20}; do
      if curl -fsS http://localhost:8000/api/v1/health >/dev/null; then break; fi
      sleep 2
    done
    docker compose exec -T api bash -lc 'cd backend && pytest -q' || true

    log "Done cycle for $TS"
  } | tee "$OUT" >/dev/null

  cp "$OUT" scripts/agent/latest_results.txt || true
  log "Sleeping for $INTERVAL seconds"
  sleep "$INTERVAL"
done



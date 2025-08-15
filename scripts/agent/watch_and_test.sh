#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"
mkdir -p scripts/agent/results

logfile="scripts/agent/results/$(date +%Y%m%d_%H%M%S).log"
latest="scripts/agent/latest_results.txt"

while true; do
  {
    echo "=== $(date) - pulling updates ==="
    git fetch origin && git reset --hard "origin/$(git rev-parse --abbrev-ref HEAD)"

    echo "=== rebuild and up core ==="
    export COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1
    docker compose up -d --build postgres api web scraper-runner

    echo "=== wait for API ==="
    for i in $(seq 1 60); do
      if curl -fsS http://localhost:8000/api/v1/health >/dev/null; then echo "API healthy"; break; fi
      sleep 2
      [[ $i -eq 60 ]] && echo "API failed to become healthy" && break
    done

    echo "=== smoke test ==="
    bash scripts/smoke-test.sh || true

    echo "=== backend tests ==="
    (cd backend && pytest -q) || true

    echo "=== done ==="
  } | tee "$logfile"
  cp "$logfile" "$latest" || true

  # Optional: publish results to a dedicated results branch
  if [[ "${PUBLISH_RESULTS:-}" == "1" ]]; then
    BRANCH="$(git rev-parse --abbrev-ref HEAD)"
    RESULTS_BRANCH="results/${BRANCH}"
    SHA="$(git rev-parse HEAD)"
    TMPDIR="$(mktemp -d)"
    set +e
    git fetch origin "$RESULTS_BRANCH"
    set -e
    # Prepare worktree for results branch
    if git ls-remote --heads origin "$RESULTS_BRANCH" | grep -q "$RESULTS_BRANCH"; then
      git worktree add "$TMPDIR" -f "origin/$RESULTS_BRANCH" || true
      (cd "$TMPDIR" && git checkout -B "$RESULTS_BRANCH")
    else
      git worktree add -b "$RESULTS_BRANCH" "$TMPDIR" || true
      (cd "$TMPDIR" && echo "Results for $BRANCH" > README.md && mkdir -p results && git add . && git commit -m "init results branch")
    fi
    mkdir -p "$TMPDIR/results"
    cp "$latest" "$TMPDIR/results/latest.txt" || true
    cp "$logfile" "$TMPDIR/results/${SHA}.log" || true
    (cd "$TMPDIR" && git add results README.md && git commit -m "results: ${SHA:0:7} at $(date -u +%FT%TZ)" && git push -u origin "$RESULTS_BRANCH") || true
    git worktree remove "$TMPDIR" --force || true
  fi

  sleep 60
done



#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

echo "== Pulling latest =="
git fetch origin
BR="$(git rev-parse --abbrev-ref HEAD)"
git pull --rebase origin "$BR"

echo "== Deploying core stack =="
bash scripts/agent/deploy_local.sh

echo "== Latest results =="
if [[ -f scripts/agent/latest_results.txt ]]; then
	cat scripts/agent/latest_results.txt | tail -n 200
else
	echo "No latest results yet."
fi

echo "== Done =="
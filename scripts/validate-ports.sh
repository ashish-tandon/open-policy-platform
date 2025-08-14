#!/usr/bin/env bash
set -euo pipefail

FILE=${1:-docker-compose.yml}

grep -q '"8000:8000"' "$FILE" || { echo "Missing API port 8000 mapping"; exit 1; }
grep -q '"5173:5173"' "$FILE" || { echo "Missing Web port 5173 mapping"; exit 1; }
grep -q '"5432:5432"' "$FILE" || { echo "Missing DB port 5432 mapping"; exit 1; }
grep -q '"6379:6379"' "$FILE" || { echo "Missing Redis port 6379 mapping"; exit 1; }
grep -q '"9090:9090"' "$FILE" || { echo "Missing Prometheus port 9090 mapping"; exit 1; }
grep -q '"3000:3000"' "$FILE" || { echo "Missing Grafana port 3000 mapping"; exit 1; }

echo "Ports validated in $FILE"
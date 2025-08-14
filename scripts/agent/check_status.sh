#!/usr/bin/env bash
set -euo pipefail

curl -fsS http://localhost:8000/api/v1/health | jq . || curl -fsS http://localhost:8000/api/v1/health || true
curl -fsS http://localhost:8000/api/v1/health/detailed | jq . || curl -fsS http://localhost:8000/api/v1/health/detailed || true
curl -fsS http://localhost:8000/api/v1/admin/workers/status | jq . || curl -fsS http://localhost:8000/api/v1/admin/workers/status || true


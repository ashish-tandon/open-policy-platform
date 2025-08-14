#!/usr/bin/env bash
set -euo pipefail

DOCKER_BUILDKIT=1 docker compose --profile workers up -d

echo "Workers profile enabled."


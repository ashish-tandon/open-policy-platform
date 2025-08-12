#!/usr/bin/env bash
set -euo pipefail

# Usage: env SERVICE_NAME=auth-service SERVICE_PATH=services/auth-service SERVICE_PORT=9001 HEALTH_PATH=/healthz ./scripts/manage-service.sh <action>
# Actions: cleanup | rebuild | start | stop | restart | status | redeploy

: "${SERVICE_NAME:?SERVICE_NAME required}"
: "${SERVICE_PATH:?SERVICE_PATH required}"
: "${SERVICE_PORT:?SERVICE_PORT required}"
: "${HEALTH_PATH:=/healthz}"

IMAGE="openpolicy/${SERVICE_NAME}:dev"
CONTAINER="svc_${SERVICE_NAME}"

cleanup() {
  echo "[cleanup] ${SERVICE_NAME}"
  docker rm -f "${CONTAINER}" 2>/dev/null || true
  docker image rm -f "${IMAGE}" 2>/dev/null || true
  rm -rf "${SERVICE_PATH}/.cache" "${SERVICE_PATH}/dist" "${SERVICE_PATH}/tmp" 2>/dev/null || true
}

rebuild() {
  echo "[rebuild] ${SERVICE_NAME}"
  docker build -t "${IMAGE}" "${SERVICE_PATH}"
  # best-effort tests inside container if pytest exists in requirements
  if grep -qi pytest "${SERVICE_PATH}/requirements.txt" 2>/dev/null; then
    docker run --rm -e PYTHONUNBUFFERED=1 -w /app -p 0:0 "${IMAGE}" sh -lc "pytest -q || true"
  fi
}

start() {
  echo "[start] ${SERVICE_NAME} on ${SERVICE_PORT}"
  docker run -d --rm \
    --name "${CONTAINER}" \
    -e PORT="${SERVICE_PORT}" \
    -p "${SERVICE_PORT}:${SERVICE_PORT}" \
    "${IMAGE}"
  CID=$(docker inspect -f '{{.Id}}' "${CONTAINER}")
  echo "${CID}" > "${SERVICE_PATH}/.pid"
  echo "[start] waiting for health ${HEALTH_PATH}"
  for i in {1..20}; do
    if curl -fsS "http://localhost:${SERVICE_PORT}${HEALTH_PATH}" >/dev/null; then
      echo "[start] healthy"
      return 0
    fi
    sleep 1
  done
  echo "[start] health check failed"; docker logs --tail=50 "${CONTAINER}" || true; exit 1
}

stop() {
  echo "[stop] ${SERVICE_NAME}"
  docker rm -f "${CONTAINER}" 2>/dev/null || true
  rm -f "${SERVICE_PATH}/.pid" 2>/dev/null || true
}

restart() { stop; start; }

status() {
  running=$(docker ps --filter "name=${CONTAINER}" --format '{{.ID}}')
  if [ -n "$running" ]; then
    echo "[status] ${SERVICE_NAME}: running as ${running} on port ${SERVICE_PORT}"
    curl -fsS "http://localhost:${SERVICE_PORT}${HEALTH_PATH}" || true
    echo "[logs] recent:"; docker logs --tail=20 "${CONTAINER}" || true
  else
    echo "[status] ${SERVICE_NAME}: not running"
  fi
}

redeploy() { cleanup; rebuild; start; }

case "${1:-}" in
  cleanup|rebuild|start|stop|restart|status|redeploy) "$1";;
  *) echo "Usage: $0 {cleanup|rebuild|start|stop|restart|status|redeploy}"; exit 2;;

esac
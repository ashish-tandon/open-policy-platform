# Service Standards

## Structure (per service)
- Dockerfile
- README.md
- .env.example (or documented env in repo root)
- /docs with API and ops notes (if applicable)
- Health endpoint at /api/v1/health (port per policy)

## Ports (policy)
- API: 8000
- Web (dev): 5173
- Postgres: 5432
- Redis: 6379
- Prometheus: 9090
- Grafana: 3000

All environments follow these defaults. Overrides must be documented and validated.

## Health & Readiness
- Readiness: /api/v1/health (HTTP 200)
- Liveness: /api/v1/health (HTTP 200)
- K8s probes configured; Compose healthchecks present where possible

## Deployment
- Images built via CI with tags
- Dispatch event triggers deployment repo
- Rolling update in k8s or compose pull+up in simple environments

## Observability
- Prometheus metrics available (where implemented)
- Logs to stdout; files mounted where needed (e.g., scraper reports/logs)
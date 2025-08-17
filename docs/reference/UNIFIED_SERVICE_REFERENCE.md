# Unified Service Reference

Authoritative source for services, APIs, healthchecks, variables, and startup order.

## Services
- API (FastAPI)
  - Entrypoint: `backend.api.main:app`
  - Port: 8000
  - Health: `/api/v1/health` (liveness), `/api/v1/health/detailed` (readiness)
  - Metrics: `/metrics` (Prometheus format)
  - OpenAPI: `dist/openapi.(json|yaml)`
- Web (Vite/React)
  - Port: 5173 (dev), behind nginx in prod
  - API base: `VITE_API_URL`
- DB: PostgreSQL
  - Databases: `openpolicy_app`, `openpolicy_scrapers`, `openpolicy_auth`
- Redis: 6379
- Nginx: reverse-proxy to API
- Monitoring: Prometheus + Grafana

## Variables
- Canonical: `DATABASE_URL`
- Optional: `APP_DATABASE_URL`, `SCRAPERS_DATABASE_URL`, `AUTH_DATABASE_URL`
- API: `API_HOST`, `API_PORT`
- Security: `SECRET_KEY`, `ALLOWED_ORIGINS`, `ALLOWED_HOSTS`
- Frontend: `VITE_API_URL`
- Tests: `TEST_DATABASE_URL`

## Startup Order (Prod)
1. PostgreSQL (healthy)
2. API (uvicorn)
3. Nginx (proxy)
4. Web (served via nginx or static host)

## Health and Heartbeat
- Liveness: `/api/v1/health`
- Readiness: `/api/v1/health/detailed`
- Metrics: `/metrics`

## Data Flow
- Scrapers → DB (`openpolicy_scrapers`) and reports/logs
- API reads DB and files → UI via REST

## Contracts
- Endpoints: `docs/api/endpoints.md`
- Schemas: `docs/api/schemas.md`

## CI/CD
- Docs links and OpenAPI export must pass

## Notes
- All variables env-injected; no hardcoded secrets
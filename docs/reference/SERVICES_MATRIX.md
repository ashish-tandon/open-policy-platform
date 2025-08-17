# Services Matrix (Comprehensive)

Authoritative inventory of services/components across the platform.

## Core Services
- API (FastAPI): `backend/api`
  - Health: `/api/v1/health`, `/api/v1/health/detailed`
  - Metrics: `/metrics`
  - Endpoints:
    - Authentication: `/api/v1/auth/*`
    - Policies: `/api/v1/policies/*`
    - Representatives: `/api/v1/representatives/*`
    - Committees: `/api/v1/committees/*`
    - Debates: `/api/v1/debates/*`
    - Votes: `/api/v1/votes/*`
    - Search: `/api/v1/search/*`
    - Analytics: `/api/v1/analytics/*`
    - Notifications: `/api/v1/notifications/*`
    - Files: `/api/v1/files/*`
    - Scrapers: `/api/v1/scrapers/*`
    - Data Management: `/api/v1/*` (data ops)
    - Dashboard: `/api/v1/dashboard/*`
    - Admin: `/api/v1/admin/*`
- Web (Vite/React): `web/`
- Database (PostgreSQL): logical DBs `openpolicy_app`, `openpolicy_scrapers`, `openpolicy_auth`
- Redis: cache/session
- Nginx: reverse proxy
- Monitoring: Prometheus + Grafana
- Scheduler: `backend/unified_daily_update_system.py` (schedule library)
- Scrapers: `backend/scrapers/`, `scrapers/`

## Admin/Operational Scripts
- Setup/Start: `scripts/`
- Scraper orchestration: `backend/run_comprehensive_scrapers.py`, `backend/openparliament_daily_updates.py`
- Monitoring/Status: `backend/production_status.py`, `backend/monitoring_system.py`
- Deploy: `backend/deploy*.py`, docker-compose files

## Infrastructure
- Docker: `backend/Dockerfile`, `backend/docker-compose.yml`, `docker-compose.yml`
- Kubernetes: `infrastructure/k8s/*`
- Monitoring configs: `backend/monitoring/*`

## Testing/CI/CD
- Tests: `backend/tests`, `web/tests`
- CI: `.github/workflows/tests.yml`, `docs-openapi.yml`

## Documentation
- Architecture, data flow, variables, deployment, operations: `docs/*`
- Unified references:
  - `docs/reference/UNIFIED_SERVICE_REFERENCE.md`
  - `docs/reference/SERVICES_MATRIX.md`

## Notes
- All env vars documented in `env.example` and `docs/operations/environment-variables.md`
- Health & readiness probes documented and implemented
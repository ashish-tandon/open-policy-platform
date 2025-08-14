# Master Execution Plan — Open Policy Platform

Version: 1.0.0
Status: Ready for execution

## Objectives
- Single, coordinated plan covering API, frontend, database, scrapers, infrastructure, security, and CI/CD
- Eliminate multiple sources of truth; all work references canonical docs
- Deliver a production-ready, testable platform with clear milestones and acceptance criteria

## Canonical sources of truth
- API Overview: `docs/api/overview.md`
- API Endpoints: `docs/api/endpoints.md`
- API Authentication: `docs/api/authentication.md`
- API Schemas: `docs/api/schemas.md`
- Quick Reference: `docs/api/quick-reference.md`
- Configuration: `backend/api/config.py` (+ `.env` from `env.example`)
- Documentation Governance: `docs/architecture/documentation-consolidation-plan.md`

## Scope
- Backend API (FastAPI @ `backend/api`) — canonical REST
- Frontend (web React app) — consumes canonical REST
- Database (PostgreSQL) — schema and performance alignment
- Scrapers — inventory, execution, monitoring
- Infrastructure — local dev, containerization, basic monitoring/logging
- Security — auth, headers, rate limiting, secrets
- CI/CD — test, build, docs verify, packaging

## Workstreams and deliverables

### 1) Backend API
- Deliverables:
  - Endpoints implemented per `docs/api/endpoints.md`
  - Request/response examples updated in `docs/api/schemas.md`
  - Security and performance middleware configured (headers, validation, rate limit, cache)
  - OpenAPI available at `/openapi.json`; Swagger `/docs` (non-production)
- Acceptance:
  - All routes return 2xx/4xx/5xx as expected
  - Lint/tests pass; OpenAPI loads
- Tasks:
  - Verify router mount prefixes and tags match docs
  - Replace hardcoded secrets via env (`Settings.secret_key`)
  - Add Pydantic response models where missing

### 2) Frontend Web
- Deliverables:
  - API service layer matching `docs/api/endpoints.md`
  - Role-based views for admin vs public
  - Environment config for base API URL
- Acceptance:
  - Happy-path flows work (login, list/search, dashboards)
- Tasks:
  - Build shared API client with base URL from env
  - Implement pages: Policies, Dashboard, Scrapers, Admin, Data

### 3) Database
- Deliverables:
  - Connection works via `Settings.database_url`
  - Reference of key tables used by API
  - Indexes for hot queries
- Acceptance:
  - DB health checks pass; query SLAs reasonable
- Tasks:
  - Document touched tables (add `docs/database/schema.md`)
  - Add minimal indexes (classification, created_at, title)

### 4) Scrapers
- Deliverables:
  - Inventory and run controls via `/api/v1/scrapers/*`
  - Latest report parsing, logs surfacing
- Acceptance:
  - Scraper status and performance endpoints return data from latest reports
- Tasks:
  - Ensure report/log paths configured; add examples in docs

### 5) Infrastructure
- Deliverables:
  - Local dev scripts (`scripts/`) verified
  - Optional docker-compose for API + Postgres
  - Basic monitoring/health endpoints documented
- Acceptance:
  - One-command start for local dev
- Tasks:
  - Provide compose file or instructions in `docs/deployment/docker.md`

### 6) Security
- Deliverables:
  - JWT secrets via env; token TTLs configured
  - Security headers verified
  - Rate limits set by env
- Acceptance:
  - Pen-test checklist baseline; headers present
- Tasks:
  - Set `SECRET_KEY`, tighten CORS/hosts for prod

### 7) CI/CD & Quality
- Deliverables:
  - Pipeline: lint, tests, build, docs link check
  - Artifacts: OpenAPI export
- Acceptance:
  - Green pipeline; docs links valid
- Tasks:
  - Add docs link checker; publish OpenAPI to `dist/`

## Milestones and timeline
- Phase 0 — Documentation sync (DONE)
- Phase 1 — Backend API verification
  - Verify endpoints, add response models; smoke tests
- Phase 2 — Frontend integration
  - API client, pages, role-based routes
- Phase 3 — Scrapers and data
  - Status/perf wired; logs and reports visible
- Phase 4 — Infra & security hardening
  - Compose (optional), secrets, CORS/hosts, monitoring
- Phase 5 — CI/CD and release
  - Docs check, OpenAPI publish, tag v1.0.1

## Coordination matrix (excerpt)
- Policies pages ↔ `/api/v1/policies` (DB: `bills_bill`)
- Dashboard ↔ `/api/v1/dashboard/*` (DB size, counts) + `/api/v1/health/*`
- Scraper admin ↔ `/api/v1/scrapers/*` (reports/logs on FS)
- Admin ops ↔ `/api/v1/admin/*`

## Execution checklist (agent)
1. Read `docs/api/overview.md` and `docs/api/endpoints.md`
2. Configure `.env` based on `env.example` (secret key, DB URL, CORS)
3. Start API; open `/docs` to validate OpenAPI
4. Run smoke tests against all endpoints; record results
5. Update `docs/api/schemas.md` with any payload changes
6. Implement frontend API client; wire pages to endpoints
7. Validate DB connectivity and create/verify indexes for hot queries
8. Verify scraper reports/logs are discoverable by API
9. Configure CORS/hosts for target environment
10. Add CI step to export OpenAPI and check docs links

## Change control
- Any API change must update `docs/api/endpoints.md` and examples in `docs/api/schemas.md`
- Any networking/config change must update `docs/api/overview.md`
- All changes must pass docs link checker before merge

## Risks & mitigations
- Legacy docs causing confusion → Consolidation plan; archive old files
- Hardcoded secrets → Mandate env-based configuration
- DB query performance → Index review and slow query log

## Acceptance criteria for completion
- All endpoints verifiable and documented; frontend consumes them
- Docs index has no broken links; single source of truth enforced
- CI green with docs check; OpenAPI exported

# Scraper Execution Plan (Master)

## Categories & Triggers
- Federal (parliamentary): daily at 06:00; outputs: bills, representatives, committees; DB: openpolicy_scrapers
- Provincial: weekly Monday 07:00; outputs: representatives; DB: openpolicy_scrapers
- City (municipal): monthly 1st 08:00; outputs: representatives; DB: openpolicy_scrapers
- Civic: daily 06:15; outputs: datasets; DB: openpolicy_scrapers
- Update (maintenance): continuous 2h; outputs: housekeeping; DB: openpolicy_scrapers

## Execution
- Container: `scraper-runner` executes `background_scraper_execution.py`
- Logs: `backend/OpenPolicyAshBack/logs/*`
- Reports: `scraper_test_report_*.json`

## Output Schema (summary)
- representatives(name, role, party, district, email, phone, image, source)
- bills(identifier, title, status, introduced_date, last_action)
- committees(name, jurisdiction, members[])

## Dependencies
- Postgres (openpolicy_scrapers)
- Network access to source sites

## Health
- API `/api/v1/health/scrapers` reads latest report and aggregates status

## Canonical Run Plan

See `backend/config/scraper_plan.py` for triggers, expected runtimes, dependencies, and outputs per category.